import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type BaseChatModelParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import type { Runnable } from "@langchain/core/runnables";
import type { Tool, ToolCall as QvacToolCall } from "@qvac/sdk";
import type {
  QvacChatCompletionFn,
  QvacChatCompletionRequest,
  QvacChatCompletionResult,
  QvacChatImageAttachment,
  QvacChatMessage,
  QvacResponseFormat,
  QvacSupportedImageMimeType,
} from "./types.js";

const QVAC_ROLE_BY_MESSAGE_TYPE: Record<string, string> = {
  system: "system",
  human: "user",
  ai: "assistant",
  tool: "tool",
};

type QvacToolProperty = Tool["parameters"]["properties"][string];
type QvacToolPropertyType = QvacToolProperty["type"];

interface JsonSchemaProperty {
  type?: string | string[];
  anyOf?: Array<{ type?: string | string[] }>;
  description?: string;
  enum?: unknown[];
}

interface JsonSchemaObject {
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
}

export interface QVACChatModelInput extends BaseChatModelParams {
  /** Runs one chat completion - see `QvacChatCompletionFn`. */
  complete: QvacChatCompletionFn;
  temperature?: number;
}

export interface ChatQVACCallOptions extends BaseChatModelCallOptions {
  tools?: Tool[];
  /** Per-call override of the constructor's `temperature`/no `seed` default. */
  temperature?: number;
  seed?: number;
  /** Per-call KV cache session key, forwarded in the completion request. */
  sessionId?: string;
  /** Caller-assigned id for this specific generation call, forwarded in the completion request. */
  requestId?: string;
  /** Structured-output constraint, forwarded in the completion request. Cannot be combined with `tools`. */
  responseFormat?: QvacResponseFormat;
}

const QVAC_PROPERTY_TYPES: ReadonlySet<string> = new Set<QvacToolPropertyType>([
  "string",
  "number",
  "integer",
  "boolean",
  "object",
  "array",
]);

function isQvacPropertyType(type: unknown): type is QvacToolPropertyType {
  return typeof type === "string" && QVAC_PROPERTY_TYPES.has(type);
}

/**
 * The SDK accepts a single type per property. A nullable property
 * (`type: ["number", "null"]`) keeps its non-null type, a union (`anyOf`)
 * keeps its first supported member's type, and anything else is `string`.
 */
function toQvacPropertyType(property: JsonSchemaProperty): QvacToolPropertyType {
  const declared = [property.type, ...(property.anyOf ?? []).map((member) => member.type)].flat();
  return declared.find(isQvacPropertyType) ?? "string";
}

/**
 * Converts a LangChain tool definition into the SDK's flat `Tool` shape.
 * The underlying completion engine's tool schema only supports
 * primitive-typed properties (no nested object/array item schemas), so
 * only `type`/`description`/`enum` survive - that's all it accepts.
 * Nested shapes belong in `responseFormat` instead, which takes a full schema.
 */
function toChatTool(tool: BindToolsInput): Tool {
  const { function: fn } = convertToOpenAITool(
    tool as Parameters<typeof convertToOpenAITool>[0],
  );
  const schema = (fn.parameters ?? {}) as JsonSchemaObject;

  const properties: Record<string, QvacToolProperty> = {};
  for (const [key, value] of Object.entries(schema.properties ?? {})) {
    properties[key] = {
      type: toQvacPropertyType(value),
      description: value.description,
      enum: value.enum as QvacToolProperty["enum"],
    };
  }

  return {
    type: "function",
    name: fn.name,
    description: fn.description ?? "",
    parameters: {
      type: "object",
      properties,
      required: schema.required,
    },
  };
}

const SUPPORTED_IMAGE_MIME_TYPES: ReadonlySet<string> = new Set<QvacSupportedImageMimeType>([
  "image/jpeg",
  "image/png",
]);

function isSupportedImageMimeType(mimeType: unknown): mimeType is QvacSupportedImageMimeType {
  return typeof mimeType === "string" && SUPPORTED_IMAGE_MIME_TYPES.has(mimeType);
}

/**
 * Extracts image content blocks from `message` into `QvacChatMessage.images`.
 * Only blocks whose `mimeType` this pipeline actually supports survive -
 * an API boundary further upstream is what enforces that on the way in;
 * this is a defensive filter against a message constructed some other way.
 */
function toChatImages(message: BaseMessage): QvacChatImageAttachment[] | undefined {
  const images: QvacChatImageAttachment[] = [];

  for (const block of message.contentBlocks) {
    if (block.type !== "image") continue;
    if (!("data" in block) || block.data === undefined) continue;
    if (!isSupportedImageMimeType(block.mimeType)) continue;

    images.push({
      mimeType: block.mimeType,
      data: typeof block.data === "string" ? Buffer.from(block.data, "base64") : block.data,
    });
  }

  return images.length > 0 ? images : undefined;
}

function toChatMessage(message: BaseMessage): QvacChatMessage {
  const role = QVAC_ROLE_BY_MESSAGE_TYPE[message.type] ?? "user";

  // Tool-call turns carry no dedicated field in `QvacChatMessage`; serialize
  // them (alongside any accompanying text) in the SDK's own `ToolCall` shape
  // so the model can see its own prior turn when the history is replayed.
  if (AIMessage.isInstance(message) && message.tool_calls?.length) {
    return {
      role,
      content: JSON.stringify({
        ...(message.text ? { text: message.text } : {}),
        tool_calls: message.tool_calls.map(({ id, name, args }) => ({ id, name, arguments: args })),
      }),
    };
  }

  // Likewise there is no tool-call id field: carry it (and the tool name) in
  // the content, so parallel results can be matched to the call that asked.
  if (ToolMessage.isInstance(message)) {
    return {
      role,
      content: JSON.stringify({
        tool_call_id: message.tool_call_id,
        ...(message.name ? { name: message.name } : {}),
        content: message.text,
      }),
    };
  }

  const images = toChatImages(message);
  return { role, content: message.text, ...(images ? { images } : {}) };
}

/** Settles with `promise`, or rejects with the abort reason as soon as `signal` fires. */
function rejectOnAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    promise
      .then(resolve, reject)
      .finally(() => signal.removeEventListener("abort", onAbort));
  });
}

function toLangChainToolCalls(toolCalls: QvacToolCall[]): ToolCall[] {
  return toolCalls.map((call) => ({
    type: "tool_call",
    id: call.id,
    name: call.name,
    args: call.arguments,
  }));
}

/** LangChain chat model that delegates every generation to an injected `complete` function. */
export class ChatQVAC extends BaseChatModel<ChatQVACCallOptions> {
  private readonly completeFn: QvacChatCompletionFn;
  private readonly temperature?: number;

  constructor(fields: QVACChatModelInput) {
    super(fields);
    this.completeFn = fields.complete;
    this.temperature = fields.temperature;
  }

  static lc_name(): string {
    return "ChatQVAC";
  }

  _llmType(): string {
    return "qvac";
  }

  bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<ChatQVACCallOptions>,
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatQVACCallOptions> {
    return this.withConfig({
      tools: tools.map(toChatTool),
      ...kwargs,
    } as Partial<ChatQVACCallOptions>);
  }

  private buildRequest(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
  ): QvacChatCompletionRequest {
    return {
      history: messages.map(toChatMessage),
      tools: options.tools,
      temperature: options.temperature ?? this.temperature,
      seed: options.seed,
      sessionId: options.sessionId,
      requestId: options.requestId,
      responseFormat: options.responseFormat,
      signal: options.signal,
    };
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    options.signal?.throwIfAborted();
    const result = await rejectOnAbort(
      this.completeFn(this.buildRequest(messages, options)),
      options.signal,
    );

    // A chunk, not an `AIMessage`: LangChain's `withStructuredOutput` parser
    // only accepts `AIMessageChunk`, the type `invoke` is declared to return.
    const aiMessage = new AIMessageChunk({
      content: result.text,
      tool_calls: toLangChainToolCalls(result.toolCalls),
      additional_kwargs: result.thinkingText
        ? { thinkingText: result.thinkingText }
        : undefined,
    });

    return {
      generations: [{ text: result.text, message: aiMessage }],
      llmOutput: result.stats ? { stats: result.stats } : undefined,
    };
  }

  /**
   * Bridges `complete`'s callback-based streaming into an async generator:
   * each `onToken` call is queued and yielded as a `ChatGenerationChunk`
   * carrying just its text delta, then one final content-empty chunk carries
   * `tool_calls`/`thinkingText`/`stats` once `complete`'s promise resolves -
   * the same result `_generate` returns, split into deltas plus a trailer.
   */
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    type QueueItem =
      | { kind: "token"; textDelta: string }
      | { kind: "done"; result: QvacChatCompletionResult }
      | { kind: "error"; error: unknown };

    const queue: QueueItem[] = [];
    let notify: (() => void) | undefined;
    const push = (item: QueueItem) => {
      queue.push(item);
      notify?.();
      notify = undefined;
    };

    this.completeFn(this.buildRequest(messages, options), (textDelta) =>
      push({ kind: "token", textDelta }),
    ).then(
      (result) => push({ kind: "done", result }),
      (error: unknown) => push({ kind: "error", error }),
    );

    while (true) {
      const item = queue.shift();
      if (!item) {
        await new Promise<void>((resolve) => {
          notify = resolve;
        });
        continue;
      }

      if (item.kind === "token") {
        await runManager?.handleLLMNewToken(item.textDelta);
        yield new ChatGenerationChunk({
          text: item.textDelta,
          message: new AIMessageChunk({ content: item.textDelta }),
        });
        continue;
      }

      if (item.kind === "error") {
        throw item.error;
      }

      yield new ChatGenerationChunk({
        text: "",
        message: new AIMessageChunk({
          content: "",
          tool_calls: toLangChainToolCalls(item.result.toolCalls),
          additional_kwargs: item.result.thinkingText
            ? { thinkingText: item.result.thinkingText }
            : undefined,
        }),
        generationInfo: item.result.stats
          ? { stats: item.result.stats }
          : undefined,
      });
      return;
    }
  }
}
