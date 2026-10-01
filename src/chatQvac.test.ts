import { describe, expect, it } from "vitest";
import * as z from "zod";
import { AIMessage, AIMessageChunk, HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { ChatQVAC } from "./chatQvac.js";
import type {
  QvacChatCompletionFn,
  QvacChatCompletionRequest,
  QvacChatCompletionResult,
} from "./types.js";

const CALL_OPTIONS = {} as Parameters<ChatQVAC["_generate"]>[1];

const JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xdb, 0x00, 0x01, 0x02, 0x03]);

/** A fake `complete`: records every request, replays `tokens` through `onToken`, then resolves `result` (or rejects with `failWith`). */
class RecordingCompletion {
  readonly requests: QvacChatCompletionRequest[] = [];
  tokens: string[] = [];
  result: QvacChatCompletionResult = { text: "ok", toolCalls: [] };
  failWith?: Error;

  readonly complete: QvacChatCompletionFn = async (request, onToken) => {
    this.requests.push(request);
    for (const token of this.tokens) onToken?.(token);
    if (this.failWith) throw this.failWith;
    return this.result;
  };

  get lastRequest(): QvacChatCompletionRequest | undefined {
    return this.requests.at(-1);
  }
}

describe("ChatQVAC._generate", () => {
  it("returns the completion's text, tool calls, thinking trace and stats", async () => {
    const completion = new RecordingCompletion();
    completion.result = {
      text: "checking",
      thinkingText: "the user wants stock",
      toolCalls: [{ id: "call-1", name: "lookup_stock", arguments: { sku: "SD-X4-001" } }],
      stats: { tokensPerSecond: 12 },
    };
    const model = new ChatQVAC({ complete: completion.complete });

    const result = await model._generate([new HumanMessage("hi")], CALL_OPTIONS);

    const generation = result.generations[0];
    expect(generation?.text).toBe("checking");
    const message = generation?.message as AIMessage;
    expect(message.tool_calls).toMatchObject([
      { id: "call-1", name: "lookup_stock", args: { sku: "SD-X4-001" } },
    ]);
    expect(message.additional_kwargs).toEqual({ thinkingText: "the user wants stock" });
    expect(result.llmOutput).toEqual({ stats: { tokensPerSecond: 12 } });
  });

  it("propagates a completion failure", async () => {
    const completion = new RecordingCompletion();
    completion.failWith = new Error("boom");
    const model = new ChatQVAC({ complete: completion.complete });

    await expect(model._generate([new HumanMessage("hi")], CALL_OPTIONS)).rejects.toThrow("boom");
  });
});

describe("ChatQVAC._streamResponseChunks", () => {
  it("yields each token as a text delta, then a trailer carrying tool calls, thinking trace and stats", async () => {
    const completion = new RecordingCompletion();
    completion.tokens = ["Hel", "lo"];
    completion.result = {
      text: "Hello",
      thinkingText: "greeting",
      toolCalls: [{ id: "call-1", name: "lookup_stock", arguments: { sku: "SD-X4-001" } }],
      stats: { generatedTokens: 2 },
    };
    const model = new ChatQVAC({ complete: completion.complete });

    const chunks = [];
    for await (const chunk of model._streamResponseChunks([new HumanMessage("hi")], CALL_OPTIONS)) {
      chunks.push(chunk);
    }

    expect(chunks.map((chunk) => chunk.text)).toEqual(["Hel", "lo", ""]);
    const trailer = chunks[2]?.message as AIMessageChunk;
    expect(trailer.tool_calls).toMatchObject([
      { id: "call-1", name: "lookup_stock", args: { sku: "SD-X4-001" } },
    ]);
    expect(trailer.additional_kwargs).toEqual({ thinkingText: "greeting" });
    expect(chunks[2]?.generationInfo).toEqual({ stats: { generatedTokens: 2 } });
  });

  it("still yields the trailer when the completion produced no tokens", async () => {
    const completion = new RecordingCompletion();
    completion.result = {
      text: "",
      toolCalls: [{ id: "call-1", name: "lookup_stock", arguments: { sku: "SD-X4-001" } }],
    };
    const model = new ChatQVAC({ complete: completion.complete });

    const chunks = [];
    for await (const chunk of model._streamResponseChunks([new HumanMessage("hi")], CALL_OPTIONS)) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(1);
    expect((chunks[0]?.message as AIMessageChunk).tool_calls).toHaveLength(1);
  });

  it("rejects the consumer when the completion fails, instead of hanging", async () => {
    const completion = new RecordingCompletion();
    completion.tokens = ["par"];
    completion.failWith = new Error("boom");
    const model = new ChatQVAC({ complete: completion.complete });

    await expect(async () => {
      for await (const _chunk of model._streamResponseChunks([new HumanMessage("hi")], CALL_OPTIONS)) {
        // draining the generator
      }
    }).rejects.toThrow("boom");
  });
});

describe("ChatQVAC image forwarding", () => {
  it("forwards an image content block as ChatMessage.images", async () => {
    const completion = new RecordingCompletion();
    const model = new ChatQVAC({ complete: completion.complete });

    await model.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: "what's wrong with this part?" },
          { type: "image", mimeType: "image/jpeg", data: JPEG_BYTES },
        ],
      }),
    ]);

    const history = completion.lastRequest?.history ?? [];
    const humanEntry = history.find((entry) => entry.role === "user");
    expect(humanEntry?.content).toBe("what's wrong with this part?");
    expect(humanEntry?.images).toEqual([{ mimeType: "image/jpeg", data: JPEG_BYTES }]);
  });

  it("omits images for a text-only message", async () => {
    const completion = new RecordingCompletion();
    const model = new ChatQVAC({ complete: completion.complete });

    await model.invoke([new HumanMessage("hello")]);

    expect(completion.lastRequest?.history[0]?.images).toBeUndefined();
  });
});

describe("ChatQVAC per-call temperature/seed/sessionId", () => {
  it("uses the per-call temperature/seed override instead of the constructor default", async () => {
    const completion = new RecordingCompletion();
    const model = new ChatQVAC({ complete: completion.complete, temperature: 0 });

    await model.invoke([new HumanMessage("hi")], { temperature: 0.9, seed: 123, sessionId: "s-1" });

    expect(completion.lastRequest?.temperature).toBe(0.9);
    expect(completion.lastRequest?.seed).toBe(123);
    expect(completion.lastRequest?.sessionId).toBe("s-1");
  });

  it("falls back to the constructor default temperature when no override is given", async () => {
    const completion = new RecordingCompletion();
    const model = new ChatQVAC({ complete: completion.complete, temperature: 0.4 });

    await model.invoke([new HumanMessage("hi")]);

    expect(completion.lastRequest?.temperature).toBe(0.4);
    expect(completion.lastRequest?.seed).toBeUndefined();
  });
});

describe("ChatQVAC.bindTools", () => {
  it("flattens a tool's primitive-typed properties into the SDK Tool shape", async () => {
    const completion = new RecordingCompletion();
    const model = new ChatQVAC({ complete: completion.complete });

    const lookupSkuTool = tool(() => "unused", {
      name: "lookup_sku",
      description: "Look up a SKU in the inventory",
      schema: z.object({
        sku: z.string().describe("The SKU to look up"),
        limit: z.number().optional(),
      }),
    });

    await model.bindTools([lookupSkuTool]).invoke([new HumanMessage("hi")]);

    expect(completion.lastRequest?.tools).toEqual([
      {
        type: "function",
        name: "lookup_sku",
        description: "Look up a SKU in the inventory",
        parameters: {
          type: "object",
          properties: {
            sku: { type: "string", description: "The SKU to look up", enum: undefined },
            limit: { type: "number", description: undefined, enum: undefined },
          },
          required: ["sku"],
        },
      },
    ]);
  });
});
