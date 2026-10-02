import type { CompletionStats, Tool, ToolCall } from "@qvac/sdk";

/**
 * Request/result value types the adapter exchanges with whatever runs the
 * completion. A consumer's own types (e.g. a backend's `ChatCompletionRequest`
 * / `ChatCompletionResult`) can be used here as-is via TypeScript's structural
 * typing, as long as their shape matches - no adapter object or cast required.
 */

/** Closed set of image formats forwarded as chat attachments. */
export type QvacSupportedImageMimeType = "image/jpeg" | "image/png";

export interface QvacChatImageAttachment {
  mimeType: QvacSupportedImageMimeType;
  data: Uint8Array;
}

/** One turn of chat history, engine-agnostic. */
export interface QvacChatMessage {
  role: string;
  content: string;
  images?: QvacChatImageAttachment[];
}

/**
 * Structured-output constraint, mirroring the SDK's `completion({ responseFormat })`
 * (not exported by `@qvac/sdk`, so declared here with the same shape). The SDK
 * turns `json_schema` into a grammar, so the reply is guaranteed to parse. It
 * cannot be combined with `tools`.
 */
export type QvacResponseFormat =
  | { type: "text" }
  | { type: "json_object" }
  | {
      type: "json_schema";
      json_schema: {
        name: string;
        description?: string;
        schema: Record<string, unknown>;
        strict?: boolean;
      };
    };

export interface QvacChatCompletionRequest {
  history: QvacChatMessage[];
  tools?: Tool[];
  temperature?: number;
  seed?: number;
  /** Per-request KV cache session key, forwarded to whatever runs the completion. */
  sessionId?: string;
  /** Caller-assigned id for this generation call, forwarded to whatever runs the completion - lets it track/cancel this call independently of any other in flight. */
  requestId?: string;
  /** Structured-output constraint for this call - see `QvacResponseFormat`. */
  responseFormat?: QvacResponseFormat;
  /**
   * Fires when the caller gives up on this call (LangChain `signal`/`timeout`,
   * a LangGraph run being aborted). The adapter already rejects on abort; the
   * implementer should also stop the generation, e.g. `cancel({ requestId })`.
   */
  signal?: AbortSignal;
}

export interface QvacChatCompletionResult {
  text: string;
  thinkingText?: string;
  toolCalls: ToolCall[];
  stats?: CompletionStats;
}

/**
 * The single dependency `ChatQVAC` needs: run one chat completion. Pass
 * `onToken` to receive text deltas as they are generated; the promise still
 * resolves with the full result (text, tool calls, thinking trace, stats).
 * Model loading, stopping a generation on `request.signal` and provider
 * recovery are the implementer's concern, not the adapter's.
 */
export type QvacChatCompletionFn = (
  request: QvacChatCompletionRequest,
  onToken?: (textDelta: string) => void,
) => Promise<QvacChatCompletionResult>;
