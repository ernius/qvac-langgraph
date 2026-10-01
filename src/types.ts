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

export interface QvacChatCompletionRequest {
  history: QvacChatMessage[];
  tools?: Tool[];
  temperature?: number;
  seed?: number;
  /** Per-request KV cache session key, forwarded to whatever runs the completion. */
  sessionId?: string;
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
 * Model loading, cancellation and provider recovery are the implementer's
 * concern, not the adapter's.
 */
export type QvacChatCompletionFn = (
  request: QvacChatCompletionRequest,
  onToken?: (textDelta: string) => void,
) => Promise<QvacChatCompletionResult>;
