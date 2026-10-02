import {
  cancel,
  close,
  completion,
  deleteCache,
  loadModel,
  QWEN3_600M_INST_Q4,
  unloadModel,
} from "@qvac/sdk";
import type { QvacChatCompletionFn } from "../src/index.js";

export interface QvacCompletion {
  complete: QvacChatCompletionFn;
  shutdown: () => Promise<void>;
}

/**
 * The one thing `ChatQVAC` needs from its host: a function that runs a chat
 * completion. Here it is a thin wrapper over the SDK's `completion()`; the
 * model is loaded once up front (downloaded on the first run).
 * Text only - image attachments on a message are ignored.
 *
 * - `sessionId` becomes the SDK's `kvCache` key, so each turn of a graph run
 *   only evaluates the messages added since the previous turn. The SDK trusts
 *   the cached prefix, so use a fresh key whenever the history is rewritten
 *   (trimmed, or resumed from an older checkpoint).
 * - `signal` cancels the generation itself, so an aborted or timed-out run
 *   frees the model for the next call instead of decoding to the end.
 */
export async function createQvacCompletion(): Promise<QvacCompletion> {
  const modelId = await loadModel({
    modelSrc: QWEN3_600M_INST_Q4.src,
    modelType: "llamacpp-completion",
    modelConfig: { ctx_size: 4096, tools: true },
  });
  const sessionIds = new Set<string>();

  const complete: QvacChatCompletionFn = async (request, onToken) => {
    if (request.sessionId) sessionIds.add(request.sessionId);

    const run = completion({
      modelId,
      history: request.history.map(({ role, content }) => ({ role, content })),
      tools: request.tools,
      responseFormat: request.responseFormat,
      kvCache: request.sessionId,
      captureThinking: true,
      stream: Boolean(onToken),
      generationParams: { temp: request.temperature, seed: request.seed },
    });

    // A cancel that races the request start is a no-op on the SDK side, so its failure is safe to ignore.
    const stop = () => {
      cancel({ requestId: run.requestId }).catch(() => undefined);
    };
    request.signal?.addEventListener("abort", stop, { once: true });

    try {
      if (onToken) {
        for await (const event of run.events) {
          if (event.type === "contentDelta") onToken(event.text);
        }
      }

      const final = await run.final;
      return {
        text: final.contentText,
        thinkingText: final.thinkingText,
        toolCalls: final.toolCalls,
        stats: final.stats,
      };
    } finally {
      request.signal?.removeEventListener("abort", stop);
    }
  };

  const shutdown = async (): Promise<void> => {
    for (const kvCacheKey of sessionIds) {
      await deleteCache({ kvCacheKey, modelId });
    }
    await unloadModel({ modelId, clearStorage: false });
    await close();
  };

  return { complete, shutdown };
}
