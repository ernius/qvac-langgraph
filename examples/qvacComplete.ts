import {
  close,
  completion,
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
 */
export async function createQvacCompletion(): Promise<QvacCompletion> {
  const modelId = await loadModel({
    modelSrc: QWEN3_600M_INST_Q4.src,
    modelType: "llamacpp-completion",
    modelConfig: { ctx_size: 4096, tools: true },
  });

  const complete: QvacChatCompletionFn = async (request, onToken) => {
    const run = completion({
      modelId,
      history: request.history.map(({ role, content }) => ({ role, content })),
      tools: request.tools,
      captureThinking: true,
      stream: Boolean(onToken),
      generationParams: { temp: request.temperature, seed: request.seed },
    });

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
  };

  const shutdown = async (): Promise<void> => {
    await unloadModel({ modelId, clearStorage: false });
    await close();
  };

  return { complete, shutdown };
}
