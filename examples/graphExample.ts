import { randomUUID } from "node:crypto";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { ChatQVAC } from "../src/index.js";
import { CALCULATOR_TOOLS } from "./calculatorTools.js";
import { createQvacCompletion } from "./qvacComplete.js";

const SYSTEM_PROMPT =
  "You are a calculator assistant. Never do arithmetic yourself: call the add or subtract tool for every operation the question asks for, all of them in a single turn, then answer using the tool results.";
const QUESTION = "What is 17 plus 25, and what is 100 minus 37?";

/**
 * Pipeline flow (state = the running message list, each node appends to it):
 *
 *   START
 *     |
 *     v
 *   +------+  reply has tool_calls  +-------+       +--------+
 *   | plan | ---------------------> | tools | ----> | answer | ----> END
 *   +------+                        +-------+       +--------+
 *     |
 *     | reply has no tool_calls (answered directly)
 *     v
 *    END
 *
 *   plan   - ChatQVAC with the tools bound asks for every `add`/`subtract` call at once
 *   tools  - ToolNode runs all requested calls in parallel
 *   answer - ChatQVAC without tools writes the reply from the tool results
 *
 * No edge leads back, so the graph is a DAG: a run makes at most two model
 * calls and needs no recursion limit. That fits when every operation is known
 * up front, as with the two independent operations here. When one result
 * decides the next operation, use a bounded cycle instead (`loopExample.ts`).
 * `plainExample.ts` codes the same flow without LangGraph.
 */
async function main(): Promise<void> {
  const { complete, shutdown } = await createQvacCompletion();

  try {
    // One KV cache session per run: `answer` only evaluates what `plan` and `tools` added.
    const sessionId = randomUUID();
    const model = new ChatQVAC({ complete, temperature: 0 });
    const planner = model.bindTools(CALCULATOR_TOOLS, { sessionId });
    const answerer = model.withConfig({ sessionId });

    // Each node forwards `config`, so callbacks, streaming and abort signals
    // reach the model on runtimes without async context propagation (Bare, Expo).
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("plan", async (state, config) => ({
        messages: [await planner.invoke(state.messages, config)],
      }))
      .addNode("tools", new ToolNode(CALCULATOR_TOOLS))
      .addNode("answer", async (state, config) => ({
        messages: [await answerer.invoke(state.messages, config)],
      }))
      .addEdge(START, "plan")
      .addConditionalEdges("plan", toolsCondition, { tools: "tools", [END]: END })
      .addEdge("tools", "answer")
      .addEdge("answer", END)
      .compile();

    const result = await graph.invoke({
      messages: [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(QUESTION)],
    });

    console.log(`Q: ${QUESTION}`);
    console.log(`A: ${result.messages.at(-1)?.text.trim()}`);
  } finally {
    await shutdown();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
