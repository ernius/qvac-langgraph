import * as z from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { ChatQVAC } from "../src/index.js";
import { createQvacCompletion } from "./qvacComplete.js";

const SYSTEM_PROMPT =
  "You are a calculator assistant. Never do arithmetic yourself: always call the add or subtract tool, one operation at a time, and use the tool results.";
const QUESTION = "What is 17 plus 25, and what is 100 minus 37?";

/** Plain LangChain tools - nothing QVAC-specific; `ChatQVAC.bindTools` converts them to the SDK's `Tool` shape. */
const addTool = tool(
  ({ a, b }) => {
    console.log(`[tool] add(${a}, ${b})`);
    return String(a + b);
  },
  {
    name: "add",
    description: "Adds two numbers and returns their sum.",
    schema: z.object({
      a: z.number().describe("The first number"),
      b: z.number().describe("The second number"),
    }),
  },
);

const subtractTool = tool(
  ({ a, b }) => {
    console.log(`[tool] subtract(${a}, ${b})`);
    return String(a - b);
  },
  {
    name: "subtract",
    description: "Subtracts the second number from the first and returns the difference.",
    schema: z.object({
      a: z.number().describe("The number to subtract from"),
      b: z.number().describe("The number to subtract"),
    }),
  },
);

const TOOLS = [addTool, subtractTool];

/**
 * Pipeline flow (state = the running message list, each node appends to it):
 *
 *   START
 *     |
 *     v
 *   +-------+   reply has tool_calls    +-------+
 *   | agent | ------------------------> | tools |
 *   +-------+ <------------------------ +-------+
 *     |         ToolMessage results
 *     | reply has no tool_calls
 *     v
 *    END
 *
 *   agent - ChatQVAC (the model) answers, or asks to call `add`/`subtract`
 *   tools - ToolNode runs each requested `add`/`subtract` call
 *   edge  - toolsCondition picks "tools" or END after every agent turn
 *
 * For "What is 17 plus 25, and what is 100 minus 37?" the two operations are
 * independent, so the model requests both in one turn:
 * agent -> tools (add 17, 25 = 42 and subtract 100, 37 = 63) -> agent -> END.
 * `plainExample.ts` codes the same flow as a loop, without LangGraph.
 */
async function main(): Promise<void> {
  const { complete, shutdown } = await createQvacCompletion();

  try {
    const model = new ChatQVAC({ complete, temperature: 0 }).bindTools(TOOLS);

    // The pipeline is declared as a graph: state (the message list), two
    // nodes, and a conditional edge that loops back until no tool is called.
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("agent", async (state) => ({
        messages: [await model.invoke(state.messages)],
      }))
      .addNode("tools", new ToolNode(TOOLS))
      .addEdge(START, "agent")
      .addConditionalEdges("agent", toolsCondition, ["tools", END])
      .addEdge("tools", "agent")
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
