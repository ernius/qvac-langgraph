import { randomUUID } from "node:crypto";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import {
  Annotation,
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
  "You are a calculator assistant. Never do arithmetic yourself: call the add or subtract tool. When an operation needs the result of another, wait for that result before calling the next tool.";
const BUDGET_EXHAUSTED_PROMPT =
  "The tool budget is used up. Answer now with the tool results you already have.";
const QUESTION = "What is 17 plus 25, minus 37?";
/** Model turns the agent may spend before it must answer. */
const MAX_AGENT_TURNS = 4;

/** The message list plus a counter of agent turns, which bounds the cycle. */
const BudgetedState = Annotation.Root({
  ...MessagesAnnotation.spec,
  agentTurns: Annotation<number>({
    reducer: (total, added) => total + added,
    default: () => 0,
  }),
});

function routeAgentReply(state: typeof BudgetedState.State): "tools" | "answer" | typeof END {
  if (toolsCondition(state) === END) return END;
  return state.agentTurns < MAX_AGENT_TURNS ? "tools" : "answer";
}

/**
 * Pipeline flow for a question whose second operation needs the first result:
 *
 *   START
 *     |
 *     v
 *   +-------+  tool_calls, budget left  +-------+
 *   | agent | ------------------------> | tools |
 *   +-------+ <------------------------ +-------+
 *     |   |      ToolMessage results
 *     |   | tool_calls, budget used up  +--------+
 *     |   +---------------------------> | answer | ----> END
 *     |                                 +--------+
 *     | reply has no tool_calls
 *     v
 *    END
 *
 * The cycle is needed here: the model cannot ask for `subtract(42, 37)`
 * before it knows 42. The `agentTurns` counter keeps it bounded, so a model
 * that keeps calling tools ends in a forced `answer` instead of a
 * `GraphRecursionError`. For operations known up front, the acyclic graph
 * in `graphExample.ts` makes fewer model calls.
 */
async function main(): Promise<void> {
  const { complete, shutdown } = await createQvacCompletion();

  try {
    const sessionId = randomUUID();
    const model = new ChatQVAC({ complete, temperature: 0 });
    const agentModel = model.bindTools(CALCULATOR_TOOLS, { sessionId });
    const answerer = model.withConfig({ sessionId });

    const graph = new StateGraph(BudgetedState)
      .addNode("agent", async (state, config) => ({
        messages: [await agentModel.invoke(state.messages, config)],
        agentTurns: 1,
      }))
      .addNode("tools", new ToolNode(CALCULATOR_TOOLS))
      .addNode("answer", async (state, config) => ({
        messages: [
          await answerer.invoke([...state.messages, new HumanMessage(BUDGET_EXHAUSTED_PROMPT)], config),
        ],
      }))
      .addEdge(START, "agent")
      .addConditionalEdges("agent", routeAgentReply, ["tools", "answer", END])
      .addEdge("tools", "agent")
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
