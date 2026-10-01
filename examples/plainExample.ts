import type { Tool } from "@qvac/sdk";
import type { QvacChatMessage } from "../src/index.js";
import { createQvacCompletion } from "./qvacComplete.js";

const SYSTEM_PROMPT =
  "You are a calculator assistant. Never do arithmetic yourself: always call the add or subtract tool, one operation at a time, and use the tool results.";
const QUESTION = "What is 17 plus 25, and what is 100 minus 37?";
const MAX_STEPS = 5;

const numberParameter = (description: string) =>
  ({ type: "number", description }) as const;

/** Tool definitions in the SDK's own `Tool` shape - what the model is shown. */
const TOOLS: Tool[] = [
  {
    type: "function",
    name: "add",
    description: "Adds two numbers and returns their sum.",
    parameters: {
      type: "object",
      properties: { a: numberParameter("The first number"), b: numberParameter("The second number") },
      required: ["a", "b"],
    },
  },
  {
    type: "function",
    name: "subtract",
    description: "Subtracts the second number from the first and returns the difference.",
    parameters: {
      type: "object",
      properties: {
        a: numberParameter("The number to subtract from"),
        b: numberParameter("The number to subtract"),
      },
      required: ["a", "b"],
    },
  },
];

/** The matching implementations, looked up by the tool name the model returns. */
const IMPLEMENTATIONS: Record<string, (a: number, b: number) => number> = {
  add: (a, b) => a + b,
  subtract: (a, b) => a - b,
};

function runTool(name: string, args: Record<string, unknown>): string {
  const implementation = IMPLEMENTATIONS[name];
  if (!implementation) throw new Error(`Model called unknown tool "${name}"`);
  if (typeof args.a !== "number" || typeof args.b !== "number") {
    throw new Error(`Tool "${name}" needs numeric a and b, got ${JSON.stringify(args)}`);
  }
  console.log(`[tool] ${name}(${args.a}, ${args.b})`);
  return String(implementation(args.a, args.b));
}

/**
 * The same pipeline as `graphExample.ts`, with neither LangGraph nor
 * LangChain - just the `complete` function and a loop over QVAC's own types:
 *
 *   graph state (MessagesAnnotation)  ->  the `history` array
 *   "agent" node (ChatQVAC)           ->  `complete({ history, tools })`
 *   "tools" node (ToolNode)           ->  `runTool` for each `reply.toolCalls` entry
 *   conditional edge (toolsCondition) ->  the `toolCalls.length === 0` exit
 *   tools -> agent edge               ->  the next loop iteration
 *
 * `ChatQVAC` exists to make this loop unnecessary to write by hand when you
 * are inside LangChain: it converts LangChain messages and tools to these
 * QVAC shapes and back. And for a single agent-with-tools loop, this is all
 * you need; LangGraph earns its keep once you want checkpointing/resume,
 * streaming modes, human approval steps, or several branching nodes.
 */
async function main(): Promise<void> {
  const { complete, shutdown } = await createQvacCompletion();

  try {
    const history: QvacChatMessage[] = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: QUESTION },
    ];

    // MAX_STEPS guards against a model that keeps calling tools forever.
    for (let step = 0; step < MAX_STEPS; step++) {
      const reply = await complete({ history, tools: TOOLS, temperature: 0 });

      if (reply.toolCalls.length === 0) {
        console.log(`Q: ${QUESTION}`);
        console.log(`A: ${reply.text.trim()}`);
        return;
      }

      // Replay the model's own tool-call turn so it sees what it asked for,
      // then one "tool" turn per result.
      history.push({
        role: "assistant",
        content: JSON.stringify({ tool_calls: reply.toolCalls }),
      });
      for (const toolCall of reply.toolCalls) {
        history.push({ role: "tool", content: runTool(toolCall.name, toolCall.arguments) });
      }
    }

    throw new Error(`No final answer after ${MAX_STEPS} steps`);
  } finally {
    await shutdown();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
