import * as z from "zod";
import { tool } from "@langchain/core/tools";

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

export const CALCULATOR_TOOLS = [addTool, subtractTool];
