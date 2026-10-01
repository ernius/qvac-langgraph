# qvac-langgraph examples

Both examples answer "What is 17 plus 25, and what is 100 minus 37?" with a small local Qwen3 model that calls an `add` and a `subtract` tool. The first run downloads the model.

| Command | What it shows |
| --- | --- |
| `npm run example:graph` | The pipeline declared as a LangGraph `StateGraph` (agent node, `ToolNode`, conditional edge). |
| `npm run example:plain` | The same pipeline as a plain loop over the `complete` function and QVAC's own types - no LangGraph, no LangChain, no `ChatQVAC`. |

Files:

- `qvacComplete.ts` - loads the model and builds the `complete` function `ChatQVAC` needs, on top of the SDK's `completion()`.
- `graphExample.ts` / `plainExample.ts` - the two pipelines, each self-contained with its own tools (LangChain `tool()`s in the graph, SDK `Tool` objects in the plain loop). `plainExample.ts` has a comment mapping each graph piece to its plain-code counterpart.
