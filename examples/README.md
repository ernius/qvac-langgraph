# qvac-langgraph examples

The examples answer arithmetic questions with a small local Qwen3 model that calls an `add` and a `subtract` tool. The first run downloads the model.

| Command | What it shows |
| --- | --- |
| `npm run example:graph` | "What is 17 plus 25, and what is 100 minus 37?" as a LangGraph DAG: a `plan` node requests both calls at once, a `ToolNode` runs them, an `answer` node replies. At most two model calls. |
| `npm run example:loop` | "What is 17 plus 25, minus 37?" as a bounded cycle: the second operation needs the first result, so the agent loops through the tools, with a turn budget that ends in a forced answer. |
| `npm run example:plain` | The same DAG as `example:graph`, as straight-line code over the `complete` function and QVAC's own types - no LangGraph, no LangChain, no `ChatQVAC`. |

Files:

- `qvacComplete.ts` - loads the model and builds the `complete` function `ChatQVAC` needs, on top of the SDK's `completion()`. It cancels the generation on abort, uses `sessionId` as the KV cache key and forwards `responseFormat`.
- `calculatorTools.ts` - the LangChain `add`/`subtract` tools shared by the two graph examples.
- `graphExample.ts` / `loopExample.ts` - the acyclic and the cyclic pipeline, each with a text diagram of its flow.
- `plainExample.ts` - self-contained, with SDK `Tool` objects, and a comment mapping each graph piece to its plain-code counterpart.
