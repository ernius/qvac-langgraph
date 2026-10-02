# qvac-langgraph

A LangChain chat model (`ChatQVAC`) for local [QVAC](https://www.npmjs.com/package/@qvac/sdk) completions, so QVAC models can drive LangChain runnables and LangGraph graphs.

The package is only the adapter. It does not load models, stop generations or recover from provider failures: you hand it one function that runs a chat completion, and it does the LangChain <-> QVAC translation around that function. It forwards the caller's abort signal so that function can stop the generation.

## Install

Consumers need `@langchain/core` and `@qvac/sdk` installed alongside it - see [Compatibility](#compatibility) for the versions.

```ts
import { ChatQVAC } from "@space-uy/qvac-langgraph";
```

## Compatibility

The package targets specific versions of the libraries it connects to. Both are required peer dependencies, so npm and pnpm report a missing or out-of-range install.

| Library | Peer range | Tested with | Used for |
| --- | --- | --- | --- |
| `@langchain/core` | `^1.2.12` | 1.2.12 | `BaseChatModel`, messages, tool conversion. |
| `@qvac/sdk` | `0.18.2` | 0.18.2 | Types only (`Tool`, `ToolCall`, `CompletionStats`); the adapter never calls the SDK. |

- `@qvac/sdk` covers `0.18.2` only
- `@langchain/langgraph` is **not** a peer dependency: the adapter never imports it, so any LangGraph release that works with your `@langchain/core` can run `ChatQVAC`. It is only a dev dependency for the examples (`^1.4.16`, tested with 1.4.17).

## Usage

`ChatQVAC` takes a `complete` function (`QvacChatCompletionFn`). Whatever runs your model provides it, for example a thin wrapper over the SDK's `completion()`:

```ts
import { ChatQVAC, type QvacChatCompletionFn } from "@space-uy/qvac-langgraph";
import { HumanMessage } from "@langchain/core/messages";

// `runMyModel` stands in for your engine call (see examples/qvacComplete.ts
// for a real one). It returns a `QvacChatCompletionResult` and calls
// `onToken(delta)` for each text delta when `onToken` is provided.
const complete: QvacChatCompletionFn = (request, onToken) => runMyModel(request, onToken);

const model = new ChatQVAC({ complete, temperature: 0 });
const reply = await model.invoke([new HumanMessage("Hello")]);
console.log(reply.text);
```

It is a normal LangChain chat model, so `invoke`, `stream`, `bindTools` and LangGraph nodes all work.

For an example `@qvac/sdk` implementation of `complete` check the [example one](./examples/qvacComplete.ts) provided as an usage example.

### The `complete` contract

```ts
type QvacChatCompletionFn = (
  request: QvacChatCompletionRequest,
  onToken?: (textDelta: string) => void,
) => Promise<QvacChatCompletionResult>;
```

| Request field | Meaning |
| --- | --- |
| `history` | Chat turns as `{ role, content, images? }`. |
| `tools` | SDK `Tool[]`, present after `bindTools`. |
| `temperature`, `seed` | Sampling settings. |
| `sessionId` | Per-request KV cache key, passed through for you to use (e.g. as the SDK's `kvCache`). |
| `requestId` | Caller-assigned id for this call, passed through for you to use. |
| `responseFormat` | Structured-output constraint in the SDK's `responseFormat` shape. Cannot be combined with `tools`. |
| `signal` | `AbortSignal` that fires when the caller gives up (LangChain `signal`/`timeout`, an aborted LangGraph run). |

| Result field | Meaning |
| --- | --- |
| `text` | The reply text. |
| `toolCalls` | SDK `ToolCall[]` (`id`, `name`, `arguments`). |
| `thinkingText` | Optional reasoning trace. |
| `stats` | Optional SDK `CompletionStats`. |

`onToken` is only passed when the caller streams (`model.stream(...)`). Even then, the returned promise must resolve with the full result.

The adapter rejects as soon as `signal` aborts, but the generation keeps running unless `complete` stops it. With the SDK, call `cancel({ requestId: run.requestId })` on abort, as [`examples/qvacComplete.ts`](./examples/qvacComplete.ts) does, so the local model is free for the next call.

### Tools

```ts
const modelWithTools = model.bindTools([myLangChainTool]);
```

LangChain tools are converted to the SDK's flat `Tool` shape. The completion engine only accepts primitive properties, so only each property's `type`, `description` and `enum` are kept; nested object or array item schemas are dropped. The SDK takes one type per property: a nullable property keeps its non-null type, and a union keeps its first member's type. Tool calls in the reply come back as regular LangChain `tool_calls`.

### Structured output

`withStructuredOutput` works through tool calling, so it shares the flat-schema limit above:

```ts
const route = await model.withStructuredOutput(z.object({ next: z.enum(["search", "answer"]) })).invoke(messages);
```

For a nested schema, such as a planner node that returns a list of steps, pass `responseFormat` instead. The SDK turns the JSON schema into a grammar, so the reply always parses:

```ts
import { createContentParser } from "@langchain/core/language_models/structured_output";
import { toJsonSchema } from "@langchain/core/utils/json_schema";

const planner = model
  .withConfig({
    responseFormat: { type: "json_schema", json_schema: { name: "plan", schema: toJsonSchema(Plan) } },
  })
  .pipe(createContentParser(Plan));
```

Your `complete` must forward `request.responseFormat` to the SDK's `completion()`.

### Per-call options

Beyond the usual LangChain call options, `invoke`/`stream` accept:

```ts
await model.invoke(messages, { temperature: 0.2, seed: 42, sessionId: "chat-123", requestId: "req-1" });
```

`temperature` overrides the constructor default. These, `responseFormat` and the `signal` are forwarded in the request.

### What comes back

- **Thinking trace:** `reply.additional_kwargs.thinkingText`.
- **Stats:** `llmOutput.stats` from `generate`, or `generationInfo.stats` on the last streamed chunk.
- **Streaming:** text deltas arrive as content chunks; one final empty chunk carries `tool_calls`, `thinkingText` and stats.

### Messages and images

- Roles map as system -> `system`, human -> `user`, ai -> `assistant`, tool -> `tool`.
- An AI message that made tool calls is replayed to the model as a JSON `{ text?, tool_calls }` string, with each call in the SDK's `{ id, name, arguments }` shape, since `QvacChatMessage` has no tool-call field.
- A tool result is sent as a JSON `{ tool_call_id, name?, content }` string, so the model can match parallel results to the calls that asked for them.
- Image content blocks with `image/jpeg` or `image/png` become `images` (bytes) on the message; other image types are dropped. Turning those bytes into whatever your engine accepts is up to `complete`.

## In LangGraph

- **Prefer a DAG when the steps are known up front.** If one model turn can request every tool call, wire `plan -> tools -> answer` with no edge back. A run then makes at most two model calls and needs no recursion limit. Prompt the model to request all independent calls in a single turn, or the DAG drops the ones it did not ask for.
- **Bound the cycle when a result decides the next step.** Keep the `agent <-> tools` loop, but count agent turns in the state and route to a tool-less answer node when the budget runs out. Otherwise the default recursion limit allows 13 model calls before a `GraphRecursionError`.
- **Forward `config` in every node.** Call `model.invoke(state.messages, config)` so callbacks, streaming and abort signals reach the model on runtimes without async context propagation, such as Bare or Expo.
- **Limit parallel model calls.** Fan-out branches (`Send`, parallel edges) all run at once, and with one local model they compete for it. Pass `{ maxConcurrency: 1 }` to `invoke`/`stream`, or queue inside `complete`.
- **Reuse the KV cache within a run.** Pass one `sessionId` per run, for example with `bindTools(tools, { sessionId })`, and use it as the SDK's `kvCache` key. Each turn then only evaluates the new messages. The SDK trusts the cached prefix, so switch to a new key when you trim the history or resume from an older checkpoint.

## Examples

The [`examples/`](./examples) folder runs a small calculator agent (an `add` and a `subtract` tool) against a local Qwen3 model:

```bash
npm run example:graph   # LangGraph DAG: plan -> tools -> answer
npm run example:loop    # LangGraph bounded cycle, for steps that depend on earlier results
npm run example:plain   # the DAG as plain code over `complete`, no LangGraph/LangChain
```

- [`examples/qvacComplete.ts`](./examples/qvacComplete.ts) - a working `complete` built on the `@qvac/sdk` SDK's `completion()`, with cancellation, KV cache sessions and `responseFormat`.
- [`examples/graphExample.ts`](./examples/graphExample.ts) - the acyclic pipeline as a graph, with a text diagram of the flow.
- [`examples/loopExample.ts`](./examples/loopExample.ts) - a question whose second operation needs the first result, as a cycle bounded by a turn budget.
- [`examples/plainExample.ts`](./examples/plainExample.ts) - the same pipeline as `graphExample.ts`, written by hand, with a comment mapping each graph piece to its plain-code counterpart.

See [`examples/README.md`](./examples/README.md) for details. The first run downloads the model.

## Development

```bash
npm ci
npm run build
npm test
```

`vite` is a direct dev dependency, not only vitest's peer, because npm otherwise skips rolldown's native binary on `npm ci` ([npm/cli#4828](https://github.com/npm/cli/issues/4828)) and `npm test` fails to start.
