# qvac-langgraph

A LangChain chat model (`ChatQVAC`) for local [QVAC](https://www.npmjs.com/package/@qvac/sdk) completions, so QVAC models can drive LangChain runnables and LangGraph graphs.

The package is only the adapter. It does not load models, cancel requests or recover from provider failures: you hand it one function that runs a chat completion, and it does the LangChain <-> QVAC translation around that function.

## Install

This package lives in the repo's npm workspace. Consumers need `@langchain/core` and `@qvac/sdk` installed alongside it - see [Compatibility](#compatibility) for the versions.

```ts
import { ChatQVAC } from "qvac-langgraph";
```

## Compatibility

The package targets specific versions of the libraries it connects to. Both are required peer dependencies, so npm and pnpm report a missing or out-of-range install.

| Library | Peer range | Tested with | Used for |
| --- | --- | --- | --- |
| `@langchain/core` | `^1.2.12` | 1.2.12 | `BaseChatModel`, messages, tool conversion. |
| `@qvac/sdk` | `^0.18.2` | 0.18.2 | Types only (`Tool`, `ToolCall`, `CompletionStats`); the adapter never calls the SDK. |

- `@qvac/sdk` is pre-1.0, so a caret range covers `0.18.x` only. A `0.19` release is outside the range until this package is checked against it and the range is widened.
- `@langchain/langgraph` is **not** a peer dependency: the adapter never imports it, so any LangGraph release that works with your `@langchain/core` can run `ChatQVAC`. It is only a dev dependency for the examples (`^1.4.16`, tested with 1.4.17).

## Usage

`ChatQVAC` takes a `complete` function (`QvacChatCompletionFn`). Whatever runs your model provides it, for example a thin wrapper over the SDK's `completion()`:

```ts
import { ChatQVAC, type QvacChatCompletionFn } from "qvac-langgraph";
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
| `sessionId` | Per-request KV cache key, passed through for you to use. |

| Result field | Meaning |
| --- | --- |
| `text` | The reply text. |
| `toolCalls` | SDK `ToolCall[]` (`id`, `name`, `arguments`). |
| `thinkingText` | Optional reasoning trace. |
| `stats` | Optional SDK `CompletionStats`. |

`onToken` is only passed when the caller streams (`model.stream(...)`). Even then, the returned promise must resolve with the full result.

### Tools

```ts
const modelWithTools = model.bindTools([myLangChainTool]);
```

LangChain tools are converted to the SDK's flat `Tool` shape. The completion engine only accepts primitive properties, so only each property's `type`, `description` and `enum` are kept; nested object or array item schemas are dropped. Tool calls in the reply come back as regular LangChain `tool_calls`.

### Per-call options

Beyond the usual LangChain call options, `invoke`/`stream` accept:

```ts
await model.invoke(messages, { temperature: 0.2, seed: 42, sessionId: "chat-123" });
```

`temperature` overrides the constructor default. All three are forwarded in the request.

### What comes back

- **Thinking trace:** `reply.additional_kwargs.thinkingText`.
- **Stats:** `llmOutput.stats` from `generate`, or `generationInfo.stats` on the last streamed chunk.
- **Streaming:** text deltas arrive as content chunks; one final empty chunk carries `tool_calls`, `thinkingText` and stats.

### Messages and images

- Roles map as system -> `system`, human -> `user`, ai -> `assistant`, tool -> `tool`.
- An AI message that made tool calls is replayed to the model as a JSON `{ text?, tool_calls }` string, since `QvacChatMessage` has no tool-call field.
- Image content blocks with `image/jpeg` or `image/png` become `images` (bytes) on the message; other image types are dropped. Turning those bytes into whatever your engine accepts is up to `complete`.

## Examples

The [`examples/`](./examples) folder runs a small calculator agent (an `add` and a `subtract` tool) against a local Qwen3 model, written two ways (using langchain and not using it):

```bash
npm run example:graph --workspace=packages/qvac-langgraph   # LangGraph StateGraph + ChatQVAC
npm run example:plain --workspace=packages/qvac-langgraph   # plain loop over `complete`, no LangGraph/LangChain
```

- [`examples/qvacComplete.ts`](./examples/qvacComplete.ts) - a working `complete` built on the SDK's `completion()`.
- [`examples/graphExample.ts`](./examples/graphExample.ts) - the pipeline as a graph, with a text diagram of the flow.
- [`examples/plainExample.ts`](./examples/plainExample.ts) - the same pipeline as a hand-written loop, with a comment mapping each graph piece to its plain-code counterpart.

See [`examples/README.md`](./examples/README.md) for details. The first run downloads the model.

## Development

```bash
npm run build 
npm test
```
