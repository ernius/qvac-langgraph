# Changelog

## Unreleased

Review of the LangGraph integration, focused on making `ChatQVAC` work in richer graphs and on using acyclic graphs (DAGs) where the steps are known up front. Suggested version: `1.2.0`. The public API only grows, but two behavior changes are listed below for review.

### Fixed

- **`withStructuredOutput` threw "Input is not an AIMessageChunk".** `_generate` returned an `AIMessage`, while LangChain's structured-output parser only accepts `AIMessageChunk`, the type `invoke` is declared to return. It now returns an `AIMessageChunk`, so router and planner nodes can use structured output.
- **Aborting a call did not stop it.** With a `signal` or `timeout`, `invoke` kept waiting for the full generation, and the local model stayed busy after a LangGraph run was aborted. The adapter now rejects as soon as the signal fires and forwards the signal to `complete` in the request.
- **Some tool schemas were rejected by the SDK.** A nullable property produced `type: ["number", "null"]` and a union produced a type array, which the SDK's own `toolSchema` rejects. Each property now gets a single supported type: a nullable keeps its non-null type, a union keeps its first member's type, anything else falls back to `string` as before.
- **`npm ci` followed by `npm test` failed on Apple Silicon.** npm skipped rolldown's native binary because `vite` was only reachable as vitest's peer ([npm/cli#4828](https://github.com/npm/cli/issues/4828)). `vite` is now a direct dev dependency at the version already locked. The lockfile only gains that entry and loses 36 `"peer": true` flags; no version changes.

### Added

- **`responseFormat` call option**, forwarded in the request as `QvacChatCompletionRequest.responseFormat`. It mirrors the SDK's `completion({ responseFormat })`, which turns a JSON schema into a grammar, so nested schemas (a list of plan steps, for example) always parse. The new `QvacResponseFormat` type is exported, since the SDK does not export its own.
- **`signal` on `QvacChatCompletionRequest`**, so `complete` can stop the generation, for example with the SDK's `cancel({ requestId })`.
- **`examples/loopExample.ts`** (`npm run example:loop`): a question whose second operation needs the first result, as an `agent <-> tools` cycle bounded by a turn budget kept in the state. When the budget runs out it routes to a tool-less answer node instead of hitting a `GraphRecursionError`.
- **`examples/calculatorTools.ts`**: the `add`/`subtract` LangChain tools, shared by the two graph examples.
- **README "In LangGraph" section**: when to use a DAG and when a bounded cycle, forwarding `config` in nodes, limiting parallel model calls with `maxConcurrency`, and KV cache sessions.

### Changed

- **`examples/graphExample.ts` is now a DAG**: `plan -> tools -> answer`, with no edge back. A run makes at most two model calls and needs no recursion limit. The previous cycle relied on the default recursion limit, which allows 13 model calls. The system prompt used to say "one operation at a time", which forced sequential tool calls; it now asks for every independent operation in a single turn.
- **`examples/plainExample.ts`** mirrors the new DAG as straight-line code, with no loop, and serializes tool turns the same way the adapter does.
- **`examples/qvacComplete.ts`** now cancels the generation when `request.signal` aborts, uses `sessionId` as the SDK's `kvCache` key, forwards `responseFormat`, and deletes the KV cache sessions it created on `shutdown`.
- **Graph nodes forward `config`** to the model, so callbacks, streaming and abort signals also reach it on runtimes without async context propagation, such as Bare or Expo.
- **README fixes**: the import path in Usage was `qvac-langgraph` instead of `@space-uy/qvac-langgraph`, the example commands used an npm workspace flag that does not apply to this repo, and the request table was missing `requestId`.

### Behavior changes to review

1. **Tool history sent to the model.** An AI tool-call turn is replayed with each call in the SDK's `{ id, name, arguments }` shape, instead of LangChain's `{ type, id, name, args }`. A tool result is sent as a JSON `{ tool_call_id, name?, content }` string, instead of the bare result text. `QvacChatMessage` has no tool-call id field, so this is the only way for the model to match parallel results to the calls that asked for them. It changes the prompt every existing consumer sends.
2. **`invoke` returns an `AIMessageChunk` instance** instead of an `AIMessage`. This is the type LangChain already declared, and `AIMessage.isInstance` checks still pass, but code that relied on the concrete class could notice.

### Verification

- 17 unit tests pass: the 12 existing ones plus 5 new ones for structured output, `responseFormat`, abort, nullable and union types, and the tool history format. Each new test fails against the previous code.
- `npm run build` and the examples typecheck pass.
- All three examples were run against the real `QWEN3_600M_INST_Q4` model:

| Example | Result |
| --- | --- |
| `example:graph` | Requested `add(17, 25)` and `subtract(100, 37)` in one turn and answered 42 and 63 |
| `example:loop` | Ran `add(17, 25)`, then `subtract(42, 37)`, and answered 5 |
| `example:plain` | Answered 42 and 63 |

- Cancellation was checked against the real model: a long streamed generation aborted at 1.5 s stopped right away, and the next call finished in 66 ms.

### Files

- `src/chatQvac.ts`, `src/types.ts`, `src/index.ts`, `src/chatQvac.test.ts`
- `examples/qvacComplete.ts`, `examples/graphExample.ts`, `examples/plainExample.ts`, `examples/README.md`
- New: `examples/loopExample.ts`, `examples/calculatorTools.ts`
- `README.md`, `package.json`, `package-lock.json`

### Notes

- A note for `qvacComplete.ts` users: SDK 0.18.2's docs say a KV cache hit sends only the last message, but its code sends every message added since the last cached turn, so several tool results per turn are safe. The SDK does not check the cached prefix against the history, so a new `sessionId` is needed whenever the history is trimmed or resumed from an older checkpoint.
- Nested tool argument schemas are still flattened, since the SDK's `Tool` shape only takes one level of properties. Use `responseFormat` when a nested shape matters.
