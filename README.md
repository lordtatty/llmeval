# llmeval

Test LLM-driven Go code against natural-language criteria, with multi-run
aggregation and `go test` integration baked in. Thin, idiomatic, no LLM-SDK
dependency — you bring your own client.

## Quick start

Evals are ordinary `Test*` functions under a build tag so they don't run
with normal `go test`:

```go
//go:build llmeval

package myapp_test

import (
    "context"
    "testing"

    "github.com/lordtatty/llmeval"
    "github.com/lordtatty/llmeval/llmevaltest"
)

func TestSentimentClassifier(t *testing.T) {
    llmevaltest.Run(t, llmeval.Eval{
        Run: func(ctx context.Context) (string, error) {
            // Replace this with your own LLM call.
            return yourLLMClient.Classify(ctx, "I love this product!")
        },
        Repeat: 5, // surface LLM non-determinism
        Assertions: []llmeval.Assertion{
            llmeval.AtLeast(0.8, llmeval.Equal("positive")), // ≥80% must label correctly
        },
    })
}
```

The core `llmeval` package has no dependency on `testing`. The `testing.T`
integration lives in `llmeval/llmevaltest` so it isn't pulled into your
production binaries.

Run only the evals:

```sh
go test -tags=llmeval ./...
```

Run everything else, skipping the evals (the default):

```sh
go test ./...
```

The build tag means eval files don't compile into normal test runs — no
accidental LLM calls, no surprise bills.

## Wiring your LLM

The core `llmeval` package has no SDK dependencies. For Anthropic and
OpenAI there are pre-wired judges in sub-modules — pull in only the one
you need, or none at all and wire your own.

### Anthropic (Claude)

```go
import (
    "os"

    anthropicsdk "github.com/anthropics/anthropic-sdk-go"
    "github.com/anthropics/anthropic-sdk-go/option"

    "github.com/lordtatty/llmeval"
    "github.com/lordtatty/llmeval/anthropic"
)

client := anthropicsdk.NewClient(option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
judge := anthropic.NewJudge(&client)
```

Defaults to `claude-haiku-4-5`, 1024 max tokens, 30s per-call timeout —
judges run many times per suite, so the default favours cost. Bump to
Sonnet or Opus when the rubric demands more reasoning.
Override with `anthropic.WithModel(...)`, `anthropic.WithMaxTokens(...)`,
`anthropic.WithTimeout(...)`.

`go get github.com/lordtatty/llmeval/anthropic` — the Anthropic SDK is
pulled in by this sub-module only, never by the core.

### OpenAI

```go
import (
    "os"

    openaisdk "github.com/openai/openai-go"
    "github.com/openai/openai-go/option"

    "github.com/lordtatty/llmeval"
    "github.com/lordtatty/llmeval/openai"
)

client := openaisdk.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
judge := openai.NewJudge(&client)
```

Defaults to `gpt-4.1-mini`, 1024 max completion tokens, 30s per-call
timeout — judges run many times per suite, so the default favours cost.
Bump to `gpt-4.1` or a reasoning model when the rubric demands more.
Override with `openai.WithModel(...)`, `openai.WithMaxTokens(...)`,
`openai.WithTimeout(...)`.

`go get github.com/lordtatty/llmeval/openai` — same isolation as above.

### Other providers (rolling your own)

If your provider isn't covered, wire a `PromptedJudge` directly. The two
sub-modules above are nothing more than thin wrappers around
`LLMFunc func(ctx, prompt) (string, error)`. Any single-turn
text-in-text-out call works — Ollama, a custom HTTP endpoint, whatever:

```go
judge := &llmeval.PromptedJudge{
    LLMFunc: func(ctx context.Context, prompt string) (string, error) {
        body, _ := json.Marshal(map[string]any{
            "model":  "llama3",
            "prompt": prompt,
            "stream": false,
        })
        req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
            "http://localhost:11434/api/generate", bytes.NewReader(body))
        resp, err := http.DefaultClient.Do(req)
        if err != nil {
            return "", err
        }
        defer resp.Body.Close()
        var out struct{ Response string }
        if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
            return "", err
        }
        return out.Response, nil
    },
}
```

### Using JSON-mode structured output

If your LLM supports JSON-mode (Claude, GPT-4o, etc.), pair
`JSONPromptTemplate` with `JSONVerdictParser`. The judge then expects a
`{"verdicts":[{"pass":bool,"reason":string},...]}` reply, which the LLM is
constrained to produce by your client's structured-output config:

```go
judge := &llmeval.PromptedJudge{
    PromptTemplate: llmeval.JSONPromptTemplate,
    Parser:         llmeval.JSONVerdictParser,
    LLMFunc:        callYourLLMInJSONMode,
}
```

## How it works

An eval calls your SUT, checks the output, optionally repeats N times, and
aggregates per-assertion pass rates. LLM-driven systems are non-deterministic;
the framework treats that as a first-class concern.

Every built-in assertion is **strict** by default — it must hold on every
repeat. Wrap any assertion with `llmeval.AtLeast(rate, asn)` to make it
**tolerant**: it then has to hold on at least `rate` of the repeats. The two
mix freely inside one eval:

```go
Assertions: []llmeval.Assertion{
    llmeval.OneOf("positive", "negative", "neutral"), // strict format
    llmeval.AtLeast(0.8, llmeval.Equal("positive")),  // tolerant accuracy
},
```

Keep hard format/safety constraints strict; let fuzzy criteria absorb the
expected drift.

## Linting

Run [golangci-lint](https://golangci-lint.run/):

```sh
golangci-lint run ./...
```

The config in `.golangci.yml` enables only the linters that catch real bugs
or high-value smells (`govet`, `staticcheck`, `ineffassign`, `errcheck`,
`gocyclo` at complexity 10, `misspell`, `modernize`). No style enforcement,
no per-file exclusions — tests have to check errors too.

## Examples

Runnable evals are under [`examples/`](examples/):

- [`examples/classifier/`](examples/classifier/) — sentiment classifier
  with a stub LLM and a stub judge. Demonstrates all the local assertion
  shapes plus a heuristic judge.

For real provider wiring, see the sub-module tests: the Anthropic and
OpenAI judges are specified end-to-end against an `httptest`-backed fake
of the SDK.

```sh
go test -tags=llmeval ./examples/... -v
```

## What works today

- `llmeval.Eval` — one declarative eval per `Test*` function
- `llmeval.Equal`, `OneOf`, `Contains`, `NotContains`, `Matches` — built-in assertion helpers
- `llmeval.Check(name, fn)` — adapter for any custom predicate (testify, go-cmp, etc.)
- `llmeval.AtLeast(rate, asn)` — tolerance wrapper for multi-run evals
- `llmeval.Judge` / `Criterion` / `PromptedJudge` — batched LLM-as-judge: one LLM call per Run, N criteria, N verdicts back
- Pluggable response format: default `PrefixVerdictParser` (PASS/FAIL prefix) or `JSONVerdictParser` + `JSONPromptTemplate` (for structured-output-capable LLMs)
- `llmeval/anthropic.NewJudge` / `llmeval/openai.NewJudge` — opt-in pre-wired judges in sub-modules so the core stays SDK-free
- `llmeval.Run` — single-eval runner (no `testing` dependency)
- `llmevaltest.Run` / `llmevaltest.RequireSuccess` — `testing.T` integration in a subpackage
- Per-assertion + per-criterion pass-rate aggregation across `Repeat` runs
- Per-`Eval.Timeout` via `context.WithTimeout`, panic recovery in the SUT

## Coming soon

- `WithConcurrency`, `WithOutput` options
- `PrintText`, `PrintJSON` reporting
