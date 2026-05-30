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

Runnable evals are under [`examples/`](examples/). They use a stub SUT so
they run without an LLM client; in your own code you'd swap the stub for your
real client.

```sh
go test -tags=llmeval ./examples/classifier/ -v
```

## What works today

- `llmeval.Eval` — one declarative eval per `Test*` function
- `llmeval.Equal`, `OneOf`, `Contains`, `NotContains`, `Matches` — built-in assertion helpers
- `llmeval.Check(name, fn)` — adapter for any custom predicate (testify, go-cmp, etc.)
- `llmeval.AtLeast(rate, asn)` — tolerance wrapper for multi-run evals
- `llmeval.Judge` / `Criterion` / `PromptedJudge` — batched LLM-as-judge: one LLM call per Run, N criteria, N verdicts back
- Pluggable response format: default `PrefixVerdictParser` (PASS/FAIL prefix) or `JSONVerdictParser` + `JSONPromptTemplate` (for structured-output-capable LLMs)
- `llmeval.Run` — single-eval runner (no `testing` dependency)
- `llmevaltest.Run` / `llmevaltest.RequireSuccess` — `testing.T` integration in a subpackage
- Per-assertion + per-criterion pass-rate aggregation across `Repeat` runs
- Per-`Eval.Timeout` via `context.WithTimeout`, panic recovery in the SUT

## Coming soon

- `WithConcurrency`, `WithOutput` options
- `PrintText`, `PrintJSON` reporting
