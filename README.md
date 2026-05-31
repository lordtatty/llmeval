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

## How an eval flows

```
   Eval ─► Run × Repeat ─► Aggregate ─► PostChecks ─► EvalResult

   Inside each Run, both SUT and Judge can record Usage via ctx
   (sub-module judges do it automatically; SUTs opt in with one call).
```

An `Eval` is declarative — what to run, how often, what to assert, what to judge, what budgets to enforce. The runner executes it in four phases:

1. **Run × `Repeat` times** (up to `Concurrency` in parallel): for each Run, call `Run(ctx)`, apply each `Assertion` to the output, then call `Judge.Evaluate` once on all `Criteria` (the judge step is skipped when either `Judge` or `Criteria` is unset). SUTs record their own LLM token usage via `llmeval.RecordUsage(ctx, ...)`; sub-module judges (`anthropic.NewDefaultJudge`, `openai.NewDefaultJudge`) do it automatically.
2. **Aggregate**: per-Assertion and per-Criterion pass rates, Usage summed by `(provider, model)`.
3. **PostChecks** run against the aggregate — `MaxCost` for dollar budgets (fail-closed on unpriced calls), `MaxTokens` for raw quotas, or any user-supplied `PostCheck`.
4. **Result**: `EvalResult` with `Pass` plus per-phase detail. `PrintText` / `PrintJSON` render it; `llmevaltest.Run` auto-logs the text report when the eval fails so debugging starts with full context.

See [`examples/classifier/classifier_eval_test.go`](examples/classifier/classifier_eval_test.go) for a six-test walkthrough from simplest (one strict assertion) to most realistic (budget + LLM judge + multi-criterion).

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
judge := anthropic.NewDefaultJudge(&client)
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
judge := openai.NewDefaultJudge(&client)
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

## Putting it together

The canonical shape — SUT, multiple assertions, an LLM judge with criteria,
and a budget — in one place:

```go
//go:build llmeval

package summariser_test

import (
    "context"
    "os"
    "regexp"
    "testing"

    anthropicsdk "github.com/anthropics/anthropic-sdk-go"
    "github.com/anthropics/anthropic-sdk-go/option"

    "github.com/lordtatty/llmeval"
    "github.com/lordtatty/llmeval/anthropic"
    "github.com/lordtatty/llmeval/llmevaltest"
)

var endsWithPeriod = regexp.MustCompile(`\.$`)

func TestSummariser(t *testing.T) {
    client := anthropicsdk.NewClient(option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))

    llmevaltest.Run(t, llmeval.Eval{
        Run: func(ctx context.Context) (string, error) {
            resp, err := client.Messages.New(ctx, anthropicsdk.MessageNewParams{
                Model: anthropicsdk.ModelClaudeHaiku4_5, MaxTokens: 100,
                Messages: []anthropicsdk.MessageParam{
                    anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock(
                        "Summarise in one sentence: ...your input text...",
                    )),
                },
            })
            if err != nil {
                return "", err
            }
            // Record SUT usage so the budget check below has something to sum.
            llmeval.RecordUsage(ctx, llmeval.Usage{
                Provider:     anthropic.ProviderName,
                Model:        string(resp.Model),
                InputTokens:  int(resp.Usage.InputTokens),
                OutputTokens: int(resp.Usage.OutputTokens),
            })
            return resp.Content[0].Text, nil
        },

        Repeat: 5, // surface LLM non-determinism

        // Deterministic format checks.
        Assertions: []llmeval.Assertion{
            llmeval.Matches(endsWithPeriod),
        },

        // LLM-judged rubric items. The judge call records its own usage
        // automatically; one call per Run evaluates all Criteria.
        Judge: anthropic.NewDefaultJudge(&client),
        Criteria: []llmeval.Criterion{
            {Description: "captures the main point of the input"},
            {Description: "is grammatically correct"},
        },

        // Aggregate-level policy: cap the suite's total dollar cost.
        PostChecks: []llmeval.PostCheck{
            llmeval.MaxCost(0.10, anthropic.Pricer()),
        },
    })
}
```

For a stub-based version that runs without API keys, see
[`examples/classifier/`](examples/classifier/).

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

## Common questions

**My SUT returns structured output, not a string** — `Eval.Run` returns
`(string, error)`, so serialize once and assert on typed fields with
`llmeval.CheckJSON`:

```go
import (
    "context"
    "encoding/json"
    "fmt"
    "testing"

    "github.com/lordtatty/llmeval"
    "github.com/lordtatty/llmeval/llmevaltest"
)

type response struct {
    Category   string
    Confidence float64
}

func TestMySUT(t *testing.T) {
    llmevaltest.Run(t, llmeval.Eval{
        Run: func(ctx context.Context) (string, error) {
            r, err := mySUT(ctx)
            if err != nil {
                return "", err
            }
            b, err := json.Marshal(r)
            if err != nil {
                return "", err
            }
            return string(b), nil
        },
        Assertions: []llmeval.Assertion{
            llmeval.CheckJSON("category is positive", func(r response) (bool, string) {
                if r.Category == "positive" {
                    return true, ""
                }
                return false, "got " + r.Category
            }),
            llmeval.CheckJSON("confidence ≥ 0.8", func(r response) (bool, string) {
                if r.Confidence >= 0.8 {
                    return true, ""
                }
                return false, fmt.Sprintf("confidence %.2f", r.Confidence)
            }),
        },
    })
}
```

The judge still sees the JSON string and can include "is well-formed JSON"
as a criterion if you want.

**`MaxCost` fails with "no matching pricer"** — you didn't pass a `Pricer`
for the provider whose usage was recorded. `MaxCost` is fail-closed:
unpriced usage means the budget can't be certified, so it fails rather
than silently miss cost. Pass the relevant `Pricer`s:

```go
llmeval.MaxCost(0.10, anthropic.Pricer(), openai.Pricer())
```

**SUT token usage isn't showing in `EvalResult.Usage`** — sub-module
judges (`anthropic.NewDefaultJudge`, `openai.NewDefaultJudge`) record
usage automatically. SUTs don't; you call `llmeval.RecordUsage(ctx, u)`
after each LLM call inside your `Run`. See the snippet in "Putting it
together" above.

**When do I use Assertion vs Criterion vs PostCheck?**

| Layer        | When                                                     | Cost            |
|--------------|----------------------------------------------------------|-----------------|
| `Assertion`  | Deterministic predicate per Run output                   | Free            |
| `Criterion`  | Natural-language rubric you can't express as a predicate | One judge call  |
| `PostCheck`  | Aggregate-level policy (budgets, multi-run patterns)     | Free, post-aggr |

**My eval hits the timeout** — `Eval.Timeout` covers the whole Run: SUT
call + assertions + judge call together. A slow SUT eats into the
judge's budget. Either bump `Timeout` or speed up the SUT.

**How do I aggregate cost across many evals?** — `EvalResult.Usage`
holds one Run's aggregate. Walk results across multiple `Run` calls and
feed the union into `TotalCost`:

```go
var all []llmeval.Usage
for _, eval := range evals {
    all = append(all, llmeval.Run(ctx, eval).Usage...)
}
total := llmeval.TotalCost(all, anthropic.Pricer(), openai.Pricer())
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, day-to-day commands,
the module layout, and PR conventions.

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
- `llmeval.CheckJSON[T](name, fn)` — like `Check` but the predicate sees the SUT output already decoded into `T`
- `llmeval.AtLeast(rate, asn)` — tolerance wrapper for multi-run evals
- `llmeval.Judge` / `Criterion` / `PromptedJudge` — batched LLM-as-judge: one LLM call per Run, N criteria, N verdicts back
- Pluggable response format: default `PrefixVerdictParser` (PASS/FAIL prefix) or `JSONVerdictParser` + `JSONPromptTemplate` (for structured-output-capable LLMs)
- `llmeval/anthropic.NewDefaultJudge` / `llmeval/openai.NewDefaultJudge` — opt-in pre-wired judges in sub-modules so the core stays SDK-free; `NewJSONJudge` ships the same shape pre-configured for JSON-mode replies
- `llmeval/judgetest` — curated `Cases` + `AssertCase(t, judge, c)` helper for live prompt-quality tests against any `llmeval.Judge` implementation
- `llmeval.Run` — single-eval runner (no `testing` dependency)
- `llmevaltest.Run` / `llmevaltest.RequireSuccess` — `testing.T` integration in a subpackage
- Per-assertion + per-criterion pass-rate aggregation across `Repeat` runs
- Per-`Eval.Timeout` via `context.WithTimeout`, panic recovery in the SUT
- `Eval.Concurrency` — cap on parallel Run invocations (default 1 = sequential)
- `llmeval.PrintText` / `llmeval.PrintJSON` — render an `EvalResult` as a human report or stable-shape JSON; `llmevaltest.Run` auto-logs the text report on failure so debugging starts with full per-run detail
- `llmevaltest.WithReporter` — swap the auto-log reporter (pass `nil` to silence)
- `llmeval.RecordUsage` / `llmeval.Usage` / `llmeval.NewUsageCtx` — track token usage from every LLM call (judge calls are recorded automatically by the sub-modules; SUT code records its own with one line); aggregated by `(provider, model)` into `EvalResult.Usage`
- `llmeval.TotalCost` + sub-module `Pricer()` — estimate dollar cost from recorded usage; sub-modules ship internal best-effort price tables, override by passing your own `llmeval.Pricer` earlier in the `TotalCost` call (first match wins)
- `Eval.PostChecks` + `llmeval.PostCheck` / `llmeval.MaxCost(limit, pricers...)` / `llmeval.MaxTokens(limit)` — policy checks that run once after aggregation with full access to the result; `MaxCost` enforces a dollar budget (fail-closed on unpriced usage), `MaxTokens` enforces a raw-token cap without needing a price table. A failed `PostCheck` marks the eval failed and `llmevaltest.Run` surfaces the failure via `t.Errorf`.
