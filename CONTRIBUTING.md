# Contributing

Thanks for taking an interest. The bar here is clear, well-tested Go that
fits the patterns already in the repo.

## Quick setup

Install Go (version matches `go.mod`), plus golangci-lint and mockery:

```sh
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.12.2
go install github.com/vektra/mockery/v3@latest
```

For live LLM tests (optional — they cost money):

```sh
cp .env.example .env.local
# fill in your keys; both .env and .env.local are gitignored
```

## Day-to-day

```sh
make            # lists every target
make check      # tests + examples + coverage + lint (no paid calls)
make test-live  # paid; needs ANTHROPIC_API_KEY / OPENAI_API_KEY
```

`make check` is the pre-PR safety net. If it's green locally, CI will be
too.

## Module layout

This is a multi-module repo:

- `.` (`github.com/lordtatty/llmeval`) — core, no SDK deps
- `./anthropic`, `./openai` — opt-in sub-modules with SDK deps
- `./judgetest`, `./llmevaltest` — sub-packages of the core
- `./examples/classifier` — runs in `make test-examples`

For cross-module edits, set up a local Go workspace (gitignored):

```sh
go work init . ./anthropic ./openai
```

## Style

- **Tests**: testify only (`assert` / `require`), sentence-style names
  (`TestThingDoesYWhenZ`), one behaviour per test. Inline per-file
  helpers are fine; no shared `helpers_test.go`.
- **Mocks**: generated via mockery (`make mocks`); config in
  `.mockery.yml`. Use the `EXPECT()` pattern (one expectation per call).
- **Comments**: only when the WHY is non-obvious. Skip explanations of
  WHAT the code does — well-named identifiers do that.
- **Lint**: `make lint` must pass. Config in `.golangci.yml`. No
  per-file exclusions.
- **Coverage**: 100% on library packages is gated in CI. Generated
  mocks and example packages are excluded.
- **Commit messages**: imperative mood (`Add X`, `Fix Y`, `Refactor Z`)
  to match the existing log.

## Pull requests

1. Branch off `main`.
2. Make changes; run `make check`.
3. If you touched a sub-module, run `make tidy` to keep `go.mod`
   honest.
4. Open a PR. CI runs tests + lint + coverage across every module on
   every push.

## Reporting bugs / requesting features

Open an issue. Include the smallest reproducer you can — for a flaky
eval, the `Eval` declaration plus the `EvalResult` JSON (`PrintJSON`)
is usually enough.
