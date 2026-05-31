// Package llmevaltest provides go test integration for the llmeval package.
//
// It lives in its own subpackage so the core llmeval package doesn't need to
// import "testing" — that import would otherwise be pulled into every
// consumer's build graph just because RunTest takes a *testing.T.
//
// Typical usage:
//
//	import (
//	    "github.com/lordtatty/llmeval"
//	    "github.com/lordtatty/llmeval/llmevaltest"
//	)
//
//	func TestSentimentClassifier(t *testing.T) {
//	    llmevaltest.Run(t, llmeval.Eval{ ... })
//	}
package llmevaltest

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/lordtatty/llmeval"
)

// TestingT is the minimal subset of *testing.T that RequireSuccess uses.
// *testing.T satisfies it directly — the interface exists so the failure
// path is unit-testable with a fake.
type TestingT interface {
	Helper()
	Errorf(format string, args ...any)
	Log(args ...any)
}

// outputTruncateLimit caps how much of each SUT output we splice into the
// failure message. Long LLM responses would otherwise dominate the test log.
const outputTruncateLimit = 200

// Option configures Run / RequireSuccess.
type Option func(*config)

type config struct {
	// reporter writes the failure-detail report. Nil silences auto-logging.
	reporter func(io.Writer, llmeval.EvalResult) error
}

// WithReporter overrides the failure-detail reporter used by Run /
// RequireSuccess on a failing eval. The default is llmeval.PrintText. Pass
// nil to silence the auto-log entirely.
func WithReporter(fn func(io.Writer, llmeval.EvalResult) error) Option {
	return func(c *config) { c.reporter = fn }
}

// Run runs eval and marks t failed via t.Errorf if any assertion did not
// meet its MinPassRate. If eval.Name is empty it defaults to t.Name().
// The eval inherits t.Context() so it's cancelled automatically when the
// test ends. On failure the configured reporter (default llmeval.PrintText)
// is written to t.Log before the per-assertion failure messages, so
// debugging starts with full per-run detail. The returned EvalResult lets
// callers inspect details after.
func Run(t *testing.T, eval llmeval.Eval, opts ...Option) llmeval.EvalResult {
	t.Helper()
	if eval.Name == "" {
		eval.Name = t.Name()
	}
	result := llmeval.Run(t.Context(), eval)
	RequireSuccess(t, result, opts...)
	return result
}

// RequireSuccess marks t failed via t.Errorf for each assertion and each
// judged criterion in result that didn't meet its MinPassRate. Before
// the failure messages it writes the configured reporter (default
// llmeval.PrintText) to t.Log so the failing test's output includes the
// full per-run detail.
//
// Run calls this automatically; you only need it when you've called
// llmeval.Run directly.
func RequireSuccess(t TestingT, result llmeval.EvalResult, opts ...Option) {
	t.Helper()
	if result.Pass {
		return
	}
	cfg := config{reporter: llmeval.PrintText}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.reporter != nil {
		var buf bytes.Buffer
		_ = cfg.reporter(&buf, result)
		t.Log(buf.String())
	}
	for i, a := range result.Assertions {
		if !a.Pass {
			t.Errorf("eval %q: assertion %q failed: %d/%d (need ≥%v)%s",
				result.Name, a.Name, a.Passed, a.Total, a.MinRate,
				assertionFailureDetails(result.Runs, i))
		}
	}
	for j, c := range result.Criteria {
		if !c.Pass {
			t.Errorf("eval %q: criterion %q failed: %d/%d (need ≥%v)%s",
				result.Name, c.Description, c.Passed, c.Total, c.MinRate,
				criterionFailureDetails(result.Runs, j))
		}
	}
}

// assertionFailureDetails returns a "\n  run N: <output> — <reason>" line for
// every Run where the assertion at idx returned Pass=false. Errored runs
// (where assertions never executed) are skipped — their failure is a
// separate concern surfaced via RunResult.Err.
func assertionFailureDetails(runs []llmeval.RunResult, idx int) string {
	var b strings.Builder
	for i, run := range runs {
		if run.Err != nil || idx >= len(run.Assertions) {
			continue
		}
		if ar := run.Assertions[idx]; !ar.Pass {
			fmt.Fprintf(&b, "\n  run %d: %q%s", i+1, truncate(run.Output), reasonSuffix("", ar.Reason))
		}
	}
	return b.String()
}

// criterionFailureDetails returns a "\n  run N: <output> — judge: <reason>"
// line for every Run where the judge's verdict at idx was Pass=false.
func criterionFailureDetails(runs []llmeval.RunResult, idx int) string {
	var b strings.Builder
	for i, run := range runs {
		if run.Err != nil || idx >= len(run.Criteria) {
			continue
		}
		if cr := run.Criteria[idx]; !cr.Pass {
			fmt.Fprintf(&b, "\n  run %d: %q%s", i+1, truncate(run.Output), reasonSuffix("judge: ", cr.Reason))
		}
	}
	return b.String()
}

// reasonSuffix renders " — <prefix><reason>" when reason is non-empty,
// and the empty string otherwise — so an empty reason doesn't leave a
// dangling separator in the failure line.
func reasonSuffix(prefix, reason string) string {
	if reason == "" {
		return ""
	}
	return " — " + prefix + reason
}

// truncate caps s at outputTruncateLimit runes, appending an ellipsis when
// shortened. Rune-aware so it never splits a UTF-8 sequence in half.
func truncate(s string) string {
	runes := []rune(s)
	if len(runes) <= outputTruncateLimit {
		return s
	}
	return string(runes[:outputTruncateLimit]) + "…"
}
