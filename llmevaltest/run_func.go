package llmevaltest

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/lordtatty/llmeval"
)

// OptionFunc configures RunFunc / RequireSuccessFunc. Mirrors Option[T]
// for the imperative EvalFunc path. Kept as a distinct type rather than a
// shared Option-with-any-T because the reporter signature carries the
// concrete result shape — EvalFuncResult vs EvalResult[T].
type OptionFunc func(*configFunc)

type configFunc struct {
	reporter func(io.Writer, llmeval.EvalFuncResult) error
}

// WithReporterFunc overrides the failure-detail reporter used by RunFunc /
// RequireSuccessFunc on a failing eval. The default is llmeval.PrintTextFunc.
// Pass nil to silence the auto-log entirely.
func WithReporterFunc(fn func(io.Writer, llmeval.EvalFuncResult) error) OptionFunc {
	return func(c *configFunc) { c.reporter = fn }
}

// RunFunc runs eval and marks t failed via t.Errorf if any assertion or
// post-check did not meet its threshold. If eval.Name is empty it defaults
// to t.Name(). The eval inherits t.Context() so it's cancelled
// automatically when the test ends. On failure the configured reporter
// (default llmeval.PrintTextFunc) is written to t.Log before the
// per-assertion failure messages, so debugging starts with full per-run
// detail.
func RunFunc(t *testing.T, eval llmeval.EvalFunc, opts ...OptionFunc) llmeval.EvalFuncResult {
	t.Helper()
	if eval.Name == "" {
		eval.Name = t.Name()
	}
	result := llmeval.RunFunc(t.Context(), eval)
	RequireSuccessFunc(t, result, opts...)
	return result
}

// RequireSuccessFunc marks t failed via t.Errorf for each assertion in
// result that didn't meet its MinPassRate and each post-check that didn't
// pass. Before the failure messages it writes the configured reporter
// (default llmeval.PrintTextFunc) to t.Log so the failing test's output
// includes the full per-run detail.
//
// RunFunc calls this automatically; you only need it when you've called
// llmeval.RunFunc directly.
func RequireSuccessFunc(t TestingT, result llmeval.EvalFuncResult, opts ...OptionFunc) {
	t.Helper()
	if result.Pass {
		return
	}
	cfg := configFunc{reporter: llmeval.PrintTextFunc}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.reporter != nil {
		var buf bytes.Buffer
		_ = cfg.reporter(&buf, result)
		t.Log(buf.String())
	}
	if len(result.Runs) > 0 && !anyRunSucceededFunc(result.Runs) {
		t.Errorf("eval %q: no successful run to evaluate (%d/%d errored)%s",
			result.Name, len(result.Runs), len(result.Runs),
			erroredRunDetailsFunc(result.Runs))
	} else {
		for _, a := range result.Assertions {
			if !a.Pass {
				t.Errorf("eval %q: assertion %q failed: %d/%d (need ≥%v)%s",
					result.Name, a.Name, a.Passed, a.Total, a.MinRate,
					assertionFailureDetailsFunc(result.Runs, a.Name))
			}
		}
	}
	for _, pc := range result.PostChecks {
		if !pc.Pass {
			t.Errorf("eval %q: post-check %q failed%s",
				result.Name, pc.Name, reasonSuffix("", pc.Reason))
		}
	}
}

// anyRunSucceededFunc reports whether at least one run completed without
// error. Used to decide between the framework-level "no successful runs"
// failure message and the per-assertion 0/0 noise.
func anyRunSucceededFunc(runs []llmeval.EvalFuncRunResult) bool {
	for _, rr := range runs {
		if rr.Err == nil {
			return true
		}
	}
	return false
}

// erroredRunDetailsFunc returns "\n  run N: <err>" lines for every errored
// run, suitable for splicing into a single t.Errorf so the diagnosis
// includes the underlying SUT failure reasons.
func erroredRunDetailsFunc(runs []llmeval.EvalFuncRunResult) string {
	var b strings.Builder
	for i, rr := range runs {
		if rr.Err != nil {
			fmt.Fprintf(&b, "\n  run %d: %v", i+1, rr.Err)
		}
	}
	return b.String()
}

// assertionFailureDetailsFunc returns a "\n  run N — <reason>" line for
// every Run where an AssertionResult with matching Name returned
// Pass=false. Unlike the index-based path used by Eval[T], lookup here is
// by Name because EvalFunc runs may legitimately return different
// assertion sets (a Run that didn't return Name simply doesn't appear).
// Errored runs are skipped — their failure surfaces via Runs[i].Err.
func assertionFailureDetailsFunc(runs []llmeval.EvalFuncRunResult, name string) string {
	var b strings.Builder
	for i, run := range runs {
		if run.Err != nil {
			continue
		}
		for _, ar := range run.Assertions {
			if ar.Name != name || ar.Pass {
				continue
			}
			fmt.Fprintf(&b, "\n  run %d%s", i+1, reasonSuffix("", truncate(ar.Reason)))
		}
	}
	return b.String()
}
