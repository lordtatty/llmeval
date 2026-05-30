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
	"testing"

	"github.com/lordtatty/llmeval"
)

// TestingT is the minimal subset of *testing.T that RequireSuccess uses.
// *testing.T satisfies it directly — the interface exists so the failure
// path is unit-testable with a fake.
type TestingT interface {
	Helper()
	Errorf(format string, args ...any)
}

// Run runs eval and marks t failed via t.Errorf if any assertion did not
// meet its MinPassRate. If eval.Name is empty it defaults to t.Name().
// The eval inherits t.Context() so it's cancelled automatically when the
// test ends. The returned EvalResult lets callers inspect details after.
func Run(t *testing.T, eval llmeval.Eval) llmeval.EvalResult {
	t.Helper()
	if eval.Name == "" {
		eval.Name = t.Name()
	}
	result := llmeval.Run(t.Context(), eval)
	RequireSuccess(t, result)
	return result
}

// RequireSuccess marks t failed via t.Errorf for each assertion and each
// judged criterion in result that didn't meet its MinPassRate. Run calls
// this automatically; you only need it when you've called llmeval.Run
// directly.
func RequireSuccess(t TestingT, result llmeval.EvalResult) {
	t.Helper()
	if result.Pass {
		return
	}
	for _, a := range result.Assertions {
		if !a.Pass {
			t.Errorf("eval %q: assertion %q failed: %d/%d (need ≥%v)",
				result.Name, a.Name, a.Passed, a.Total, a.MinRate)
		}
	}
	for _, c := range result.Criteria {
		if !c.Pass {
			t.Errorf("eval %q: criterion %q failed: %d/%d (need ≥%v)",
				result.Name, c.Description, c.Passed, c.Total, c.MinRate)
		}
	}
}
