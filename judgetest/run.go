package judgetest

import (
	"context"

	"github.com/lordtatty/llmeval"
)

// TestingT is the subset of *testing.T that AssertCase uses. Real tests
// satisfy it implicitly; the package's own unit tests pass a fake.
type TestingT interface {
	Helper()
	Errorf(format string, args ...any)
}

// AssertCase calls judge.Evaluate for c and reports a failure if any
// verdict's Pass value disagrees with c.Wants, if the judge errors, or if
// the verdict count doesn't match the criteria list.
//
// Use it inside a t.Run loop over Cases so the iteration is visible at
// the call site:
//
//	for _, c := range judgetest.Cases {
//	    t.Run(c.Name, func(t *testing.T) {
//	        judgetest.AssertCase(t, judge, c)
//	    })
//	}
func AssertCase(t TestingT, judge llmeval.Judge, c Case) {
	t.Helper()
	verdicts, err := judge.Evaluate(context.Background(), c.Output, c.Criteria)
	if err != nil {
		t.Errorf("judge error: %v", err)
		return
	}
	if len(verdicts) != len(c.Wants) {
		t.Errorf("got %d verdicts, want %d", len(verdicts), len(c.Wants))
		return
	}
	for i, v := range verdicts {
		if v.Pass != c.Wants[i] {
			t.Errorf("criterion %d (%q): got Pass=%v, want Pass=%v (reason=%q)",
				i, c.Criteria[i].Description, v.Pass, c.Wants[i], v.Reason)
		}
	}
}
