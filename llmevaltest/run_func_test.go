package llmevaltest_test

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/llmevaltest"
	"github.com/lordtatty/llmeval/llmevaltest/mocks"
)

// ─────────────────────────────────────────────────────────────────────────────
// llmevaltest.RunFunc — the *testing.T entry point for the imperative path.
// ─────────────────────────────────────────────────────────────────────────────

func TestRunFuncDoesNotFailTheTestWhenTheEvalPasses(t *testing.T) {
	result := llmevaltest.RunFunc(t, llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
	})
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestRunFuncDefaultsTheEvalNameToTName(t *testing.T) {
	result := llmevaltest.RunFunc(t, llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
	})
	assert.Equal(t, t.Name(), result.Name)
}

func TestRunFuncPreservesAUserSuppliedEvalName(t *testing.T) {
	result := llmevaltest.RunFunc(t, llmeval.EvalFunc{
		Name: "my custom name",
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
	})
	assert.Equal(t, "my custom name", result.Name)
}

// ─────────────────────────────────────────────────────────────────────────────
// RequireSuccessFunc — what the *testing.T sees when an EvalFunc fails.
// ─────────────────────────────────────────────────────────────────────────────

func TestRequireSuccessFuncIsSilentForAPassingEval(t *testing.T) {
	m := mocks.NewMockTestingT(t)
	m.EXPECT().Helper().Maybe()

	llmevaltest.RequireSuccessFunc(m, llmeval.EvalFuncResult{Pass: true})
}

func TestRequireSuccessFuncReportsOneErrorPerFailedAssertion(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
			{Name: "b", Passed: 1, Total: 1, MinRate: 1.0, Pass: true},
			{Name: "c", Passed: 3, Total: 5, MinRate: 0.8, Pass: false},
		},
	})

	require.Len(t, r.messages, 2)
	assert.Contains(t, r.messages[0], `assertion "a" failed: 0/1`)
	assert.Contains(t, r.messages[1], `assertion "c" failed: 3/5`)
}

func TestRequireSuccessFuncReportsOneErrorPerFailedPostCheck(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Name: "demo",
		Pass: false,
		PostChecks: []llmeval.PostCheckResult{
			{Name: "max cost: $0.10", Pass: false, Reason: "spent $0.20, limit $0.10"},
			{Name: "always-ok", Pass: true},
		},
	})

	require.Len(t, r.messages, 1)
	assert.Contains(t, r.messages[0], `post-check "max cost: $0.10" failed`)
	assert.Contains(t, r.messages[0], `spent $0.20, limit $0.10`)
}

// ── Per-run detail in failure messages ──────────────────────────────────────

func TestFuncFailedAssertionMessageIncludesPerFailedRunDetail(t *testing.T) {
	// EvalFunc assertions are matched by Name across runs (vs. Eval[T]'s
	// by-index lookup). A run that didn't include the named assertion is
	// silently skipped in the detail loop.
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "category", Passed: 1, Total: 3, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.EvalFuncRunResult{
			{Assertions: []llmeval.AssertionResult{{Name: "category", Pass: true}}},
			{Assertions: []llmeval.AssertionResult{{Name: "category", Pass: false, Reason: `got "neutral"`}}},
			{Assertions: []llmeval.AssertionResult{{Name: "category", Pass: false, Reason: `got "blue"`}}},
		},
	})

	require.Len(t, r.messages, 1)
	msg := r.messages[0]
	assert.Contains(t, msg, `assertion "category" failed: 1/3`)
	assert.Contains(t, msg, "run 2")
	assert.Contains(t, msg, `got "neutral"`)
	assert.Contains(t, msg, "run 3")
	assert.Contains(t, msg, `got "blue"`)
	assert.NotContains(t, msg, "run 1", "the passing run should not be in the failure details")
}

func TestFuncFailedAssertionDetailIgnoresRunsThatDidNotReturnTheName(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "category", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.EvalFuncRunResult{
			// Run 1 returned an unrelated assertion — should not show up.
			{Assertions: []llmeval.AssertionResult{{Name: "confidence", Pass: true}}},
			{Assertions: []llmeval.AssertionResult{{Name: "category", Pass: false, Reason: "nope"}}},
		},
	})

	require.Len(t, r.messages, 1)
	msg := r.messages[0]
	assert.Contains(t, msg, "run 2")
	assert.Contains(t, msg, "nope")
	assert.NotContains(t, msg, "run 1")
}

func TestFuncErroredRunsAreNotReportedAsAssertionFailures(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "category", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.EvalFuncRunResult{
			{Err: errors.New("SUT exploded")},
			{Assertions: []llmeval.AssertionResult{{Name: "category", Pass: false, Reason: "wrong"}}},
		},
	})

	require.Len(t, r.messages, 1)
	assert.NotContains(t, r.messages[0], "SUT exploded")
	assert.Contains(t, r.messages[0], "run 2")
	assert.Contains(t, r.messages[0], "wrong")
}

func TestFuncLongReasonsAreTruncatedInFailureMessages(t *testing.T) {
	long := strings.Repeat("x", 500)
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.EvalFuncRunResult{
			{Assertions: []llmeval.AssertionResult{{Name: "a", Pass: false, Reason: long}}},
		},
	})

	require.Len(t, r.messages, 1)
	assert.Contains(t, r.messages[0], "…")
	assert.NotContains(t, r.messages[0], long)
}

func TestFuncAnEmptyAssertionReasonOmitsTheDashSeparator(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.EvalFuncRunResult{
			{Assertions: []llmeval.AssertionResult{{Name: "a", Pass: false, Reason: ""}}},
		},
	})

	require.Len(t, r.messages, 1)
	assert.Contains(t, r.messages[0], "run 1")
	assert.NotContains(t, r.messages[0], "run 1 —")
}

// ── Auto-log on failure ────────────────────────────────────────────────────

func TestRequireSuccessFuncAutoLogsPrintTextOnFailure(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.EvalFuncRunResult{
			{Assertions: []llmeval.AssertionResult{{Name: "a", Pass: false, Reason: "nope"}}, Duration: time.Millisecond},
		},
	})

	require.Len(t, r.logs, 1)
	assert.Contains(t, r.logs[0], "Eval: demo")
	assert.Contains(t, r.logs[0], "FAIL")
}

func TestWithReporterFuncReplacesTheDefaultReporter(t *testing.T) {
	r := captureErrorfMessages(t)
	called := false

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
	}, llmevaltest.WithReporterFunc(func(w io.Writer, _ llmeval.EvalFuncResult) error {
		called = true
		_, err := w.Write([]byte("custom report"))
		return err
	}))

	assert.True(t, called, "custom reporter should have been invoked")
	require.Len(t, r.logs, 1)
	assert.Contains(t, r.logs[0], "custom report")
}

func TestWithReporterFuncNilSilencesTheAutoLog(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccessFunc(r.T, llmeval.EvalFuncResult{
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
	}, llmevaltest.WithReporterFunc(nil))

	assert.Empty(t, r.logs)
	assert.NotEmpty(t, r.messages)
}
