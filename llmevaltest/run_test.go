package llmevaltest_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/llmevaltest"
	"github.com/lordtatty/llmeval/llmevaltest/mocks"
)

// errorfRecorder is a TestingT mock that captures both Errorf failure
// messages and Log auto-report payloads so tests can inspect shape and
// ordering without configuring exact-argument expectations on mockery.
type errorfRecorder struct {
	messages []string
	logs     []string
	T        *mocks.MockTestingT
}

func captureErrorfMessages(t *testing.T) *errorfRecorder {
	r := &errorfRecorder{T: mocks.NewMockTestingT(t)}
	r.T.EXPECT().Helper().Maybe()
	r.T.EXPECT().Errorf(mock.AnythingOfType("string"), mock.Anything).
		Run(func(format string, args ...any) {
			r.messages = append(r.messages, fmt.Sprintf(format, args...))
		}).Maybe()
	r.T.EXPECT().Log(mock.Anything).
		Run(func(args ...any) {
			r.logs = append(r.logs, fmt.Sprint(args...))
		}).Maybe()
	return r
}

// ─────────────────────────────────────────────────────────────────────────────
// llmevaltest.Run — the *testing.T entry point.
// ─────────────────────────────────────────────────────────────────────────────

func TestRunDoesNotFailTheTestWhenTheEvalPasses(t *testing.T) {
	// If Run had marked t failed, this enclosing test would itself be failing.
	result := llmevaltest.Run(t, llmeval.Eval{
		Run:        func(context.Context) (string, error) { return "hello", nil },
		Assertions: []llmeval.Assertion{llmeval.Equal("hello")},
	})
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestRunDefaultsTheEvalNameToTName(t *testing.T) {
	result := llmevaltest.Run(t, llmeval.Eval{
		Run:        func(context.Context) (string, error) { return "x", nil },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	assert.Equal(t, t.Name(), result.Name)
}

func TestRunPreservesAUserSuppliedEvalName(t *testing.T) {
	result := llmevaltest.Run(t, llmeval.Eval{
		Name:       "my custom name",
		Run:        func(context.Context) (string, error) { return "x", nil },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	assert.Equal(t, "my custom name", result.Name)
}

// ─────────────────────────────────────────────────────────────────────────────
// RequireSuccess — what the *testing.T sees when an eval fails.
// ─────────────────────────────────────────────────────────────────────────────

func TestRequireSuccessIsSilentForAPassingEval(t *testing.T) {
	// The behaviour is "Errorf is never called for a passing eval."
	// Helper() may or may not be called; we don't constrain it (that would
	// couple the test to implementation rather than behaviour).
	m := mocks.NewMockTestingT(t)
	m.EXPECT().Helper().Maybe()
	// No Errorf expectation — if RequireSuccess calls it, mockery fails the test.

	llmevaltest.RequireSuccess(m, llmeval.EvalResult{Pass: true})
}

func TestRequireSuccessReportsOneErrorPerFailedAssertion(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
			{Name: "b", Passed: 1, Total: 1, MinRate: 1.0, Pass: true}, // passed → no Errorf
			{Name: "c", Passed: 3, Total: 5, MinRate: 0.8, Pass: false},
		},
	})

	require.Len(t, r.messages, 2)
	assert.Contains(t, r.messages[0], `assertion "a" failed: 0/1`)
	assert.Contains(t, r.messages[1], `assertion "c" failed: 3/5`)
}

func TestRequireSuccessReportsOneErrorPerFailedCriterion(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Criteria: []llmeval.CriterionRate{
			{Description: "is concise", Passed: 1, Total: 5, MinRate: 0.8, Pass: false},
			{Description: "is on-topic", Passed: 5, Total: 5, MinRate: 1.0, Pass: true},
		},
	})

	require.Len(t, r.messages, 1)
	assert.Contains(t, r.messages[0], `criterion "is concise" failed: 1/5`)
}

func TestRequireSuccessReportsAssertionsAndCriteriaTogether(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "mixed",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Criteria: []llmeval.CriterionRate{
			{Description: "c1", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
	})

	assert.Len(t, r.messages, 2)
}

// ── Per-run detail in failure messages ──────────────────────────────────────

func TestFailedAssertionMessageIncludesPerFailedRunDetail(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "equals positive", Passed: 1, Total: 3, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: "positive", Assertions: []llmeval.AssertionResult{{Pass: true}}},
			{Output: "neutral", Assertions: []llmeval.AssertionResult{{Pass: false, Reason: `got "neutral"`}}},
			{Output: "blue", Assertions: []llmeval.AssertionResult{{Pass: false, Reason: `got "blue"`}}},
		},
	})

	require.Len(t, r.messages, 1)
	msg := r.messages[0]
	assert.Contains(t, msg, `assertion "equals positive" failed: 1/3`)
	assert.Contains(t, msg, "run 2")
	assert.Contains(t, msg, "neutral")
	assert.Contains(t, msg, `got "neutral"`)
	assert.Contains(t, msg, "run 3")
	assert.Contains(t, msg, "blue")
	assert.NotContains(t, msg, "run 1", "the passing run should not be in the failure details")
}

func TestFailedCriterionMessageIncludesTheJudgesReason(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Criteria: []llmeval.CriterionRate{
			{Description: "mentions TLS", Passed: 1, Total: 2, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: "summary about TLS handshake", Criteria: []llmeval.CriterionResult{{Pass: true}}},
			{Output: "summary about encryption",
				Criteria: []llmeval.CriterionResult{{Pass: false, Reason: "summary discusses encryption but never names TLS"}}},
		},
	})

	require.Len(t, r.messages, 1)
	msg := r.messages[0]
	assert.Contains(t, msg, `criterion "mentions TLS" failed: 1/2`)
	assert.Contains(t, msg, "run 2")
	assert.Contains(t, msg, "summary about encryption")
	assert.Contains(t, msg, "summary discusses encryption but never names TLS")
}

func TestLongSUTOutputIsTruncatedInFailureMessages(t *testing.T) {
	long := strings.Repeat("x", 500)
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: long, Assertions: []llmeval.AssertionResult{{Pass: false, Reason: "nope"}}},
		},
	})

	require.Len(t, r.messages, 1)
	assert.Contains(t, r.messages[0], "…")
	assert.NotContains(t, r.messages[0], long)
}

// ── Errored runs are not attributed to assertion / criterion failures ─────

func TestErroredRunsAreNotReportedAsAssertionFailures(t *testing.T) {
	// A Run that errored never executed the assertions — attributing the
	// failure to it would be misleading.
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Err: errors.New("SUT exploded")},
			{Output: "x", Assertions: []llmeval.AssertionResult{{Pass: false, Reason: "wrong"}}},
		},
	})

	require.Len(t, r.messages, 1)
	assert.NotContains(t, r.messages[0], "SUT exploded")
	assert.Contains(t, r.messages[0], "run 2")
	assert.Contains(t, r.messages[0], "wrong")
}

func TestErroredRunsAreNotReportedAsCriterionFailures(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Criteria: []llmeval.CriterionRate{
			{Description: "c", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Err: errors.New("SUT exploded")},
			{Output: "x", Criteria: []llmeval.CriterionResult{{Pass: false, Reason: "judge says no"}}},
		},
	})

	require.Len(t, r.messages, 1)
	assert.NotContains(t, r.messages[0], "SUT exploded")
	assert.Contains(t, r.messages[0], "run 2")
	assert.Contains(t, r.messages[0], "judge says no")
}

// ── Failed PostCheck reporting ─────────────────────────────────────────────

func TestRequireSuccessReportsOneErrorPerFailedPostCheck(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		PostChecks: []llmeval.PostCheckResult{
			{Name: "max cost: $0.10", Pass: false, Reason: "spent $0.20, limit $0.10"},
			{Name: "always-ok", Pass: true}, // passing → no Errorf
		},
	})

	require.Len(t, r.messages, 1)
	assert.Contains(t, r.messages[0], `post-check "max cost: $0.10" failed`)
	assert.Contains(t, r.messages[0], `spent $0.20, limit $0.10`)
}

// ── Auto-log on failure ────────────────────────────────────────────────────

func TestRequireSuccessAutoLogsPrintTextOnFailure(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: "x", Assertions: []llmeval.AssertionResult{{Pass: false, Reason: "nope"}}},
		},
	})

	require.Len(t, r.logs, 1)
	// PrintText's header reaches the log, so debugging starts with context.
	assert.Contains(t, r.logs[0], "Eval: demo")
	assert.Contains(t, r.logs[0], "FAIL")
}

func TestWithReporterReplacesTheDefaultReporter(t *testing.T) {
	r := captureErrorfMessages(t)
	called := false

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
	}, llmevaltest.WithReporter(func(w io.Writer, _ llmeval.EvalResult) error {
		called = true
		_, err := w.Write([]byte("custom report"))
		return err
	}))

	assert.True(t, called, "custom reporter should have been invoked")
	require.Len(t, r.logs, 1)
	assert.Contains(t, r.logs[0], "custom report")
}

func TestWithReporterNilSilencesTheAutoLog(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
	}, llmevaltest.WithReporter(nil))

	assert.Empty(t, r.logs, "WithReporter(nil) should suppress the auto-log")
	// Failure messages still fire — silencing is about the report, not the
	// per-assertion error signal.
	assert.NotEmpty(t, r.messages)
}

// ── Empty-reason handling ───────────────────────────────────────────────────

func TestAnEmptyAssertionReasonOmitsTheDashSeparator(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: "x", Assertions: []llmeval.AssertionResult{{Pass: false, Reason: ""}}},
		},
	})

	require.Len(t, r.messages, 1)
	assert.Contains(t, r.messages[0], `run 1: "x"`)
	assert.NotContains(t, r.messages[0], `"x" —`)
}

func TestAnEmptyCriterionReasonOmitsTheJudgePrefix(t *testing.T) {
	r := captureErrorfMessages(t)

	llmevaltest.RequireSuccess(r.T, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Criteria: []llmeval.CriterionRate{
			{Description: "c", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: "x", Criteria: []llmeval.CriterionResult{{Pass: false, Reason: ""}}},
		},
	})

	require.Len(t, r.messages, 1)
	assert.Contains(t, r.messages[0], `run 1: "x"`)
	assert.NotContains(t, r.messages[0], "judge:")
}
