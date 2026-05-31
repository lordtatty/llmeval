package llmevaltest_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/llmevaltest"
)

// fakeT implements llmevaltest.TestingT so we can verify RequireSuccess's
// failure path without poisoning the surrounding test.
type fakeT struct {
	helperCalls int
	errors      []string
}

func (f *fakeT) Helper() { f.helperCalls++ }
func (f *fakeT) Errorf(format string, args ...any) {
	f.errors = append(f.errors, fmt.Sprintf(format, args...))
}

func TestRun_PassingEval_DoesNotFailT(t *testing.T) {
	result := llmevaltest.Run(t, llmeval.Eval{
		Run:        func(ctx context.Context) (string, error) { return "hello", nil },
		Assertions: []llmeval.Assertion{llmeval.Equal("hello")},
	})
	assert.True(t, result.Pass, "result=%+v", result)
	// If Run had called t.Errorf, this test would already be failing.
}

func TestRun_NameDefaultsToTName(t *testing.T) {
	result := llmevaltest.Run(t, llmeval.Eval{
		Run:        func(ctx context.Context) (string, error) { return "x", nil },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	assert.Equal(t, t.Name(), result.Name)
}

func TestRequireSuccess_PassingEval_DoesNothing(t *testing.T) {
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{Pass: true})
	assert.Empty(t, fake.errors)
	assert.Equal(t, 1, fake.helperCalls, "Helper should always be called once")
}

func TestRequireSuccess_FailingEval_CallsErrorfPerFailedAssertion(t *testing.T) {
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
			{Name: "b", Passed: 1, Total: 1, MinRate: 1.0, Pass: true}, // passed — should not generate an Errorf
			{Name: "c", Passed: 3, Total: 5, MinRate: 0.8, Pass: false},
		},
	})
	require.Len(t, fake.errors, 2, "one Errorf per failed assertion")
	assert.Contains(t, fake.errors[0], `assertion "a" failed: 0/1`)
	assert.Contains(t, fake.errors[1], `assertion "c" failed: 3/5`)
}

func TestRequireSuccess_FailingEval_CallsErrorfPerFailedCriterion(t *testing.T) {
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Criteria: []llmeval.CriterionRate{
			{Description: "is concise", Passed: 1, Total: 5, MinRate: 0.8, Pass: false},
			{Description: "is on-topic", Passed: 5, Total: 5, MinRate: 1.0, Pass: true}, // passed — no Errorf
		},
	})
	require.Len(t, fake.errors, 1)
	assert.Contains(t, fake.errors[0], `criterion "is concise" failed: 1/5`)
}

func TestRequireSuccess_FailedAssertion_IncludesPerRunDetails(t *testing.T) {
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
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
	require.Len(t, fake.errors, 1)
	msg := fake.errors[0]
	// The header still includes pass-rate info.
	assert.Contains(t, msg, `assertion "equals positive" failed: 1/3`)
	// And now also per-failed-run details: run number, output, reason.
	assert.Contains(t, msg, "run 2")
	assert.Contains(t, msg, "neutral")
	assert.Contains(t, msg, `got "neutral"`)
	assert.Contains(t, msg, "run 3")
	assert.Contains(t, msg, "blue")
	// The passing run (run 1) should NOT be in the details.
	assert.NotContains(t, msg, "run 1")
}

func TestRequireSuccess_FailedCriterion_IncludesJudgeReason(t *testing.T) {
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Criteria: []llmeval.CriterionRate{
			{Description: "mentions TLS", Passed: 1, Total: 2, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: "summary about TLS handshake", Criteria: []llmeval.CriterionResult{{Pass: true}}},
			{Output: "summary about encryption", Criteria: []llmeval.CriterionResult{{Pass: false, Reason: "summary discusses encryption but never names TLS"}}},
		},
	})
	require.Len(t, fake.errors, 1)
	msg := fake.errors[0]
	assert.Contains(t, msg, `criterion "mentions TLS" failed: 1/2`)
	assert.Contains(t, msg, "run 2")
	assert.Contains(t, msg, "summary about encryption")
	assert.Contains(t, msg, "summary discusses encryption but never names TLS")
}

func TestRequireSuccess_EmptyAssertionReason_OmitsTrailingSeparator(t *testing.T) {
	// A custom assertion may leave Reason empty. The detail line should
	// just show "run N: <output>" without a dangling "— ".
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: "x", Assertions: []llmeval.AssertionResult{{Pass: false, Reason: ""}}},
		},
	})
	require.Len(t, fake.errors, 1)
	msg := fake.errors[0]
	assert.Contains(t, msg, `run 1: "x"`)
	assert.NotContains(t, msg, `"x" —`, "should not dangle a separator when Reason is empty")
}

func TestRequireSuccess_EmptyCriterionReason_OmitsTrailingSeparator(t *testing.T) {
	// Same for criteria — if the judge returns Pass=false with no Reason,
	// don't append " — judge:" with nothing after.
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Criteria: []llmeval.CriterionRate{
			{Description: "c", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: "x", Criteria: []llmeval.CriterionResult{{Pass: false, Reason: ""}}},
		},
	})
	require.Len(t, fake.errors, 1)
	msg := fake.errors[0]
	assert.Contains(t, msg, `run 1: "x"`)
	assert.NotContains(t, msg, "judge:", "should not include 'judge:' prefix when Reason is empty")
}

func TestRequireSuccess_LongOutputIsTruncated(t *testing.T) {
	long := strings.Repeat("x", 500)
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
		Name: "demo",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Runs: []llmeval.RunResult{
			{Output: long, Assertions: []llmeval.AssertionResult{{Pass: false, Reason: "nope"}}},
		},
	})
	require.Len(t, fake.errors, 1)
	msg := fake.errors[0]
	// Long output is truncated with an ellipsis marker.
	assert.Contains(t, msg, "…")
	// And the message doesn't contain the full 500-x payload.
	assert.NotContains(t, msg, long)
}

func TestRequireSuccess_ErroredRunsAreNotReportedAsFailedCriteria(t *testing.T) {
	// Symmetric with the assertion case: a Run that errored before the judge
	// step shouldn't appear in the criterion failure details.
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
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
	require.Len(t, fake.errors, 1)
	msg := fake.errors[0]
	assert.NotContains(t, msg, "SUT exploded")
	assert.Contains(t, msg, "run 2")
	assert.Contains(t, msg, "judge says no")
}

func TestRequireSuccess_ErroredRunsAreNotReportedAsFailedAssertions(t *testing.T) {
	// A run with Err set never executed the assertions. The detail block
	// shouldn't show that run as an assertion failure.
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
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
	require.Len(t, fake.errors, 1)
	msg := fake.errors[0]
	// The errored run (1) shouldn't appear; the real failure (2) should.
	assert.NotContains(t, msg, "SUT exploded")
	assert.Contains(t, msg, "run 2")
	assert.Contains(t, msg, "wrong")
}

func TestRequireSuccess_FailingEval_ReportsBothAssertionsAndCriteria(t *testing.T) {
	fake := &fakeT{}
	llmevaltest.RequireSuccess(fake, llmeval.EvalResult{
		Name: "mixed",
		Pass: false,
		Assertions: []llmeval.AssertionRate{
			{Name: "a", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
		Criteria: []llmeval.CriterionRate{
			{Description: "c1", Passed: 0, Total: 1, MinRate: 1.0, Pass: false},
		},
	})
	require.Len(t, fake.errors, 2)
}
