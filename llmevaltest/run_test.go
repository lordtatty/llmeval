package llmevaltest_test

import (
	"context"
	"fmt"
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
