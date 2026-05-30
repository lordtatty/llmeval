package llmeval_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
)

func TestRun_Repeat_RunsNTimes(t *testing.T) {
	calls := 0
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run: func(ctx context.Context) (string, error) {
			calls++
			return "x", nil
		},
		Repeat:     5,
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	assert.Equal(t, 5, calls, "SUT call count")
	assert.Len(t, result.Runs, 5)
	assert.True(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Assertions, 1)
	assert.Equal(t, 5, result.Assertions[0].Passed)
	assert.Equal(t, 5, result.Assertions[0].Total)
}

func TestRun_SUTError_FailsEval(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(ctx context.Context) (string, error) { return "", errors.New("boom") },
		Assertions: []llmeval.Assertion{llmeval.Equal("anything")},
	})
	assert.False(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Runs, 1)
	assert.Error(t, result.Runs[0].Err)
}

func TestRun_SUTPanic_IsRecoveredAndRecordedAsError(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(ctx context.Context) (string, error) { panic("boom") },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	assert.False(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Runs, 1)
	require.Error(t, result.Runs[0].Err)
	assert.Contains(t, result.Runs[0].Err.Error(), "boom",
		"panic value should appear in the recovered error")
}

func TestRun_StopsWhenCtxCancelledBetweenRepeats(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	result := llmeval.Run(ctx, llmeval.Eval{
		Run: func(ctx context.Context) (string, error) {
			calls++
			cancel() // cancel after the first call
			return "x", nil
		},
		Repeat:     10,
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	// SUT cancels during call 1; loop sees ctx.Err() before call 2 → exactly one run.
	assert.Equal(t, 1, calls, "expected exactly one SUT call")
	assert.Len(t, result.Runs, 1, "expected exactly one RunResult")
}

func TestRun_Timeout_DoesNotFireWhenSUTFinishesInTime(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(ctx context.Context) (string, error) { return "x", nil },
		Timeout:    time.Second,
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	assert.True(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Runs, 1)
	assert.NoError(t, result.Runs[0].Err)
}

func TestRun_Timeout_FiresAndSetsErr(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run: func(ctx context.Context) (string, error) {
			// Block until ctx times out; respect cancellation.
			<-ctx.Done()
			return "", ctx.Err()
		},
		Timeout:    5 * time.Millisecond,
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	assert.False(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Runs, 1)
	assert.ErrorIs(t, result.Runs[0].Err, context.DeadlineExceeded)
}

// ExampleRun shows the most common shape of an eval: one SUT call, a strict
// format check, a tolerant accuracy check, and 10 repeats. In a real test
// it would be a func TestX(t *testing.T) calling llmevaltest.Run; here we
// use llmeval.Run so the godoc example produces deterministic output.
func ExampleRun() {
	// A stand-in SUT — your code would call your LLM here.
	classify := func(_ context.Context) (string, error) {
		return "positive", nil
	}

	result := llmeval.Run(context.Background(), llmeval.Eval{
		Name:   "sentiment classifier",
		Run:    classify,
		Repeat: 10,
		Assertions: []llmeval.Assertion{
			llmeval.OneOf("positive", "negative", "neutral"), // strict format
			llmeval.AtLeast(0.8, llmeval.Equal("positive")),  // tolerant accuracy
		},
	})

	fmt.Printf("pass=%v\n", result.Pass)
	for _, a := range result.Assertions {
		fmt.Printf("  %s: %d/%d (need ≥%v) pass=%v\n", a.Name, a.Passed, a.Total, a.MinRate, a.Pass)
	}
	// Output:
	// pass=true
	//   one of: positive, negative, neutral: 10/10 (need ≥1) pass=true
	//   equal: "positive": 10/10 (need ≥0.8) pass=true
}
