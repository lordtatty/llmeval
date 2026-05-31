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

// callCounter records SUT invocations — used by the Repeat / cancellation
// tests to specify how many times the runner actually invoked the SUT.
type callCounter struct{ calls int }

func (c *callCounter) returning(output string) func(context.Context) (string, error) {
	return func(context.Context) (string, error) {
		c.calls++
		return output, nil
	}
}

// ── Repeat ──────────────────────────────────────────────────────────────────

func TestRepeatInvokesTheSUTOnceForEachRepeat(t *testing.T) {
	counter := &callCounter{}

	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        counter.returning("x"),
		Repeat:     5,
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})

	assert.Equal(t, 5, counter.calls)
	assert.True(t, result.Pass, "result=%+v", result)
	assert.Len(t, result.Runs, 5)
	assert.Equal(t, 5, result.Assertions[0].Passed)
}

// ── SUT error / panic ───────────────────────────────────────────────────────

func TestSUTErrorIsRecordedAndCausesEvalToFail(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(context.Context) (string, error) { return "", errors.New("boom") },
		Assertions: []llmeval.Assertion{llmeval.Equal("anything")},
	})

	assert.False(t, result.Pass)
	require.Len(t, result.Runs, 1)
	assert.Error(t, result.Runs[0].Err)
}

func TestSUTPanicIsRecoveredAndRecordedAsAnError(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(context.Context) (string, error) { panic("boom") },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})

	assert.False(t, result.Pass)
	require.Len(t, result.Runs, 1)
	require.Error(t, result.Runs[0].Err)
	assert.Contains(t, result.Runs[0].Err.Error(), "boom")
}

// ── Timeout ─────────────────────────────────────────────────────────────────

func TestTimeoutDoesNotFireWhenSUTReturnsInTime(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(context.Context) (string, error) { return "x", nil },
		Timeout:    time.Second,
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})

	assert.True(t, result.Pass)
	assert.NoError(t, result.Runs[0].Err)
}

func TestTimeoutFiresAndIsSurfacedAsDeadlineExceeded(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run: func(ctx context.Context) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		},
		Timeout:    5 * time.Millisecond,
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})

	assert.False(t, result.Pass)
	assert.ErrorIs(t, result.Runs[0].Err, context.DeadlineExceeded)
}

// ── Parent ctx cancellation ────────────────────────────────────────────────

func TestParentCtxCancellationAbortsRemainingRepeats(t *testing.T) {
	// The SUT cancels the context on its first call. The runner should
	// observe the cancellation before starting a second iteration, even
	// though Repeat asks for 10.
	ctx, cancel := context.WithCancel(context.Background())

	result := llmeval.Run(ctx, llmeval.Eval{
		Run: func(context.Context) (string, error) {
			cancel()
			return "x", nil
		},
		Repeat:     10,
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})

	assert.Len(t, result.Runs, 1, "loop should stop after first iteration sees ctx.Err")
}

// ── ExampleRun (godoc example, runs under go test) ─────────────────────────

// ExampleRun shows the most common shape of an eval: one SUT call, a strict
// format check, a tolerant accuracy check, and 10 repeats. In a real test
// it would be a func TestX(t *testing.T) calling llmevaltest.Run; here we
// use llmeval.Run so the godoc example produces deterministic output.
func ExampleRun() {
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
