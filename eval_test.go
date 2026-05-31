package llmeval_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
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

	assert.False(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Runs, 1)
	assert.Error(t, result.Runs[0].Err)
}

func TestSUTPanicDoesNotCrashTheTestProcessAndFailsTheEval(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(context.Context) (string, error) { panic("boom") },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})

	assert.False(t, result.Pass, "result=%+v", result)
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

	assert.True(t, result.Pass, "result=%+v", result)
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

	assert.False(t, result.Pass, "result=%+v", result)
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

// ── Concurrency ─────────────────────────────────────────────────────────────

func TestConcurrencyRunsInParallelWhenSet(t *testing.T) {
	// All n SUT calls must be in flight simultaneously — they each push on
	// `started` and block on `release`. If the runner is sequential, the
	// second `started` send never arrives and the timeout fires.
	const n = 5
	started := make(chan struct{}, n)
	release := make(chan struct{})

	done := make(chan llmeval.EvalResult, 1)
	go func() {
		done <- llmeval.Run(context.Background(), llmeval.Eval{
			Repeat:      n,
			Concurrency: n,
			Run: func(context.Context) (string, error) {
				started <- struct{}{}
				<-release
				return "x", nil
			},
		})
	}()

	for range n {
		select {
		case <-started:
		case <-time.After(time.Second):
			close(release)
			t.Fatal("not all runs reached the barrier — concurrency didn't happen")
		}
	}
	close(release)

	result := <-done
	assert.Len(t, result.Runs, n)
}

func TestConcurrencyDoesNotExceedTheConfiguredLimit(t *testing.T) {
	const repeat = 20
	const limit = 4
	var inFlight, peak atomic.Int32

	result := llmeval.Run(context.Background(), llmeval.Eval{
		Repeat:      repeat,
		Concurrency: limit,
		Run: func(context.Context) (string, error) {
			cur := inFlight.Add(1)
			for {
				p := peak.Load()
				if cur <= p || peak.CompareAndSwap(p, cur) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			inFlight.Add(-1)
			return "x", nil
		},
	})

	assert.Len(t, result.Runs, repeat)
	assert.LessOrEqual(t, peak.Load(), int32(limit), "peak in-flight exceeded the configured limit")
}

func TestConcurrencyProducesADistinctResultPerRepeat(t *testing.T) {
	// Each Run returns a unique string. Slot clobbering (two goroutines
	// writing the same runs[idx]) would surface as a duplicate or missing
	// output here.
	const n = 50
	var id atomic.Int32

	result := llmeval.Run(context.Background(), llmeval.Eval{
		Repeat:      n,
		Concurrency: 8,
		Run: func(context.Context) (string, error) {
			return fmt.Sprintf("%d", id.Add(1)), nil
		},
	})

	require.Len(t, result.Runs, n)
	seen := make(map[string]bool)
	for _, r := range result.Runs {
		require.False(t, seen[r.Output], "duplicate output: %s", r.Output)
		seen[r.Output] = true
	}
	assert.Len(t, seen, n)
}

func TestConcurrencyHandlesConcurrencyGreaterThanRepeat(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Repeat:      3,
		Concurrency: 10,
		Run:         func(context.Context) (string, error) { return "x", nil },
		Assertions:  []llmeval.Assertion{llmeval.Equal("x")},
	})

	assert.True(t, result.Pass, "result=%+v", result)
	assert.Len(t, result.Runs, 3)
}

func TestConcurrencyRespectsParentCtxCancellation(t *testing.T) {
	// Parallel variant: the first SUT call cancels the parent ctx. The
	// runner must stop spawning new repeats; what's already in flight may
	// finish or error. We assert the result is bounded by the in-flight
	// batch (≤ Concurrency) and strictly less than Repeat.
	const repeat = 20
	const concurrency = 4
	ctx, cancel := context.WithCancel(context.Background())
	var once sync.Once

	result := llmeval.Run(ctx, llmeval.Eval{
		Repeat:      repeat,
		Concurrency: concurrency,
		Run: func(context.Context) (string, error) {
			once.Do(cancel)
			return "x", nil
		},
	})

	// Up to `concurrency` goroutines may already be mid-runOnce when the
	// cancel fires (they passed the post-acquire ctx check before the
	// canceller ran), and they're allowed to finish. Anything spawned
	// after their sem slots free will short-circuit on the re-check.
	assert.LessOrEqual(t, len(result.Runs), concurrency)
	assert.Less(t, len(result.Runs), repeat, "some repeats should have been skipped")
}

func TestConcurrencyRecoversSUTPanicsInEachGoroutine(t *testing.T) {
	// Each parallel goroutine calls runOnce, which has its own deferred
	// recover. A panic in one SUT call must not take down the runner or
	// affect any other in-flight goroutine.
	const repeat = 8
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Repeat:      repeat,
		Concurrency: 4,
		Run:         func(context.Context) (string, error) { panic("boom") },
		Assertions:  []llmeval.Assertion{llmeval.Equal("x")},
	})

	require.Len(t, result.Runs, repeat)
	for i, r := range result.Runs {
		require.Error(t, r.Err, "run %d should have errored", i)
		assert.Contains(t, r.Err.Error(), "boom")
	}
	assert.False(t, result.Pass)
}

// ── Usage aggregation ──────────────────────────────────────────────────────

func TestRunAggregatesUsageRecordedFromInsideTheSUT(t *testing.T) {
	// The SUT records usage via RecordUsage; Run aggregates per
	// (provider, model) into EvalResult.Usage.
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Repeat: 3,
		Run: func(ctx context.Context) (string, error) {
			llmeval.RecordUsage(ctx, llmeval.Usage{
				Provider: "openai", Model: "gpt-4.1-mini",
				InputTokens: 100, OutputTokens: 50,
			})
			return "x", nil
		},
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})

	require.Len(t, result.Usage, 1)
	assert.Equal(t, "openai", result.Usage[0].Provider)
	assert.Equal(t, 300, result.Usage[0].InputTokens)
	assert.Equal(t, 150, result.Usage[0].OutputTokens)
}

func TestRunUsageIsEmptyWhenNoLLMCallsAreRecorded(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(context.Context) (string, error) { return "x", nil },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})

	assert.Empty(t, result.Usage)
}

func TestRunIsolatesUsageBetweenSeparateEvalInvocations(t *testing.T) {
	// EvalResult.Usage reflects only that Run's calls, not anything from
	// an earlier Run sharing the same parent ctx.
	ctx := context.Background()
	sut := func(ctx context.Context) (string, error) {
		llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "p", Model: "m", InputTokens: 10})
		return "x", nil
	}

	first := llmeval.Run(ctx, llmeval.Eval{Run: sut, Assertions: []llmeval.Assertion{llmeval.Equal("x")}})
	second := llmeval.Run(ctx, llmeval.Eval{Run: sut, Assertions: []llmeval.Assertion{llmeval.Equal("x")}})

	require.Len(t, first.Usage, 1)
	assert.Equal(t, 10, first.Usage[0].InputTokens)
	require.Len(t, second.Usage, 1)
	assert.Equal(t, 10, second.Usage[0].InputTokens, "second eval should not see first eval's tokens")
}

// ── RunResult.MarshalJSON ──────────────────────────────────────────────────

func TestRunResultMarshalJSONOmitsErrFieldWhenNil(t *testing.T) {
	data, err := json.Marshal(llmeval.RunResult{Output: "ok", Pass: true})
	require.NoError(t, err)
	assert.NotContains(t, string(data), `"err"`)
}

func TestRunResultMarshalJSONRendersErrAsAString(t *testing.T) {
	data, err := json.Marshal(llmeval.RunResult{Err: errors.New("rate limited")})
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, "rate limited", decoded["err"])
}

func TestRunResultMarshalJSONConvertsDurationToMilliseconds(t *testing.T) {
	data, err := json.Marshal(llmeval.RunResult{Duration: 1234 * time.Millisecond})
	require.NoError(t, err)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, float64(1234), decoded["durationMs"])
}

func TestRunResultMarshalJSONPreservesAssertionsAndCriteria(t *testing.T) {
	data, err := json.Marshal(llmeval.RunResult{
		Output:     "x",
		Pass:       false,
		Assertions: []llmeval.AssertionResult{{Pass: false, Reason: "nope"}},
		Criteria:   []llmeval.CriterionResult{{Pass: true, Reason: "ok"}},
	})
	require.NoError(t, err)

	var decoded struct {
		Output     string                       `json:"output"`
		Pass       bool                         `json:"pass"`
		Assertions []llmeval.AssertionResult    `json:"assertions"`
		Criteria   []llmeval.CriterionResult    `json:"criteria"`
	}
	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, "x", decoded.Output)
	assert.False(t, decoded.Pass)
	assert.Equal(t, "nope", decoded.Assertions[0].Reason)
	assert.Equal(t, "ok", decoded.Criteria[0].Reason)
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
