package llmeval_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
)

// ─────────────────────────────────────────────────────────────────────────────
// RunFunc basics
// ─────────────────────────────────────────────────────────────────────────────

func TestRunFuncCallsRunRepeatTimes(t *testing.T) {
	var calls atomic.Int32
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			calls.Add(1)
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
		Repeat: 5,
	})
	assert.Equal(t, int32(5), calls.Load())
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestRunFuncDefaultsRepeatToOneWhenUnset(t *testing.T) {
	var calls atomic.Int32
	llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			calls.Add(1)
			return nil, nil
		},
	})
	assert.Equal(t, int32(1), calls.Load())
}

func TestRunFuncAggregatesAssertionResultsByNameAcrossRepeats(t *testing.T) {
	// Two repeats, each returning a "category" and "confidence" result.
	// Aggregated rates should reflect the totals per name.
	i := 0
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			i++
			return []llmeval.AssertionResult{
				{Name: "category", Pass: true},                // both pass
				{Name: "confidence", Pass: i == 1},            // first passes, second fails
			}, nil
		},
		Repeat: 2,
	})

	require.Len(t, result.Assertions, 2)
	assert.Equal(t, "category", result.Assertions[0].Name)
	assert.Equal(t, 2, result.Assertions[0].Passed)
	assert.Equal(t, 2, result.Assertions[0].Total)
	assert.True(t, result.Assertions[0].Pass)

	assert.Equal(t, "confidence", result.Assertions[1].Name)
	assert.Equal(t, 1, result.Assertions[1].Passed)
	assert.Equal(t, 2, result.Assertions[1].Total)
	assert.False(t, result.Assertions[1].Pass, "strict MinPassRate of 1.0 should fail on 1/2")
	assert.False(t, result.Pass, "any failing assertion fails the whole eval")
}

func TestRunFuncCountsDuplicateAssertionNamesWithinOneRunAsIndependentObservations(t *testing.T) {
	// Pins the documented semantic: the framework counts every occurrence
	// of a Name, even within the same Run. Adopters who don't want this
	// should deduplicate inside their Run closure before returning.
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			return []llmeval.AssertionResult{
				{Name: "category", Pass: true},
				{Name: "category", Pass: false, Reason: "second occurrence disagrees"},
			}, nil
		},
	})

	require.Len(t, result.Runs, 1)
	assert.False(t, result.Runs[0].Pass, "run-level Pass should be false when any AssertionResult failed")
	require.Len(t, result.Assertions, 1)
	assert.Equal(t, 2, result.Assertions[0].Total, "every occurrence is a separate observation")
	assert.Equal(t, 1, result.Assertions[0].Passed)
}

func TestRunFuncMissingAssertionsDoNotInflateTheirTotal(t *testing.T) {
	// Repeat 1 returns "a", repeat 2 returns "a" + "b". "a" should have
	// Total=2; "b" should have Total=1.
	i := 0
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			i++
			if i == 1 {
				return []llmeval.AssertionResult{{Name: "a", Pass: true}}, nil
			}
			return []llmeval.AssertionResult{
				{Name: "a", Pass: true},
				{Name: "b", Pass: true},
			}, nil
		},
		Repeat: 2,
	})

	require.Len(t, result.Assertions, 2)
	assert.Equal(t, 2, result.Assertions[0].Total, "a was returned by both repeats")
	assert.Equal(t, 1, result.Assertions[1].Total, "b was only returned by the second repeat")
}

func TestRunFuncRespectsMinPassRatesOverride(t *testing.T) {
	// "flaky" passes only once out of three — would fail strict (1.0) but
	// the MinPassRates override of 0.3 lets it through.
	i := 0
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			i++
			return []llmeval.AssertionResult{
				{Name: "flaky", Pass: i == 1},
			}, nil
		},
		Repeat:       3,
		MinPassRates: map[string]float64{"flaky": 0.3},
	})

	require.Len(t, result.Assertions, 1)
	assert.Equal(t, 0.3, result.Assertions[0].MinRate)
	assert.True(t, result.Assertions[0].Pass, "1/3 >= 0.3")
	assert.True(t, result.Pass)
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors and panics
// ─────────────────────────────────────────────────────────────────────────────

func TestRunFuncMarksResultFailedWhenEveryRunErrored(t *testing.T) {
	// Without this guard the eval has nothing to flip Pass to false on
	// (result.Assertions stays empty because nothing was returned), and
	// an outage where every SUT call failed would silently report as a
	// passing eval — the exact opposite of the truth.
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			return nil, errors.New("boom")
		},
		Repeat: 3,
	})

	require.Len(t, result.Runs, 3)
	for _, rr := range result.Runs {
		assert.Error(t, rr.Err)
	}
	assert.False(t, result.Pass, "an eval with no successful runs cannot pass")
}

func TestRunFuncErroredRunsAreRecordedButSkippedInAggregation(t *testing.T) {
	i := 0
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			i++
			if i == 1 {
				return nil, errors.New("boom")
			}
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
		Repeat: 2,
	})

	require.Len(t, result.Runs, 2)
	assert.Error(t, result.Runs[0].Err)
	require.Len(t, result.Assertions, 1)
	assert.Equal(t, 1, result.Assertions[0].Total, "errored run shouldn't contribute to totals")
}

func TestRunFuncAppliesPerRunTimeout(t *testing.T) {
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(ctx context.Context) ([]llmeval.AssertionResult, error) {
			select {
			case <-time.After(200 * time.Millisecond):
				return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
		Timeout: 10 * time.Millisecond,
	})

	require.Len(t, result.Runs, 1)
	require.Error(t, result.Runs[0].Err)
	assert.ErrorIs(t, result.Runs[0].Err, context.DeadlineExceeded)
}

func TestRunFuncRecoversFromPanicsInsideRun(t *testing.T) {
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			panic("boom")
		},
	})

	require.Len(t, result.Runs, 1)
	require.Error(t, result.Runs[0].Err)
	assert.Contains(t, result.Runs[0].Err.Error(), "boom")
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrency
// ─────────────────────────────────────────────────────────────────────────────

func TestRunFuncIsSafeForConcurrentRuns(t *testing.T) {
	var inFlight, peak atomic.Int32
	const limit = 4
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			cur := inFlight.Add(1)
			for {
				p := peak.Load()
				if cur <= p || peak.CompareAndSwap(p, cur) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			inFlight.Add(-1)
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
		Repeat:      20,
		Concurrency: limit,
	})

	assert.Len(t, result.Runs, 20)
	assert.LessOrEqual(t, peak.Load(), int32(limit))
}

// ─────────────────────────────────────────────────────────────────────────────
// PostChecks + Usage + Summary
// ─────────────────────────────────────────────────────────────────────────────

func TestRunFuncRunsPostChecksAgainstTheSummary(t *testing.T) {
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
		PostChecks: []llmeval.PostCheck{
			{
				Name:  "always-false",
				Check: func(llmeval.EvalSummary) (bool, string) { return false, "by design" },
			},
		},
	})

	assert.False(t, result.Pass)
	require.Len(t, result.PostChecks, 1)
	assert.Equal(t, "by design", result.PostChecks[0].Reason)
}

func TestRunFuncCollectsUsageRecordedInsideRun(t *testing.T) {
	// SUT records usage; aggregation surfaces it in the result and a
	// MaxCost PostCheck sees it via Summary().
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(ctx context.Context) ([]llmeval.AssertionResult, error) {
			llmeval.RecordUsage(ctx, llmeval.Usage{
				Provider: "openai", Model: "m", InputTokens: 100, OutputTokens: 50,
			})
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
		Repeat: 3,
	})

	require.Len(t, result.Usage, 1)
	assert.Equal(t, 300, result.Usage[0].InputTokens)
	assert.Equal(t, 150, result.Usage[0].OutputTokens)
}

func TestEvalFuncResultSummaryExposesTheNonRunFields(t *testing.T) {
	r := llmeval.EvalFuncResult{
		Name:       "demo",
		Pass:       false,
		Assertions: []llmeval.AssertionRate{{Name: "a", Passed: 1, Total: 2}},
		Usage:      []llmeval.Usage{{Provider: "p"}},
	}
	s := r.Summary()
	assert.Equal(t, "demo", s.Name)
	assert.False(t, s.Pass)
	assert.Equal(t, "a", s.Assertions[0].Name)
	assert.Equal(t, "p", s.Usage[0].Provider)
}

// ─────────────────────────────────────────────────────────────────────────────
// JudgeAll helper
// ─────────────────────────────────────────────────────────────────────────────

// stubJudgeFixed returns one verdict per criterion with a configured Pass
// and Reason. Used to verify JudgeAll's verdict-to-AssertionResult
// conversion without an LLM.
type stubJudgeFixed struct {
	pass   bool
	reason string
}

func (j stubJudgeFixed) Evaluate(_ context.Context, _ string, criteria []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
	out := make([]llmeval.CriterionResult, len(criteria))
	for i := range criteria {
		out[i] = llmeval.CriterionResult{Pass: j.pass, Reason: j.reason}
	}
	return out, nil
}

func TestJudgeAllReturnsOneAssertionResultPerCriterionNamedByDescription(t *testing.T) {
	results := llmeval.JudgeAll(context.Background(),
		stubJudgeFixed{pass: true},
		"the output",
		[]llmeval.Criterion{
			{Description: "first criterion"},
			{Description: "second criterion"},
		},
	)
	require.Len(t, results, 2)
	assert.Equal(t, "first criterion", results[0].Name)
	assert.True(t, results[0].Pass)
	assert.Equal(t, "second criterion", results[1].Name)
}

// stubJudgeErroring returns the configured error from Evaluate.
type stubJudgeErroring struct{ err error }

func (j stubJudgeErroring) Evaluate(context.Context, string, []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
	return nil, j.err
}

func TestJudgeAllMarksEveryCriterionFailedWhenTheJudgeErrors(t *testing.T) {
	results := llmeval.JudgeAll(context.Background(),
		stubJudgeErroring{err: errors.New("rate limited")},
		"x",
		[]llmeval.Criterion{
			{Description: "first"},
			{Description: "second"},
		},
	)
	require.Len(t, results, 2)
	for _, r := range results {
		assert.False(t, r.Pass)
		assert.Contains(t, r.Reason, "rate limited")
	}
}

// stubJudgeCount returns fewer verdicts than criteria — exercises the
// verdict-count-mismatch path.
type stubJudgeCount struct{ n int }

func (j stubJudgeCount) Evaluate(context.Context, string, []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
	out := make([]llmeval.CriterionResult, j.n)
	for i := range out {
		out[i] = llmeval.CriterionResult{Pass: true}
	}
	return out, nil
}

func TestJudgeAllMarksEveryCriterionFailedWhenVerdictCountMismatchesCriteria(t *testing.T) {
	results := llmeval.JudgeAll(context.Background(),
		stubJudgeCount{n: 1},
		"x",
		[]llmeval.Criterion{
			{Description: "first"},
			{Description: "second"},
		},
	)
	require.Len(t, results, 2)
	for _, r := range results {
		assert.False(t, r.Pass)
		assert.Contains(t, r.Reason, "verdicts for")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Misc: pass=true on every run
// ─────────────────────────────────────────────────────────────────────────────

func TestRunFuncMarksRunPassWhenAllAssertionsPass(t *testing.T) {
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			return []llmeval.AssertionResult{
				{Name: "a", Pass: true},
				{Name: "b", Pass: true},
			}, nil
		},
	})
	require.Len(t, result.Runs, 1)
	assert.True(t, result.Runs[0].Pass)
}

func TestRunFuncMarksRunFailWhenAnyAssertionFails(t *testing.T) {
	result := llmeval.RunFunc(context.Background(), llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			return []llmeval.AssertionResult{
				{Name: "a", Pass: true},
				{Name: "b", Pass: false, Reason: "nope"},
			}, nil
		},
	})
	require.Len(t, result.Runs, 1)
	assert.False(t, result.Runs[0].Pass)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parent ctx cancellation (matches Eval[T]'s semantic)
// ─────────────────────────────────────────────────────────────────────────────

func TestRunFuncAbortsRemainingRepeatsWhenParentCtxIsCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var once sync.Once

	result := llmeval.RunFunc(ctx, llmeval.EvalFunc{
		Run: func(context.Context) ([]llmeval.AssertionResult, error) {
			once.Do(cancel)
			return []llmeval.AssertionResult{{Name: "ok", Pass: true}}, nil
		},
		Repeat: 20,
	})

	assert.Less(t, len(result.Runs), 20, "cancellation should have aborted remaining repeats")
}
