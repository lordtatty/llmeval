package llmeval_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
)

// ─────────────────────────────────────────────────────────────────────────────
// MaxCost
// ─────────────────────────────────────────────────────────────────────────────

func TestMaxCostPassesWhenSpentIsBelowLimit(t *testing.T) {
	flat := func(_ llmeval.Usage) (float64, bool) { return 0.05, true }
	pc := llmeval.MaxCost(0.10, flat)

	pass, _ := pc.Check(llmeval.EvalResult{
		Usage: []llmeval.Usage{{Provider: "p"}},
	})
	assert.True(t, pass)
}

func TestMaxCostPassesWhenSpentEqualsLimitExactly(t *testing.T) {
	flat := func(_ llmeval.Usage) (float64, bool) { return 0.10, true }
	pc := llmeval.MaxCost(0.10, flat)

	pass, _ := pc.Check(llmeval.EvalResult{
		Usage: []llmeval.Usage{{Provider: "p"}},
	})
	assert.True(t, pass, "spent==limit should still be within budget")
}

func TestMaxCostFailsWhenSpentExceedsLimit(t *testing.T) {
	flat := func(_ llmeval.Usage) (float64, bool) { return 0.25, true }
	pc := llmeval.MaxCost(0.10, flat)

	pass, reason := pc.Check(llmeval.EvalResult{
		Usage: []llmeval.Usage{{Provider: "p"}},
	})
	assert.False(t, pass)
	assert.Contains(t, reason, "0.2500")
	assert.Contains(t, reason, "0.1000")
}

func TestMaxCostPassesWithZeroLimitWhenNothingWasSpent(t *testing.T) {
	pc := llmeval.MaxCost(0, func(_ llmeval.Usage) (float64, bool) { return 0, false })

	pass, _ := pc.Check(llmeval.EvalResult{Usage: nil})
	assert.True(t, pass)
}

func TestMaxCostNameSurfacesTheConfiguredLimit(t *testing.T) {
	pc := llmeval.MaxCost(0.05, nil)
	assert.Contains(t, pc.Name, "0.05")
}

func TestMaxCostFailsWhenAnyUsageHasNoMatchingPricer(t *testing.T) {
	// The pricer only matches provider "a"; the usage list has one "a"
	// and one "b" — the unpriced "b" should fail the budget rather than
	// silently contribute $0.
	pricer := func(u llmeval.Usage) (float64, bool) {
		if u.Provider == "a" {
			return 0.01, true
		}
		return 0, false
	}
	pc := llmeval.MaxCost(0.10, pricer)

	pass, reason := pc.Check(llmeval.EvalResult{
		Usage: []llmeval.Usage{
			{Provider: "a"},
			{Provider: "b"},
		},
	})
	assert.False(t, pass)
	assert.Contains(t, reason, "no matching pricer")
	assert.Contains(t, reason, "1")
}

func TestMaxCostFailsWhenNoPricersAreConfiguredButUsageWasRecorded(t *testing.T) {
	// Zero pricers + non-empty usage = "we have no way to certify the
	// budget" → fail. The alternative (silent pass) would be the exact
	// footgun this design exists to prevent.
	pc := llmeval.MaxCost(0.10)

	pass, reason := pc.Check(llmeval.EvalResult{
		Usage: []llmeval.Usage{{Provider: "p"}},
	})
	assert.False(t, pass)
	assert.Contains(t, reason, "no matching pricer")
}

func TestMaxCostDoesNotPanicWhenAPricerInTheVariadicIsNil(t *testing.T) {
	pricer := func(_ llmeval.Usage) (float64, bool) { return 0.01, true }
	pc := llmeval.MaxCost(0.10, nil, pricer)

	pass, _ := pc.Check(llmeval.EvalResult{Usage: []llmeval.Usage{{Provider: "p"}}})
	assert.True(t, pass, "second (non-nil) pricer should match; nil should be skipped")
}

// ─────────────────────────────────────────────────────────────────────────────
// MaxTokens
// ─────────────────────────────────────────────────────────────────────────────

func TestMaxTokensPassesWhenSumIsBelowLimit(t *testing.T) {
	pc := llmeval.MaxTokens(1000)
	pass, _ := pc.Check(llmeval.EvalResult{
		Usage: []llmeval.Usage{{InputTokens: 200, OutputTokens: 100}},
	})
	assert.True(t, pass)
}

func TestMaxTokensPassesWhenSumEqualsLimitExactly(t *testing.T) {
	pc := llmeval.MaxTokens(300)
	pass, _ := pc.Check(llmeval.EvalResult{
		Usage: []llmeval.Usage{{InputTokens: 200, OutputTokens: 100}},
	})
	assert.True(t, pass)
}

func TestMaxTokensFailsWhenSumExceedsLimit(t *testing.T) {
	pc := llmeval.MaxTokens(100)
	pass, reason := pc.Check(llmeval.EvalResult{
		Usage: []llmeval.Usage{
			{InputTokens: 80, OutputTokens: 30},
			{InputTokens: 50, OutputTokens: 10},
		},
	})
	assert.False(t, pass)
	assert.Contains(t, reason, "170") // total used
	assert.Contains(t, reason, "100") // limit
}

func TestMaxTokensPassesWhenUsageIsEmpty(t *testing.T) {
	pc := llmeval.MaxTokens(100)
	pass, _ := pc.Check(llmeval.EvalResult{})
	assert.True(t, pass)
}

func TestMaxTokensNameSurfacesTheConfiguredLimit(t *testing.T) {
	pc := llmeval.MaxTokens(500)
	assert.Contains(t, pc.Name, "500")
}

func TestMaxCostMultiplePricersUseFirstMatchingOne(t *testing.T) {
	cheap := func(u llmeval.Usage) (float64, bool) {
		if u.Provider == "p" {
			return 0.01, true
		}
		return 0, false
	}
	expensive := func(u llmeval.Usage) (float64, bool) {
		if u.Provider == "p" {
			return 0.99, true
		}
		return 0, false
	}
	pc := llmeval.MaxCost(0.05, cheap, expensive)

	pass, _ := pc.Check(llmeval.EvalResult{Usage: []llmeval.Usage{{Provider: "p"}}})
	assert.True(t, pass, "cheap pricer should win; expensive one never runs")
}

// ─────────────────────────────────────────────────────────────────────────────
// Run integration
// ─────────────────────────────────────────────────────────────────────────────

func TestRunRecordsPostCheckResultsOnEvalResult(t *testing.T) {
	pc := llmeval.PostCheck{
		Name:  "always-true",
		Check: func(llmeval.EvalResult) (bool, string) { return true, "" },
	}

	result := runForPostChecks(t, []llmeval.PostCheck{pc})

	require.Len(t, result.PostChecks, 1)
	assert.Equal(t, "always-true", result.PostChecks[0].Name)
	assert.True(t, result.PostChecks[0].Pass)
}

func TestRunMarksResultFailedWhenAnyPostCheckFails(t *testing.T) {
	pc := llmeval.PostCheck{
		Name:  "always-false",
		Check: func(llmeval.EvalResult) (bool, string) { return false, "by design" },
	}

	result := runForPostChecks(t, []llmeval.PostCheck{pc})

	assert.False(t, result.Pass)
	require.Len(t, result.PostChecks, 1)
	assert.Equal(t, "by design", result.PostChecks[0].Reason)
}

func TestRunPostChecksSeeTheAggregatedUsage(t *testing.T) {
	// SUT records usage; the PostCheck reads from result.Usage rather
	// than the raw per-run records, so aggregation must happen before
	// post-checks fire.
	seen := []llmeval.Usage{}
	pc := llmeval.PostCheck{
		Name: "capture-usage",
		Check: func(r llmeval.EvalResult) (bool, string) {
			seen = r.Usage
			return true, ""
		},
	}

	result := llmeval.Run(t.Context(), llmeval.Eval{
		Run: func(ctx context.Context) (string, error) {
			llmeval.RecordUsage(ctx, llmeval.Usage{
				Provider: "p", Model: "m", InputTokens: 100,
			})
			return "x", nil
		},
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
		PostChecks: []llmeval.PostCheck{pc},
	})

	require.True(t, result.Pass, "eval should pass when post-check returns true")
	require.Len(t, seen, 1)
	assert.Equal(t, 100, seen[0].InputTokens)
}

// runForPostChecks builds a minimal passing eval and attaches the given
// post-checks. Keeps the per-test boilerplate down.
func runForPostChecks(t *testing.T, checks []llmeval.PostCheck) llmeval.EvalResult {
	t.Helper()
	return llmeval.Run(t.Context(), llmeval.Eval{
		Run:        func(context.Context) (string, error) { return "x", nil },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
		PostChecks: checks,
	})
}
