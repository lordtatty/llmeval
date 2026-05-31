package llmeval_test

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
)

// ─────────────────────────────────────────────────────────────────────────────
// RecordUsage + NewUsageCtx + UsageCollector
// ─────────────────────────────────────────────────────────────────────────────

func TestRecordUsageIsANoOpWhenCtxHasNoCollector(t *testing.T) {
	// Plain background ctx — no panic, no error, just nothing happens.
	llmeval.RecordUsage(context.Background(), llmeval.Usage{
		Provider: "anthropic", Model: "haiku", InputTokens: 1,
	})
}

func TestNewUsageCtxAttachesAFreshCollectorThatRecordsCanWriteInto(t *testing.T) {
	ctx, c := llmeval.NewUsageCtx(context.Background())
	require.NotNil(t, c)

	llmeval.RecordUsage(ctx, llmeval.Usage{
		Provider: "anthropic", Model: "haiku", InputTokens: 10, OutputTokens: 5,
	})

	records := c.Records()
	require.Len(t, records, 1)
	assert.Equal(t, "anthropic", records[0].Provider)
	assert.Equal(t, 10, records[0].InputTokens)
}

func TestUsageCollectorRecordsPreservesEveryCallInOrder(t *testing.T) {
	ctx, c := llmeval.NewUsageCtx(context.Background())
	llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "p", Model: "m1", InputTokens: 1})
	llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "p", Model: "m1", InputTokens: 2})
	llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "p", Model: "m2", InputTokens: 4})

	records := c.Records()
	require.Len(t, records, 3)
	assert.Equal(t, 1, records[0].InputTokens)
	assert.Equal(t, 2, records[1].InputTokens)
	assert.Equal(t, 4, records[2].InputTokens)
}

func TestUsageCollectorAggregatedSumsTokensByProviderAndModel(t *testing.T) {
	ctx, c := llmeval.NewUsageCtx(context.Background())
	llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "p", Model: "m1", InputTokens: 10, OutputTokens: 5})
	llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "p", Model: "m1", InputTokens: 20, OutputTokens: 7})
	llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "p", Model: "m2", InputTokens: 3, OutputTokens: 1})

	agg := c.Aggregated()
	require.Len(t, agg, 2)

	// Order matches first-seen order.
	assert.Equal(t, "m1", agg[0].Model)
	assert.Equal(t, 30, agg[0].InputTokens)
	assert.Equal(t, 12, agg[0].OutputTokens)
	assert.Equal(t, "m2", agg[1].Model)
	assert.Equal(t, 3, agg[1].InputTokens)
}

func TestRecordUsageIsSafeForConcurrentCallers(t *testing.T) {
	// Repeat counter is meaningful here: Eval.Concurrency > 1 means many
	// goroutines call RecordUsage simultaneously, so the collector must
	// serialise appends. The aggregate count proves nothing was lost.
	ctx, c := llmeval.NewUsageCtx(context.Background())
	const n = 200
	var wg sync.WaitGroup
	for range n {
		wg.Go(func() {
			llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "x", Model: "y", InputTokens: 1})
		})
	}
	wg.Wait()

	agg := c.Aggregated()
	require.Len(t, agg, 1)
	assert.Equal(t, n, agg[0].InputTokens)
}

func TestUsageCollectorRecordsReturnsACopySoCallersCannotMutateInternals(t *testing.T) {
	ctx, c := llmeval.NewUsageCtx(context.Background())
	llmeval.RecordUsage(ctx, llmeval.Usage{Provider: "p", Model: "m", InputTokens: 1})

	records := c.Records()
	records[0].InputTokens = 9999 // mutate the returned slice

	// Internal state untouched.
	assert.Equal(t, 1, c.Records()[0].InputTokens)
}

// ─────────────────────────────────────────────────────────────────────────────
// TotalCost
// ─────────────────────────────────────────────────────────────────────────────

func TestTotalCostSumsAcrossUsagesUsingFirstMatchingPricer(t *testing.T) {
	anth := func(u llmeval.Usage) (float64, bool) {
		if u.Provider == "anthropic" {
			return 0.10, true
		}
		return 0, false
	}
	oai := func(u llmeval.Usage) (float64, bool) {
		if u.Provider == "openai" {
			return 0.05, true
		}
		return 0, false
	}

	usages := []llmeval.Usage{
		{Provider: "anthropic", Model: "haiku"},
		{Provider: "openai", Model: "gpt-4.1-mini"},
		{Provider: "anthropic", Model: "haiku"},
		{Provider: "unknown", Model: "x"},
	}

	total := llmeval.TotalCost(usages, anth, oai)
	assert.InDelta(t, 0.10+0.05+0.10, total, 1e-9)
}

func TestTotalCostIsZeroWhenNoPricerMatches(t *testing.T) {
	usages := []llmeval.Usage{
		{Provider: "unknown", Model: "x", InputTokens: 1000},
	}
	assert.Equal(t, float64(0), llmeval.TotalCost(usages))
}

func TestTotalCostStopsAtTheFirstMatchingPricerEvenIfLaterOnesWouldAlsoMatch(t *testing.T) {
	// First pricer wins. Documents the convention so users can layer
	// custom price tables before the defaults.
	first := func(_ llmeval.Usage) (float64, bool) { return 0.01, true }
	second := func(_ llmeval.Usage) (float64, bool) { return 999.99, true }

	total := llmeval.TotalCost([]llmeval.Usage{{Provider: "p"}}, first, second)
	assert.InDelta(t, 0.01, total, 1e-9)
}

func TestTotalCostSkipsNilPricersInTheVariadic(t *testing.T) {
	// A nil entry shouldn't panic; it's treated as if absent.
	pricer := func(_ llmeval.Usage) (float64, bool) { return 0.05, true }
	total := llmeval.TotalCost([]llmeval.Usage{{Provider: "p"}}, nil, pricer)
	assert.InDelta(t, 0.05, total, 1e-9)
}

// ─────────────────────────────────────────────────────────────────────────────
// Price + NewPricer
// ─────────────────────────────────────────────────────────────────────────────

func TestPriceCostIsZeroForZeroTokens(t *testing.T) {
	p := llmeval.Price{Input: 1.00, Output: 5.00}
	assert.InDelta(t, 0.0, p.Cost(llmeval.Usage{}), 1e-9)
}

func TestPriceCostScalesLinearlyAcrossInputAndOutputRates(t *testing.T) {
	p := llmeval.Price{Input: 2.00, Output: 8.00}
	// 1M input × $2 + 1M output × $8 = $10.
	cost := p.Cost(llmeval.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	assert.InDelta(t, 10.0, cost, 1e-9)
}

func TestNewPricerReturnsOkFalseForADifferentProvider(t *testing.T) {
	p := llmeval.NewPricer("anthropic", func(string) (llmeval.Price, bool) {
		return llmeval.Price{Input: 1.00, Output: 5.00}, true
	})
	_, ok := p(llmeval.Usage{Provider: "openai"})
	assert.False(t, ok)
}

func TestNewPricerReturnsOkFalseWhenLookupRejectsModel(t *testing.T) {
	p := llmeval.NewPricer("anthropic", func(string) (llmeval.Price, bool) {
		return llmeval.Price{}, false
	})
	_, ok := p(llmeval.Usage{Provider: "anthropic", Model: "ghost"})
	assert.False(t, ok)
}

func TestNewPricerComputesCostWhenProviderAndLookupBothMatch(t *testing.T) {
	p := llmeval.NewPricer("anthropic", func(model string) (llmeval.Price, bool) {
		assert.Equal(t, "claude-x", model, "lookup should receive Usage.Model")
		return llmeval.Price{Input: 1.00, Output: 5.00}, true
	})
	cost, ok := p(llmeval.Usage{
		Provider:     "anthropic",
		Model:        "claude-x",
		InputTokens:  1_000_000,
		OutputTokens: 1_000_000,
	})
	require.True(t, ok)
	assert.InDelta(t, 6.0, cost, 1e-9)
}
