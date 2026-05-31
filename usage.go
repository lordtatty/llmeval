package llmeval

import (
	"context"
	"sync"
)

// Usage records token consumption from one LLM call. Sub-modules' judges
// record this automatically when ctx carries a collector; users record
// their own SUT LLM calls by calling RecordUsage from inside Eval.Run.
type Usage struct {
	Provider     string `json:"provider"`
	Model        string `json:"model"`
	InputTokens  int    `json:"inputTokens"`
	OutputTokens int    `json:"outputTokens"`
}

// Pricer maps one Usage to its dollar cost. Returns ok=false when this
// pricer doesn't recognise the (provider, model) — TotalCost then tries
// the next pricer in line. Sub-modules expose a Pricer() function each.
type Pricer func(Usage) (cost float64, ok bool)

// UsageCollector accumulates Usage records across an eval. Safe for
// concurrent use; Eval.Concurrency > 1 means many goroutines may record
// simultaneously.
type UsageCollector struct {
	mu      sync.Mutex
	records []Usage
}

// Records returns a copy of the raw per-call usage list in record order.
// Mutating the returned slice does not affect the collector.
func (c *UsageCollector) Records() []Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Usage, len(c.records))
	copy(out, c.records)
	return out
}

// Aggregated sums InputTokens and OutputTokens by (Provider, Model),
// returning one Usage per distinct pair in first-seen order.
func (c *UsageCollector) Aggregated() []Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return aggregateUsages(c.records)
}

func aggregateUsages(records []Usage) []Usage {
	type key struct{ provider, model string }
	byKey := map[key]int{} // key → index in out
	var out []Usage
	for _, u := range records {
		k := key{u.Provider, u.Model}
		if i, ok := byKey[k]; ok {
			out[i].InputTokens += u.InputTokens
			out[i].OutputTokens += u.OutputTokens
			continue
		}
		byKey[k] = len(out)
		out = append(out, u)
	}
	return out
}

type usageCtxKey struct{}

// RecordUsage records u into the collector attached to ctx. No-op if ctx
// has no collector — safe to call unconditionally from sub-modules or
// from inside Eval.Run.
func RecordUsage(ctx context.Context, u Usage) {
	c, _ := ctx.Value(usageCtxKey{}).(*UsageCollector)
	if c == nil {
		return
	}
	c.mu.Lock()
	c.records = append(c.records, u)
	c.mu.Unlock()
}

// NewUsageCtx attaches a fresh UsageCollector to ctx and returns both.
// Use this when calling judges outside of llmeval.Run (e.g., from
// judgetest helpers or direct sub-module use) so RecordUsage has
// somewhere to write. Run handles this internally; you don't need it
// for typical evals.
func NewUsageCtx(ctx context.Context) (context.Context, *UsageCollector) {
	c := &UsageCollector{}
	return context.WithValue(ctx, usageCtxKey{}, c), c
}

// TotalCost sums dollar cost across usages. For each Usage, pricers are
// tried in order; the first that returns ok=true wins. nil pricers in the
// variadic are skipped. Usages with no matching pricer contribute $0 —
// provide your own Pricer to cover them or to override a sub-module's
// default rate (the user-supplied pricer runs first because it's earlier
// in the list). To detect unpriced usages (e.g. for budget enforcement),
// use MaxCost which fails closed in that case.
func TotalCost(usages []Usage, pricers ...Pricer) float64 {
	total, _ := costBreakdown(usages, pricers)
	return total
}

// costBreakdown is the shared cost-resolution loop. Returns the dollar
// total of priced usages and the count of usages no pricer matched.
// Unexported because the (total, unpriced) split is an internal detail
// MaxCost uses to fail closed; TotalCost discards the unpriced count.
func costBreakdown(usages []Usage, pricers []Pricer) (total float64, unpriced int) {
	for _, u := range usages {
		matched := false
		for _, p := range pricers {
			if p == nil {
				continue
			}
			if c, ok := p(u); ok {
				total += c
				matched = true
				break
			}
		}
		if !matched {
			unpriced++
		}
	}
	return total, unpriced
}

// Price is the per-million-token rate pair (input, output) in USD for one
// model. Sub-modules' price tables hold these.
type Price struct {
	Input  float64
	Output float64
}

// Cost returns the dollar cost of u at this rate.
func (p Price) Cost(u Usage) float64 {
	return (float64(u.InputTokens)*p.Input + float64(u.OutputTokens)*p.Output) / 1_000_000
}

// NewPricer builds a Pricer for one provider. Sub-modules wrap their own
// SDK-typed price tables with this helper so the (provider-name check +
// model lookup + cost calc) plumbing lives in one place.
//
// lookup receives the raw model string from Usage.Model — typically the
// caller converts it to their SDK's Model type and indexes their internal
// table.
func NewPricer(provider string, lookup func(model string) (Price, bool)) Pricer {
	return func(u Usage) (float64, bool) {
		if u.Provider != provider {
			return 0, false
		}
		p, ok := lookup(u.Model)
		if !ok {
			return 0, false
		}
		return p.Cost(u), true
	}
}
