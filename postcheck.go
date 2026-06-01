package llmeval

import "fmt"

// PostCheck fires once after all runs complete and has access to the
// aggregated EvalSummary (the non-generic view of EvalResult, excluding
// per-Run typed outputs). Use it for budget assertions and other policy
// checks that only make sense after the whole eval is done — see MaxCost
// for the canonical example.
//
// PostCheck is intentionally non-generic so it composes cleanly into any
// Eval[T] without needing inference of T at every call site.
//
// Check returns (pass, reason). Reason is surfaced on PostCheckResult
// and in PrintText / PrintJSON; keep it short and concrete (numbers,
// what limit was exceeded) so failures are debuggable from the log
// alone.
type PostCheck struct {
	Name  string
	Check func(EvalSummary) (pass bool, reason string)
}

// PostCheckResult is one PostCheck's outcome, recorded on EvalResult.
type PostCheckResult struct {
	Name   string `json:"name"`
	Pass   bool   `json:"pass"`
	Reason string `json:"reason,omitempty"`
}

// MaxCost returns a PostCheck that fails when the total dollar cost of
// EvalResult.Usage (resolved via the supplied pricers, first-match-wins)
// exceeds limit. Use to enforce a per-eval budget in CI or for one-off
// cost-bounded development runs.
//
// Fail-closed semantics:
//
//   - If any Usage record can't be priced by the given pricers (no pricer
//     matched the provider/model), MaxCost fails. The alternative — treat
//     unpriced calls as $0 — would silently miss costs when adding a new
//     provider or model and is the wrong default for a budget assertion.
//     Pass a catch-all pricer (e.g. one that returns `0, true` for the
//     providers you want to ignore) to opt out.
//
//   - "spent == limit" passes — the convention is that limit is the
//     maximum you're willing to spend, not a strict-less-than threshold.
//
//   - nil entries in the pricers variadic are skipped (no panic).
func MaxCost(limit float64, pricers ...Pricer) PostCheck {
	return PostCheck{
		Name: fmt.Sprintf("max cost: $%.2f", limit),
		Check: func(s EvalSummary) (bool, string) {
			spent, unpriced := costBreakdown(s.Usage, pricers)
			if unpriced > 0 {
				return false, fmt.Sprintf(
					"%d usage record(s) had no matching pricer — can't certify budget",
					unpriced,
				)
			}
			if spent > limit {
				return false, fmt.Sprintf("spent $%.4f, limit $%.4f", spent, limit)
			}
			return true, ""
		},
	}
}

// MaxTokens returns a PostCheck that fails when the total token count
// (input + output, summed across every recorded Usage) exceeds limit.
// Use when you want a budget that doesn't depend on a price table — handy
// for caps expressed against rate-limit quotas rather than dollars.
//
// "used == limit" passes (limit is the ceiling, not strict-less-than),
// matching MaxCost's convention. An eval with no recorded Usage uses 0
// tokens and therefore always passes.
func MaxTokens(limit int) PostCheck {
	return PostCheck{
		Name: fmt.Sprintf("max tokens: %d", limit),
		Check: func(s EvalSummary) (bool, string) {
			used := 0
			for _, u := range s.Usage {
				used += u.InputTokens + u.OutputTokens
			}
			if used > limit {
				return false, fmt.Sprintf("used %d, limit %d", used, limit)
			}
			return true, ""
		},
	}
}
