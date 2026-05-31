package llmeval

import "fmt"

// PostCheck fires once after all runs complete and has access to the
// fully aggregated EvalResult. Use it for budget assertions and other
// policy checks that only make sense after the whole eval is done — see
// MaxCost for the canonical example.
//
// Check returns (pass, reason). Reason is surfaced on PostCheckResult
// and in PrintText / PrintJSON; keep it short and concrete (numbers,
// what limit was exceeded) so failures are debuggable from the log
// alone.
type PostCheck struct {
	Name  string
	Check func(EvalResult) (pass bool, reason string)
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
		Check: func(r EvalResult) (bool, string) {
			spent, unpriced := costBreakdown(r.Usage, pricers)
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
