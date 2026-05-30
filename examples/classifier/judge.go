package classifier

import (
	"context"
	"strings"

	"github.com/lordtatty/llmeval"
)

// FakeJudge is a stand-in for an LLM-driven judge. In a real project you'd
// use llmeval.PromptedJudge wired to your LLM client (Anthropic, OpenAI, etc.)
// — that judge sends the output plus criteria to the model, parses the
// PASS/FAIL verdicts back, and returns them.
//
// This fake skips the LLM call: it inspects criteria descriptions for a few
// known keywords and decides PASS/FAIL based on whether the output meets the
// stub rubric. It exists only so the example evals run without an API key.
type FakeJudge struct{}

// Evaluate implements llmeval.Judge using heuristic stand-ins for each
// supported criterion. Returns one verdict per criterion, in order.
func (FakeJudge) Evaluate(_ context.Context, output string, criteria []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
	verdicts := make([]llmeval.CriterionResult, len(criteria))
	for i, c := range criteria {
		verdicts[i] = judgeOne(output, c.Description)
	}
	return verdicts, nil
}

// judgeOne is the tiny heuristic that stands in for a real LLM rubric check.
func judgeOne(output, criterion string) llmeval.CriterionResult {
	lower := strings.ToLower(criterion)
	switch {
	case strings.Contains(lower, "single word"):
		if len(strings.Fields(output)) == 1 {
			return llmeval.CriterionResult{Pass: true, Reason: "output is one word"}
		}
		return llmeval.CriterionResult{Pass: false, Reason: "output is more than one word"}
	case strings.Contains(lower, "valid label"):
		if output == "positive" || output == "negative" || output == "neutral" {
			return llmeval.CriterionResult{Pass: true, Reason: "matches the closed label set"}
		}
		return llmeval.CriterionResult{Pass: false, Reason: "not a recognised label"}
	default:
		// Unknown criterion in the stub — pass by default rather than block
		// the example with arbitrary fails.
		return llmeval.CriterionResult{Pass: true, Reason: "stub judge: no rule for this criterion"}
	}
}
