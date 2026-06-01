//go:build llmeval

// Evals for the classifier SUT.
//
//	go test -tags=llmeval ./examples/classifier/
//
// Normal `go test ./...` skips this file (different build tag) — these are
// the kind of calls you don't want billed accidentally.
//
// The six evals below progress from simplest to most realistic:
//
//   - TestClassify_LabelsPositive          — strict Equal on one Run
//   - TestClassify_LabelsNegative          — same shape, negative case
//   - TestClassify_OutputIsValidLabel      — closed-set format check via OneOf
//   - TestClassify_AccuratePositive_WithDrift
//                                          — multi-run, strict format + tolerant
//                                            accuracy in the same eval
//   - TestClassify_WithBudgetEnforcement   — MaxTokens + MaxCost PostChecks
//                                            wired alongside assertions and
//                                            a judged criterion
//   - TestClassify_JudgedByFakeLLM         — local assertion + LLM-judged
//                                            criteria in a single eval (the
//                                            shape that makes the framework
//                                            worth its keep over a for-loop)
//
// The last one is the shape most real LLM evals take.

package classifier_test

import (
	"context"
	"testing"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/examples/classifier"
	"github.com/lordtatty/llmeval/llmevaltest"
)

// validLabels is the closed set the LLM is supposed to choose from.
var validLabels = []string{"positive", "negative", "neutral"}

// Strict, single-shot: clearly-positive text must be labelled "positive".
// Demonstrates: Equal, default Repeat (1).
func TestClassify_LabelsPositive(t *testing.T) {
	llmevaltest.Run(t, llmeval.Eval[string]{
		Run: func(ctx context.Context) (string, error) {
			return classifier.Classify(ctx, "I absolutely love this product!")
		},
		Assertions: []llmeval.Assertion[string]{llmeval.Equal("positive")},
	})
}

// Strict, single-shot: clearly-negative text must be labelled "negative".
func TestClassify_LabelsNegative(t *testing.T) {
	llmevaltest.Run(t, llmeval.Eval[string]{
		Run: func(ctx context.Context) (string, error) {
			return classifier.Classify(ctx, "This is the worst thing I've ever bought.")
		},
		Assertions: []llmeval.Assertion[string]{llmeval.Equal("negative")},
	})
}

// Format guarantee: the output must always be one of the valid labels.
// Catches an LLM that invents a fourth label, adds punctuation, or rambles.
// Demonstrates: OneOf.
func TestClassify_OutputIsValidLabel(t *testing.T) {
	llmevaltest.Run(t, llmeval.Eval[string]{
		Run: func(ctx context.Context) (string, error) {
			return classifier.Classify(ctx, "It's fine, I guess.")
		},
		Assertions: []llmeval.Assertion[string]{llmeval.OneOf(validLabels...)},
	})
}

// Tolerant accuracy + strict format: the LLM occasionally drifts on positive
// text, but should be right at least 60% of the time. The output format must
// be valid every time.
//
// Demonstrates: Repeat, AtLeast (tolerant), strict assertion in the same eval.
func TestClassify_AccuratePositive_WithDrift(t *testing.T) {
	llmevaltest.Run(t, llmeval.Eval[string]{
		Run: func(ctx context.Context) (string, error) {
			return classifier.FlakyClassify(ctx, "I love this!")
		},
		Repeat: 10,
		Assertions: []llmeval.Assertion[string]{
			llmeval.AtLeast(0.6, llmeval.Equal("positive")), // accuracy: ≥60% labelled positive
			llmeval.OneOf(validLabels...),                   // format: every output is a valid label
		},
	})
}

// Budget enforcement via PostCheck.
//
// Demonstrates the cost-tracking layer: the SUT and judge both record
// stub Usage records (via llmeval.RecordUsage); MaxTokens caps the raw
// token sum and MaxCost caps the dollar total resolved via a Pricer.
//
// In production swap classifier.StubPricer() for anthropic.Pricer() /
// openai.Pricer() (or compose multiples) — the eval shape doesn't change.
func TestClassify_WithBudgetEnforcement(t *testing.T) {
	llmevaltest.Run(t, llmeval.Eval[string]{
		Run: func(ctx context.Context) (string, error) {
			return classifier.Classify(ctx, "I love this product!")
		},
		Repeat:     5,
		Assertions: []llmeval.Assertion[string]{llmeval.OneOf(validLabels...)},
		Judge:      classifier.FakeJudge{},
		Criteria: []llmeval.Criterion{
			{Description: "output is a single word"},
		},
		PostChecks: []llmeval.PostCheck{
			llmeval.MaxTokens(1_000),                       // raw cap
			llmeval.MaxCost(0.01, classifier.StubPricer()), // dollar cap
		},
	})
}

// Local assertion + LLM-judged criteria, in one eval.
//
// Demonstrates the framework's real differentiator: rubric items the SUT
// must satisfy that you can't express as a pure predicate. Here we use
// classifier.FakeJudge so the example runs without an API key; swap it for
// llmeval.PromptedJudge wired to your real LLM client in production.
//
// One judge call per Run evaluates BOTH criteria in one go.
func TestClassify_JudgedByFakeLLM(t *testing.T) {
	llmevaltest.Run(t, llmeval.Eval[string]{
		Run: func(ctx context.Context) (string, error) {
			return classifier.Classify(ctx, "I love this product!")
		},
		Repeat: 3,

		// Pure check: the output is exactly "positive".
		Assertions: []llmeval.Assertion[string]{llmeval.Equal("positive")},

		// LLM-judged criteria: things a regex/equality check can't express well.
		Judge: classifier.FakeJudge{},
		Criteria: []llmeval.Criterion{
			{Description: "output is a single word"},
			{Description: "output is a valid label"},
		},
	})
}
