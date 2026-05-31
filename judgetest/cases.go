// Package judgetest exposes a curated set of hand-picked judge probes plus
// a per-case assertion helper. Callers iterate Cases themselves with t.Run,
// so the test makes its structure visible (loop, per-case sub-tests, what's
// asserted) instead of hiding it behind a single opaque call.
//
// Cases are intentionally unambiguous — any reasonable model should agree.
// If one flakes, the case is wrong, not the model.
package judgetest

import "github.com/lordtatty/llmeval"

// Case is one judge probe: an output, the criteria to judge it against,
// and the expected pass/fail for each criterion.
type Case struct {
	// Name labels the case in test failure messages and sub-test output.
	Name string

	// Output is what we pretend the SUT produced.
	Output string

	// Criteria is the rubric list passed to Judge.Evaluate.
	Criteria []llmeval.Criterion

	// Wants is the expected Pass value per criterion, parallel to Criteria.
	Wants []bool
}

// Cases is the curated probe set. Each entry is a slam-dunk; any reasonable
// LLM should agree on the verdict. Replace any case that becomes flaky —
// we're testing the prompt + parser contract, not the model's reasoning.
var Cases = []Case{
	{
		Name:     "one-word reply is one-sentence-or-shorter",
		Output:   "yes.",
		Criteria: []llmeval.Criterion{{Description: "is one sentence or shorter"}},
		Wants:    []bool{true},
	},
	{
		Name:     "one-word reply is not three-sentences-long",
		Output:   "yes.",
		Criteria: []llmeval.Criterion{{Description: "is at least three sentences long"}},
		Wants:    []bool{false},
	},
	{
		Name:     "refusal does not answer the question",
		Output:   "I cannot help with that.",
		Criteria: []llmeval.Criterion{{Description: "answers the question directly"}},
		Wants:    []bool{false},
	},
	{
		Name:   "code reply hits both language and length criteria",
		Output: "JavaScript uses prototypal inheritance.",
		Criteria: []llmeval.Criterion{
			{Description: "mentions a programming language"},
			{Description: "is at least one sentence"},
		},
		Wants: []bool{true, true},
	},
	{
		Name:   "off-topic reply passes length but not language",
		Output: "I love cats.",
		Criteria: []llmeval.Criterion{
			{Description: "mentions a programming language"},
			{Description: "is at least one sentence"},
		},
		Wants: []bool{false, true},
	},
}
