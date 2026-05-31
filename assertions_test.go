package llmeval_test

import (
	"context"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
)

// runWith executes a single-shot eval over the given SUT output and one
// assertion, returning the result. Most tests below specify just that:
// "given output X and assertion A, the eval passes/fails."
func runWith(output string, asn llmeval.Assertion) llmeval.EvalResult {
	return llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(context.Context) (string, error) { return output, nil },
		Assertions: []llmeval.Assertion{asn},
	})
}

// ── Equal ────────────────────────────────────────────────────────────────────

func TestEqualPassesWhenOutputMatchesExactly(t *testing.T) {
	result := runWith("hello", llmeval.Equal("hello"))
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestEqualFailsWhenOutputDiffers(t *testing.T) {
	result := runWith("goodbye", llmeval.Equal("hello"))
	assert.False(t, result.Pass, "result=%+v", result)
	assert.Equal(t, 0, result.Assertions[0].Passed)
}

// ── Contains / NotContains ──────────────────────────────────────────────────

func TestContainsPassesWhenSubstringIsPresent(t *testing.T) {
	assert.True(t, runWith("hello world", llmeval.Contains("world")).Pass)
}

func TestContainsFailsWhenSubstringIsAbsent(t *testing.T) {
	assert.False(t, runWith("hello", llmeval.Contains("missing")).Pass)
}

func TestNotContainsPassesWhenSubstringIsAbsent(t *testing.T) {
	assert.True(t, runWith("clean", llmeval.NotContains("dirty")).Pass)
}

func TestNotContainsFailsWhenSubstringIsPresent(t *testing.T) {
	assert.False(t, runWith("I'm sorry", llmeval.NotContains("sorry")).Pass)
}

// ── OneOf ────────────────────────────────────────────────────────────────────

func TestOneOfPassesWhenOutputMatchesAnyAllowedValue(t *testing.T) {
	allowed := []string{"positive", "negative", "neutral"}
	for _, out := range allowed {
		t.Run(out, func(t *testing.T) {
			assert.True(t, runWith(out, llmeval.OneOf(allowed...)).Pass)
		})
	}
}

func TestOneOfFailsWhenOutputMatchesNoAllowedValue(t *testing.T) {
	assert.False(t, runWith("maybe", llmeval.OneOf("positive", "negative", "neutral")).Pass)
}

func TestOneOfIsStrictAndRejectsTrailingWhitespace(t *testing.T) {
	// OneOf is documented strict; lenient matching is the caller's job.
	assert.False(t, runWith("positive ", llmeval.OneOf("positive", "negative", "neutral")).Pass)
}

// ── Matches ──────────────────────────────────────────────────────────────────

func TestMatchesPassesWhenRegexMatches(t *testing.T) {
	assert.True(t, runWith("42 items", llmeval.Matches(regexp.MustCompile(`^\d+ items$`))).Pass)
}

func TestMatchesFailsWhenRegexDoesNotMatch(t *testing.T) {
	result := runWith("not numeric", llmeval.Matches(regexp.MustCompile(`^\d+ items$`)))
	assert.False(t, result.Pass, "result=%+v", result)
	assert.Equal(t, "no match", result.Runs[0].Assertions[0].Reason)
}

// ── Check (custom predicate adapter) ────────────────────────────────────────

func TestCheckAdaptsACustomPredicateIntoAnAssertion(t *testing.T) {
	exactlyThreeChars := llmeval.Check("len == 3", func(o string) (bool, string) {
		if len(o) == 3 {
			return true, ""
		}
		return false, "wrong length"
	})
	assert.True(t, runWith("abc", exactlyThreeChars).Pass)
}

// ── CheckJSON (typed predicate over JSON-encoded output) ───────────────────

func TestCheckJSONPassesWhenPredicateAcceptsTheDecodedValue(t *testing.T) {
	type response struct{ Category string }
	asn := llmeval.CheckJSON("category", func(r response) (bool, string) {
		if r.Category == "positive" {
			return true, ""
		}
		return false, "got " + r.Category
	})
	assert.True(t, runWith(`{"Category":"positive"}`, asn).Pass)
}

func TestCheckJSONFailsWithReasonWhenPredicateRejects(t *testing.T) {
	type response struct{ Category string }
	asn := llmeval.CheckJSON("category", func(r response) (bool, string) {
		if r.Category == "positive" {
			return true, ""
		}
		return false, "got " + r.Category
	})
	result := runWith(`{"Category":"negative"}`, asn)
	require.False(t, result.Pass)
	require.Len(t, result.Runs, 1)
	assert.Contains(t, result.Runs[0].Assertions[0].Reason, "negative")
}

func TestCheckJSONFailsWithUnmarshalReasonWhenOutputIsNotValidJSON(t *testing.T) {
	type response struct{ Category string }
	// Predicate must not be called on invalid JSON — we use a flag rather
	// than t.Fatal so the test still exposes the assertion's failure
	// reason rather than dying inside the closure.
	called := false
	asn := llmeval.CheckJSON("category", func(response) (bool, string) {
		called = true
		return false, ""
	})
	result := runWith("not json at all", asn)
	require.False(t, result.Pass)
	assert.False(t, called, "predicate should be skipped on invalid JSON")
	assert.Contains(t, result.Runs[0].Assertions[0].Reason, "not valid JSON")
}

func TestCheckJSONFailsWithReasonWhenOutputIsLiteralJSONNull(t *testing.T) {
	// `null` decodes successfully to zero-value T — left unchecked, the
	// predicate would then report a per-field miss for the user instead of
	// the real "model returned JSON null" failure. Catch it explicitly.
	type response struct{ Category string }
	called := false
	asn := llmeval.CheckJSON("category", func(response) (bool, string) {
		called = true
		return true, ""
	})
	result := runWith("null", asn)
	require.False(t, result.Pass)
	assert.False(t, called, "predicate should not see JSON null")
	assert.Contains(t, result.Runs[0].Assertions[0].Reason, "null")
}

func TestCheckJSONHandlesPrimitiveTypes(t *testing.T) {
	// T can be any json.Unmarshal-compatible type, not just a struct.
	asn := llmeval.CheckJSON("positive int", func(n int) (bool, string) {
		if n > 0 {
			return true, ""
		}
		return false, "non-positive"
	})
	assert.True(t, runWith("42", asn).Pass)
}

// ── AtLeast (tolerance wrapper) ─────────────────────────────────────────────
//
// AtLeast needs multiple runs to be meaningful — these tests use a
// sequenced SUT directly rather than runWith.

func runOverSequence(outputs []string, asn llmeval.Assertion) llmeval.EvalResult {
	i := 0
	return llmeval.Run(context.Background(), llmeval.Eval{
		Run: func(context.Context) (string, error) {
			out := outputs[i]
			i++
			return out, nil
		},
		Repeat:     len(outputs),
		Assertions: []llmeval.Assertion{asn},
	})
}

func TestAtLeastPassesWhenPassRateMeetsThreshold(t *testing.T) {
	// 3 of 5 match → 60%, meets the ≥50% threshold.
	result := runOverSequence(
		[]string{"hello", "goodbye", "hello", "goodbye", "hello"},
		llmeval.AtLeast(0.5, llmeval.Equal("hello")),
	)
	assert.True(t, result.Pass, "result=%+v", result)
	assert.Equal(t, 3, result.Assertions[0].Passed)
	assert.Equal(t, 0.5, result.Assertions[0].MinRate)
}

func TestAtLeastFailsWhenPassRateDropsBelowThreshold(t *testing.T) {
	// 1 of 5 matches → 20%, below ≥50%.
	result := runOverSequence(
		[]string{"hello", "goodbye", "goodbye", "goodbye", "goodbye"},
		llmeval.AtLeast(0.5, llmeval.Equal("hello")),
	)
	assert.False(t, result.Pass, "result=%+v", result)
}

func TestAtLeastClampsNegativeRateToZero(t *testing.T) {
	// rate=-0.5 is clamped to 0; effectively "any rate is acceptable."
	result := runWith("anything", llmeval.AtLeast(-0.5, llmeval.Equal("nope")))
	assert.True(t, result.Pass, "result=%+v", result)
	assert.Equal(t, 0.0, result.Assertions[0].MinRate)
}

func TestAtLeastClampsRateAboveOneToOne(t *testing.T) {
	// rate=2.0 is clamped to 1.0; effectively strict.
	result := runWith("hello", llmeval.AtLeast(2.0, llmeval.Equal("hello")))
	assert.True(t, result.Pass, "result=%+v", result)
	assert.Equal(t, 1.0, result.Assertions[0].MinRate)
}
