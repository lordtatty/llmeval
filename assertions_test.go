package llmeval_test

import (
	"context"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
)

// runWith is a tiny helper that invokes one assertion against a fixed output
// and returns the EvalResult. Assertion tests use this to keep the test body
// focused on the assertion's behaviour rather than runner plumbing.
func runWith(output string, asn llmeval.Assertion) llmeval.EvalResult {
	return llmeval.Run(context.Background(), llmeval.Eval{
		Run:     func(ctx context.Context) (string, error) { return output, nil },
		Assertions: []llmeval.Assertion{asn},
	})
}

func TestEqual_Passes(t *testing.T) {
	result := runWith("hello", llmeval.Equal("hello"))
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestEqual_Fails(t *testing.T) {
	result := runWith("goodbye", llmeval.Equal("hello"))
	assert.False(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Assertions, 1)
	assert.Equal(t, 0, result.Assertions[0].Passed)
	assert.Equal(t, 1, result.Assertions[0].Total)
}

func TestContains_Passes(t *testing.T) {
	result := runWith("hello world", llmeval.Contains("world"))
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestContains_Fails(t *testing.T) {
	result := runWith("hello", llmeval.Contains("missing"))
	assert.False(t, result.Pass, "result=%+v", result)
}

func TestNotContains_Passes(t *testing.T) {
	result := runWith("clean", llmeval.NotContains("dirty"))
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestNotContains_Fails(t *testing.T) {
	result := runWith("I'm sorry", llmeval.NotContains("sorry"))
	assert.False(t, result.Pass, "result=%+v", result)
}

func TestOneOf_Passes(t *testing.T) {
	for _, out := range []string{"positive", "negative", "neutral"} {
		t.Run(out, func(t *testing.T) {
			result := runWith(out, llmeval.OneOf("positive", "negative", "neutral"))
			assert.True(t, result.Pass, "result=%+v", result)
		})
	}
}

func TestOneOf_Fails(t *testing.T) {
	result := runWith("maybe", llmeval.OneOf("positive", "negative", "neutral"))
	assert.False(t, result.Pass, "result=%+v", result)
}

func TestOneOf_Strict_RejectsSurroundingWhitespace(t *testing.T) {
	// "positive " (trailing space) is not equal to "positive". OneOf is strict
	// by design — fuzzy normalisation belongs upstream.
	result := runWith("positive ", llmeval.OneOf("positive", "negative", "neutral"))
	assert.False(t, result.Pass, "result=%+v", result)
}

func TestMatches_Passes(t *testing.T) {
	result := runWith("42 items", llmeval.Matches(regexp.MustCompile(`^\d+ items$`)))
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestMatches_Fails(t *testing.T) {
	result := runWith("not numeric", llmeval.Matches(regexp.MustCompile(`^\d+ items$`)))
	assert.False(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Runs, 1)
	require.Len(t, result.Runs[0].Assertions, 1)
	assert.Equal(t, "no match", result.Runs[0].Assertions[0].Reason)
}

func TestCheck_Adapter(t *testing.T) {
	result := runWith("abc", llmeval.Check("len == 3", func(o string) (bool, string) {
		if len(o) == 3 {
			return true, ""
		}
		return false, "wrong length"
	}))
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestAtLeast_PassesAboveThreshold(t *testing.T) {
	// Alternates: hello,goodbye,hello,goodbye,hello → 3/5 match (60% ≥ 50%)
	calls := 0
	outputs := []string{"hello", "goodbye", "hello", "goodbye", "hello"}
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run: func(ctx context.Context) (string, error) {
			out := outputs[calls]
			calls++
			return out, nil
		},
		Repeat:  5,
		Assertions: []llmeval.Assertion{llmeval.AtLeast(0.5, llmeval.Equal("hello"))},
	})
	assert.True(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Assertions, 1)
	assert.Equal(t, 0.5, result.Assertions[0].MinRate)
	assert.Equal(t, 3, result.Assertions[0].Passed)
}

func TestAtLeast_ClampsNegativeRateToZero(t *testing.T) {
	// rate=-0.5 is clamped to 0; with 0 successes that's still 0/1 ≥ 0 ⇒ pass.
	result := runWith("anything", llmeval.AtLeast(-0.5, llmeval.Equal("nope")))
	assert.True(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Assertions, 1)
	assert.Equal(t, 0.0, result.Assertions[0].MinRate)
}

func TestAtLeast_ClampsRateAboveOneToOne(t *testing.T) {
	// rate=2.0 is clamped to 1.0; equivalent to strict.
	result := runWith("hello", llmeval.AtLeast(2.0, llmeval.Equal("hello")))
	assert.True(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Assertions, 1)
	assert.Equal(t, 1.0, result.Assertions[0].MinRate)
}

func TestAtLeast_FailsBelowThreshold(t *testing.T) {
	calls := 0
	outputs := []string{"hello", "goodbye", "goodbye", "goodbye", "goodbye"} // 1/5 = 20%
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run: func(ctx context.Context) (string, error) {
			out := outputs[calls]
			calls++
			return out, nil
		},
		Repeat:  5,
		Assertions: []llmeval.Assertion{llmeval.AtLeast(0.5, llmeval.Equal("hello"))},
	})
	assert.False(t, result.Pass, "result=%+v", result)
}
