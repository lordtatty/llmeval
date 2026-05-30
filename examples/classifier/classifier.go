// Package classifier is a stand-in for an LLM-driven sentiment classifier.
//
// In your project, the SUT would call your real LLM client — something like
// "send a prompt to Claude/GPT/etc., parse the response, return the label."
// Here we fake it with keyword matching so the examples can run without
// API keys or network access.
package classifier

import (
	"context"
	"math/rand/v2"
	"strings"
)

// Classify returns "positive", "negative", or "neutral" for the given text.
//
// Pretend this is calling an LLM with a prompt like:
//
//	"Classify the sentiment of the following text as
//	 'positive', 'negative', or 'neutral'. Reply with one word only.\n\n" + text
//
// In real code: out, err := myLLMClient.Complete(ctx, prompt) ...
func Classify(_ context.Context, text string) (string, error) {
	lower := strings.ToLower(text)
	if containsAny(lower, "love", "great", "amazing", "excellent", "wonderful") {
		return "positive", nil
	}
	if containsAny(lower, "hate", "terrible", "awful", "bad", "worst") {
		return "negative", nil
	}
	return "neutral", nil
}

// FlakyClassify is the same as Classify but simulates LLM drift:
// 70% of the time it returns the "correct" label; 30% of the time it
// returns a random alternative. Used to demonstrate Repeat + AtLeast.
func FlakyClassify(ctx context.Context, text string) (string, error) {
	correct, _ := Classify(ctx, text)
	if rand.IntN(10) < 7 {
		return correct, nil
	}
	labels := []string{"positive", "negative", "neutral"}
	return labels[rand.IntN(3)], nil
}

func containsAny(s string, terms ...string) bool {
	for _, t := range terms {
		if strings.Contains(s, t) {
			return true
		}
	}
	return false
}
