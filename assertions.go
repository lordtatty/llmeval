package llmeval

import (
	"context"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// Equal returns an Assertion that passes when the output exactly equals want.
// Works for any comparable T — string, numeric, bool, fixed-size arrays, and
// pointer-comparable struct types.
func Equal[T comparable](want T) Assertion[T] {
	return predicate(fmt.Sprintf("equal: %s", formatValue(want)), func(o T) (bool, string) {
		if o == want {
			return true, ""
		}
		return false, fmt.Sprintf("got %s", formatValue(o))
	})
}

// OneOf returns an Assertion that passes when the output is exactly one of
// values. Works for any comparable T. Strict equality — surrounding
// whitespace, case differences, or trailing punctuation on string outputs
// fail. Normalise upstream (in your Run closure) if you need lenient
// matching.
func OneOf[T comparable](values ...T) Assertion[T] {
	joined := joinValues(values)
	return predicate(fmt.Sprintf("one of: %s", joined), func(o T) (bool, string) {
		if slices.Contains(values, o) {
			return true, ""
		}
		return false, fmt.Sprintf("got %s, expected one of: %s", formatValue(o), joined)
	})
}

// joinValues renders a comma-separated list with each value rendered via
// fmt's default %v (no quoting). Preserves OneOf's historic "positive,
// negative, neutral" output shape for string T.
func joinValues[T any](values []T) string {
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = fmt.Sprintf("%v", v)
	}
	return strings.Join(parts, ", ")
}

// formatValue renders a value for inclusion in an assertion name or reason.
// Strings get quoted (matching the historic %q behaviour from when Equal /
// OneOf were string-only); everything else uses Go's default %v.
func formatValue[T any](v T) string {
	if s, ok := any(v).(string); ok {
		return fmt.Sprintf("%q", s)
	}
	return fmt.Sprintf("%v", v)
}

// ContainsString returns an Assertion that passes when the string output
// contains substr. String-only because Contains is intrinsically a string
// operation; for struct outputs, use Check[T] with strings.Contains on a
// specific field.
func ContainsString(substr string) Assertion[string] {
	return predicate(fmt.Sprintf("contains: %q", substr), func(o string) (bool, string) {
		if strings.Contains(o, substr) {
			return true, ""
		}
		return false, fmt.Sprintf("missing %q", substr)
	})
}

// NotContainsString returns an Assertion that passes when the string output
// does not contain substr. String-only; see ContainsString.
func NotContainsString(substr string) Assertion[string] {
	return predicate(fmt.Sprintf("not contains: %q", substr), func(o string) (bool, string) {
		if !strings.Contains(o, substr) {
			return true, ""
		}
		return false, fmt.Sprintf("contained %q", substr)
	})
}

// MatchesString returns an Assertion that passes when re matches the string
// output. String-only because regex is intrinsically a string operation;
// for struct outputs, use Check[T] with re.MatchString on a field.
func MatchesString(re *regexp.Regexp) Assertion[string] {
	return predicate(fmt.Sprintf("matches: %s", re), func(o string) (bool, string) {
		if re.MatchString(o) {
			return true, ""
		}
		return false, "no match"
	})
}

// Check adapts a user-supplied predicate into an Assertion. Use when the
// built-ins don't fit — including when you want to call out to testify's
// ObjectsAreEqual, go-cmp, or any other library that returns a bool — or
// when you have a struct output and want to inspect specific fields.
func Check[T any](name string, fn func(output T) (pass bool, reason string)) Assertion[T] {
	return predicate(name, fn)
}

// AtLeast wraps a to lower its required pass rate. AtLeast(0.8, asn) means
// "asn must pass on at least 80% of the eval's repeats." Without AtLeast,
// every assertion is strict (must pass every repeat).
//
// rate is clamped to [0, 1].
func AtLeast[T any](rate float64, a Assertion[T]) Assertion[T] {
	if rate < 0 {
		rate = 0
	}
	if rate > 1 {
		rate = 1
	}
	return rateOverride[T]{Assertion: a, rate: rate}
}

// predicate is the internal Assertion implementation backing the helpers
// above (and Check). All helpers are strict (MinPassRate 1.0); use AtLeast
// to tolerate failures.
func predicate[T any](name string, fn func(T) (bool, string)) Assertion[T] {
	return predicateAssertion[T]{name: name, fn: fn}
}

type predicateAssertion[T any] struct {
	name string
	fn   func(T) (bool, string)
}

func (p predicateAssertion[T]) Name() string { return p.name }
func (p predicateAssertion[T]) Check(_ context.Context, output T) AssertionResult {
	pass, reason := p.fn(output)
	return AssertionResult{Pass: pass, Reason: reason}
}
func (predicateAssertion[T]) MinPassRate() float64 { return 1.0 }

// rateOverride wraps another Assertion to lower its MinPassRate. Used by
// AtLeast to make strict assertions tolerant.
type rateOverride[T any] struct {
	Assertion[T]
	rate float64
}

func (r rateOverride[T]) MinPassRate() float64 { return r.rate }
