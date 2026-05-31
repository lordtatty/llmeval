package llmeval

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"sync"
)

// Equal returns an Assertion that passes when the output exactly equals want.
func Equal(want string) Assertion {
	return predicate(fmt.Sprintf("equal: %q", want), func(o string) (bool, string) {
		if o == want {
			return true, ""
		}
		return false, fmt.Sprintf("got %q", o)
	})
}

// OneOf returns an Assertion that passes when the output is exactly one of
// values. Strict equality — surrounding whitespace, case differences, or
// trailing punctuation fail. Normalise upstream (in your Run closure) if you
// need lenient matching.
func OneOf(values ...string) Assertion {
	joined := strings.Join(values, ", ")
	return predicate("one of: "+joined, func(o string) (bool, string) {
		if slices.Contains(values, o) {
			return true, ""
		}
		return false, fmt.Sprintf("got %q, expected one of: %s", o, joined)
	})
}

// Contains returns an Assertion that passes when the output contains substr.
func Contains(substr string) Assertion {
	return predicate(fmt.Sprintf("contains: %q", substr), func(o string) (bool, string) {
		if strings.Contains(o, substr) {
			return true, ""
		}
		return false, fmt.Sprintf("missing %q", substr)
	})
}

// NotContains returns an Assertion that passes when the output does not
// contain substr.
func NotContains(substr string) Assertion {
	return predicate(fmt.Sprintf("not contains: %q", substr), func(o string) (bool, string) {
		if !strings.Contains(o, substr) {
			return true, ""
		}
		return false, fmt.Sprintf("contained %q", substr)
	})
}

// Matches returns an Assertion that passes when re matches the output.
func Matches(re *regexp.Regexp) Assertion {
	return predicate(fmt.Sprintf("matches: %s", re), func(o string) (bool, string) {
		if re.MatchString(o) {
			return true, ""
		}
		return false, "no match"
	})
}

// Check adapts a user-supplied predicate into an Assertion. Use when the
// built-ins don't fit — including when you want to call out to testify's
// ObjectsAreEqual, go-cmp, or any other library that returns a bool.
func Check(name string, fn func(output string) (pass bool, reason string)) Assertion {
	return predicate(name, fn)
}

// CheckJSON decodes the SUT output as JSON into a value of type T and runs
// fn against the decoded value. Use for a single typed assertion against a
// structured output; when you have several assertions against the same
// struct, prefer SharedDecoder so the decode happens once per output
// rather than once per assertion.
//
// Fails with reason "not valid JSON: ..." when the output isn't decodable
// into T, and "output was JSON null" when the output is the literal token
// `null` (which decodes silently to a zero-value T — a real failure mode
// for JSON-mode models that's worse than malformed JSON because the
// predicate would otherwise see uninitialised fields and report
// per-field misses instead of the actual problem). Otherwise delegates
// pass/fail to fn.
func CheckJSON[T any](name string, fn func(T) (pass bool, reason string)) Assertion {
	return (&SharedDecoder[T]{}).Check(name, fn)
}

// SharedDecoder lets several Assertions over the same Run output share a
// single JSON decode. Use when you have multiple typed checks against
// different fields of one structured response: sd.Check returns
// Assertions that all hit a per-output cache, so each distinct output is
// decoded once instead of once per assertion.
//
//	sd := &llmeval.SharedDecoder[MyStruct]{}
//	Assertions: []llmeval.Assertion{
//	    sd.Check("category", func(r MyStruct) (bool, string) { ... }),
//	    sd.Check("confidence", func(r MyStruct) (bool, string) { ... }),
//	}
//
// Construct with `&` so the embedded mutex stays at a stable address —
// a value copy (e.g. `sd2 := sd`) would silently fork the cache and
// trip go vet's lock-copy detector.
//
// Safe for concurrent use — under Eval.Concurrency > 1 multiple
// goroutines may hit the same decoder simultaneously; the internal mutex
// serialises cache mutations.
//
// Decode failure modes match CheckJSON: "not valid JSON: <err>" when
// Unmarshal fails, "output was JSON null" when the output is the literal
// token `null`. In both cases the predicate is skipped so it never sees
// an uninitialised T.
type SharedDecoder[T any] struct {
	mu    sync.Mutex
	cache map[string]decodedJSON[T]
}

// decodedJSON holds one resolved decode result. reason is empty when the
// decode succeeded; non-empty values propagate to the assertion's failure
// Reason and skip the predicate.
type decodedJSON[T any] struct {
	value  T
	reason string
}

// Check returns an Assertion that retrieves the decoded value from the
// shared cache (decoding on first miss) and runs fn against it.
func (sd *SharedDecoder[T]) Check(name string, fn func(T) (pass bool, reason string)) Assertion {
	return predicate(name, func(output string) (bool, string) {
		d := sd.get(output)
		if d.reason != "" {
			return false, d.reason
		}
		return fn(d.value)
	})
}

func (sd *SharedDecoder[T]) get(output string) decodedJSON[T] {
	sd.mu.Lock()
	defer sd.mu.Unlock()
	if sd.cache == nil {
		sd.cache = make(map[string]decodedJSON[T])
	}
	if d, ok := sd.cache[output]; ok {
		return d
	}
	d := decodeJSON[T](output)
	sd.cache[output] = d
	return d
}

func decodeJSON[T any](output string) decodedJSON[T] {
	var v T
	if strings.TrimSpace(output) == "null" {
		return decodedJSON[T]{reason: "output was JSON null"}
	}
	if err := json.Unmarshal([]byte(output), &v); err != nil {
		return decodedJSON[T]{reason: fmt.Sprintf("not valid JSON: %v", err)}
	}
	return decodedJSON[T]{value: v}
}

// AtLeast wraps a to lower its required pass rate. AtLeast(0.8, asn) means
// "asn must pass on at least 80% of the eval's repeats." Without AtLeast,
// every assertion is strict (must pass every repeat).
//
// rate is clamped to [0, 1].
func AtLeast(rate float64, a Assertion) Assertion {
	if rate < 0 {
		rate = 0
	}
	if rate > 1 {
		rate = 1
	}
	return rateOverride{Assertion: a, rate: rate}
}

// predicate is the internal Assertion implementation backing the helpers
// above (and Check). All helpers are strict (MinPassRate 1.0); use AtLeast
// to tolerate failures.
func predicate(name string, fn func(output string) (pass bool, reason string)) Assertion {
	return predAssertion{name: name, fn: fn}
}

type predAssertion struct {
	name string
	fn   func(output string) (pass bool, reason string)
}

func (p predAssertion) Name() string         { return p.name }
func (p predAssertion) MinPassRate() float64 { return 1.0 }
func (p predAssertion) Check(_ context.Context, output string) AssertionResult {
	pass, reason := p.fn(output)
	return AssertionResult{Pass: pass, Reason: reason}
}

// rateOverride wraps any Assertion and swaps its MinPassRate. Used by AtLeast.
type rateOverride struct {
	Assertion
	rate float64
}

func (r rateOverride) MinPassRate() float64 { return r.rate }
