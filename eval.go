// Package llmeval is a small Go framework for evaluating LLM outputs.
// See README.md in the repository for usage.
//
// For `go test` integration (Run-and-mark-t-failed), import the
// llmeval/llmevaltest subpackage instead — it isolates the "testing"
// import from consumers' production builds.
package llmeval

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Eval is one evaluation: a system-under-test call (Run), local assertions to
// apply to its output, and how many times to repeat. The runner invokes Run
// `Repeat` times, applies every Assert to each output, and aggregates the
// pass rate per assertion.
//
// T is the type your SUT returns. For most LLM applications T is string;
// for structured outputs (JSON-mode classifiers, extractors, multi-field
// responses) T is your struct.
type Eval[T any] struct {
	// Name identifies the eval in reports. Optional; RunTest defaults to
	// t.Name() when this is empty.
	Name string

	// Run is the SUT closure — invoke your LLM (or LLM-driven function) and
	// return its output as T plus any error. Called Repeat times.
	Run func(ctx context.Context) (T, error)

	// Repeat is how many times to invoke Run. Defaults to 1. Use 5+ to surface
	// LLM non-determinism.
	Repeat int

	// Assertions holds the predicates evaluated against each Run output. They
	// are pure (no LLM calls). Wrap any assertion with AtLeast to allow some
	// failures across Repeat runs; otherwise it must hold every time.
	Assertions []Assertion[T]

	// Judge, if non-nil, is called once per Run after assertions to evaluate
	// the SUT output against Criteria. The judge sees the output as a string —
	// the framework serializes T via Serializer (or its default — string
	// passes through, anything else goes through json.Marshal). Set Judge
	// AND Criteria together, or leave both unset.
	Judge Judge

	// Criteria are the rubric items the Judge evaluates per Run. The judge
	// receives the full list in one call and returns one verdict per criterion.
	Criteria []Criterion

	// Serializer converts T to the string the judge sees and reports display.
	// Optional — when nil the framework uses a default that returns string
	// values as-is and json.Marshal's anything else.
	Serializer func(T) (string, error)

	// Timeout, if non-zero, caps each individual Run via context.WithTimeout.
	// The timeout covers the whole runOnce: SUT call + assertions + judge
	// call — so a slow SUT that eats most of the timeout will starve the
	// judge. The user's Run (and any Judge implementation) must respect
	// ctx for this to fire.
	Timeout time.Duration

	// Concurrency caps how many Run invocations may be in flight at once.
	// Defaults to 1 (sequential). Set > 1 for parallel runs when each Run
	// is I/O-bound (e.g. an LLM call). Run and Judge must be safe to invoke
	// concurrently when set > 1.
	Concurrency int

	// PostChecks fire once after all runs and judges complete, with access
	// to the fully aggregated EvalSummary (everything except per-Run
	// typed outputs). Use for budget assertions (see MaxCost) and other
	// policy checks that operate on the eval as a whole. A failed
	// PostCheck marks EvalResult.Pass false.
	PostChecks []PostCheck
}

// Assertion is a single check against the SUT output. The runner calls Check
// once per Run, accumulates Pass/Total counts across all repeats, and decides
// whether the assertion overall passes by comparing the rate to MinPassRate.
type Assertion[T any] interface {
	// Name is a short label for the assertion, used in reports.
	Name() string

	// Check runs the predicate against one SUT output. ctx is the same
	// context passed to Eval.Run.
	Check(ctx context.Context, output T) AssertionResult

	// MinPassRate is the fraction of Repeat runs in which Check must return
	// Pass=true for this assertion to pass overall. Built-in helpers return
	// 1.0 (strict). Use AtLeast to wrap an assertion with a lower threshold.
	MinPassRate() float64
}

// AssertionResult is the outcome of a single Assertion.Check call.
type AssertionResult struct {
	// Pass is true if the predicate held for this output.
	Pass bool `json:"pass"`

	// Reason explains a failure. Empty when Pass is true.
	Reason string `json:"reason,omitempty"`
}

// EvalResult is the aggregate outcome of one Eval execution (all repeats).
type EvalResult[T any] struct {
	// Name is the eval's name (or t.Name() under RunTest).
	Name string `json:"name,omitempty"`

	// Runs holds one RunResult per repeat, in execution order. Failed Runs
	// (Err != nil) appear here too but don't contribute to assertion rates.
	Runs []RunResult[T] `json:"runs,omitempty"`

	// Assertions aggregates each assertion across all Runs. Nil when no
	// assertions were defined.
	Assertions []AssertionRate `json:"assertions,omitempty"`

	// Criteria aggregates each judged criterion across all Runs. Nil when
	// no Judge+Criteria were defined.
	Criteria []CriterionRate `json:"criteria,omitempty"`

	// Pass is true only if every AssertionRate.Pass AND every
	// CriterionRate.Pass is true.
	Pass bool `json:"pass"`

	// Usage is the aggregated token usage across all judge and SUT LLM
	// calls in this eval, grouped by (Provider, Model). Empty when the
	// eval made no recorded calls — sub-module judges record
	// automatically; SUT code records via RecordUsage(ctx, ...).
	Usage []Usage `json:"usage,omitempty"`

	// PostChecks holds the outcome of each Eval.PostCheck, in the order
	// they were declared. Empty when the eval defined no PostChecks.
	PostChecks []PostCheckResult `json:"postChecks,omitempty"`
}

// Summary returns the non-T-dependent view of this result. PostChecks
// receive this view, which lets PostCheck stay non-generic — Go's
// inference doesn't propagate slice-element types back into generic
// function calls, so a generic PostCheck[T] would force every adopter
// to write `MaxCost[T](...)` explicitly.
func (r EvalResult[T]) Summary() EvalSummary {
	return EvalSummary{
		Name:       r.Name,
		Pass:       r.Pass,
		Assertions: r.Assertions,
		Criteria:   r.Criteria,
		Usage:      r.Usage,
		PostChecks: r.PostChecks,
	}
}

// EvalSummary is the non-generic view of an EvalResult that PostChecks
// receive. Excludes Runs (which is T-typed). Created by EvalResult.Summary.
type EvalSummary struct {
	Name       string            `json:"name,omitempty"`
	Pass       bool              `json:"pass"`
	Assertions []AssertionRate   `json:"assertions,omitempty"`
	Criteria   []CriterionRate   `json:"criteria,omitempty"`
	Usage      []Usage           `json:"usage,omitempty"`
	PostChecks []PostCheckResult `json:"postChecks,omitempty"`
}

// CriterionRate aggregates a single judged criterion across an eval's Repeat runs.
type CriterionRate struct {
	// Description is the Criterion.Description at eval-build time.
	Description string `json:"description"`

	// Passed is the number of Runs in which the judge returned Pass=true
	// for this criterion.
	Passed int `json:"passed"`

	// Total is the number of Runs in which this criterion was evaluated.
	// Runs that errored before assertions/judging ran don't count here;
	// runs where the judge itself errored DO count and contribute to Total
	// (as failures), so judge instability is visible in the rate.
	Total int `json:"total"`

	// MinRate is the Criterion.MinPassRate from the input Criterion (where
	// 0 means strict, i.e. effectively 1.0 — applied at runtime).
	MinRate float64 `json:"minRate"`

	// Pass is true if Passed/Total >= effective MinRate (and Total > 0).
	Pass bool `json:"pass"`
}

// AssertionRate aggregates a single assertion across an eval's Repeat runs.
type AssertionRate struct {
	// Name is the assertion's Name() at the time the eval was built.
	Name string `json:"name"`

	// Passed is the number of Runs in which this assertion returned Pass=true.
	Passed int `json:"passed"`

	// Total is the number of Runs in which this assertion was evaluated.
	// Runs that errored before assertions ran (Err != nil) don't count here.
	Total int `json:"total"`

	// MinRate is the threshold this assertion needed to meet, copied from
	// Assertion.MinPassRate() at runtime.
	MinRate float64 `json:"minRate"`

	// Pass is true if Passed/Total >= MinRate (and Total > 0).
	Pass bool `json:"pass"`
}

// RunResult is the outcome of a single Run (one repeat).
type RunResult[T any] struct {
	// Output is what Eval.Run returned. Zero value of T if Err != nil.
	Output T `json:"output,omitempty"`

	// Assertions holds the per-assertion outcome for this Run, in the same
	// order as Eval.Assertions. Empty when Err != nil (assertions are skipped).
	Assertions []AssertionResult `json:"assertions,omitempty"`

	// Criteria holds the per-criterion verdict for this Run, in the same
	// order as Eval.Criteria. Empty when Eval.Judge is nil OR when Err != nil.
	Criteria []CriterionResult `json:"criteria,omitempty"`

	// Pass is true only when Err is nil AND every assertion AND every
	// criterion verdict for this Run is Pass=true. Note this is per-run;
	// EvalResult.Pass is the per-eval aggregate.
	Pass bool `json:"pass"`

	// Err records a Run-time failure: a non-nil error from Eval.Run, a
	// recovered panic, or a context timeout. JSON-marshalled as a string
	// via MarshalJSON since error has no useful default representation.
	Err error `json:"-"`

	// Duration is how long Eval.Run took. JSON-marshalled as milliseconds
	// for downstream-tool readability.
	Duration time.Duration `json:"-"`
}

// MarshalJSON renders RunResult with err as a string (or omitted when nil)
// and duration in milliseconds — both more usable shapes for JSON than
// Go's defaults (error → empty struct, duration → nanoseconds).
func (r RunResult[T]) MarshalJSON() ([]byte, error) {
	var errStr string
	if r.Err != nil {
		errStr = r.Err.Error()
	}
	return json.Marshal(struct {
		Output     T                 `json:"output,omitempty"`
		Assertions []AssertionResult `json:"assertions,omitempty"`
		Criteria   []CriterionResult `json:"criteria,omitempty"`
		Pass       bool              `json:"pass"`
		Err        string            `json:"err,omitempty"`
		DurationMS int64             `json:"durationMs"`
	}{
		Output:     r.Output,
		Assertions: r.Assertions,
		Criteria:   r.Criteria,
		Pass:       r.Pass,
		Err:        errStr,
		DurationMS: r.Duration.Milliseconds(),
	})
}

// defaultSerialize converts v to a string. For T = string it returns the
// value as-is; otherwise it json.Marshal's the value. Used internally to
// bridge the typed Eval[T] surface to string-based reporters and Judge.
func defaultSerialize[T any](v T) (string, error) {
	if s, ok := any(v).(string); ok {
		return s, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// serializerOf returns eval.Serializer if non-nil, otherwise the default.
func serializerOf[T any](eval Eval[T]) func(T) (string, error) {
	if eval.Serializer != nil {
		return eval.Serializer
	}
	return defaultSerialize[T]
}

// Run executes eval and returns the aggregated result. It invokes eval.Run
// `Repeat` times (or once if Repeat < 1) with up to `Concurrency` invocations
// in flight (or sequentially if Concurrency < 2), applies every assertion to
// each non-erroring output, and computes per-assertion pass rates.
//
// Run attaches a fresh UsageCollector to ctx so EvalResult.Usage reflects
// only this invocation's calls. Any UsageCollector you attached to ctx via
// NewUsageCtx is shadowed for the duration of Run and will not see records
// from this eval — aggregate across multiple Runs by walking EvalResult.Usage
// yourself.
//
// Run does not depend on the testing package. For `go test` integration use
// llmevaltest.Run, which wraps Run and reports failures via *testing.T.
func Run[T any](ctx context.Context, eval Eval[T]) EvalResult[T] {
	repeat := max(eval.Repeat, 1)
	concurrency := max(eval.Concurrency, 1)
	result := EvalResult[T]{Name: eval.Name}
	assTallies := newCheckTallies(len(eval.Assertions))
	critTallies := newCheckTallies(len(eval.Criteria))

	// Fresh collector per eval so EvalResult.Usage reflects only this
	// run's calls. Any collector pre-attached to ctx is shadowed for the
	// duration of Run, not modified.
	ctx, collector := NewUsageCtx(ctx)

	runs, ran := runAll(ctx, eval, repeat, concurrency)

	for i, rr := range runs {
		if !ran[i] {
			continue
		}
		result.Runs = append(result.Runs, rr)
		if rr.Err != nil {
			continue
		}
		for j, ar := range rr.Assertions {
			assTallies.add(j, ar.Pass)
		}
		for j, cr := range rr.Criteria {
			critTallies.add(j, cr.Pass)
		}
	}

	result.Pass = true
	for i, a := range eval.Assertions {
		ar := assTallies.rate(i, a.MinPassRate())
		result.Assertions = append(result.Assertions, AssertionRate{
			Name: a.Name(), Passed: ar.passed, Total: ar.total, MinRate: ar.threshold, Pass: ar.pass,
		})
		if !ar.pass {
			result.Pass = false
		}
	}
	for i, c := range eval.Criteria {
		ar := critTallies.rate(i, effectiveMinRate(c.MinPassRate))
		result.Criteria = append(result.Criteria, CriterionRate{
			Description: c.Description, Passed: ar.passed, Total: ar.total, MinRate: ar.threshold, Pass: ar.pass,
		})
		if !ar.pass {
			result.Pass = false
		}
	}
	result.Usage = collector.Aggregated()
	applyPostChecks(&result, eval.PostChecks)
	return result
}

// applyPostChecks runs each PostCheck against the aggregated result and
// records the outcomes. A failed PostCheck marks result.Pass false.
// Separated from Run because PostChecks are a distinct phase (operating
// on the whole aggregate), not a substep of any single Run.
func applyPostChecks[T any](result *EvalResult[T], checks []PostCheck) {
	for _, pc := range checks {
		pass, reason := pc.Check(result.Summary())
		result.PostChecks = append(result.PostChecks, PostCheckResult{
			Name: pc.Name, Pass: pass, Reason: reason,
		})
		if !pass {
			result.Pass = false
		}
	}
}

// runAll executes the eval's Run closure up to `repeat` times with at most
// `concurrency` invocations in flight. Each goroutine writes into its own
// runs[idx] slot and sets ran[idx] true on completion; slots whose goroutine
// short-circuited (ctx cancelled) stay zero-valued and ran[idx] false, so
// the caller can distinguish "didn't run" from "ran and returned empty".
func runAll[T any](ctx context.Context, eval Eval[T], repeat, concurrency int) ([]RunResult[T], []bool) {
	runs := make([]RunResult[T], repeat)
	ran := make([]bool, repeat)
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i := range repeat {
		if ctx.Err() != nil {
			// Parent ctx cancelled — finish whatever's running, don't start more.
			break
		}
		sem <- struct{}{}
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()
			// Re-check after slot acquisition: ctx may have been cancelled
			// while this goroutine was blocked waiting for a free slot.
			if ctx.Err() != nil {
				return
			}
			runs[idx] = runOnce(ctx, eval)
			ran[idx] = true
		}(i)
	}
	wg.Wait()
	return runs, ran
}

// checkTallies accumulates per-check pass/total counts across an eval's Runs.
// Used identically for assertions and judged criteria.
type checkTallies struct{ passed, total []int }

func newCheckTallies(n int) checkTallies {
	return checkTallies{passed: make([]int, n), total: make([]int, n)}
}

func (c checkTallies) add(idx int, pass bool) {
	c.total[idx]++
	if pass {
		c.passed[idx]++
	}
}

type rateOutcome struct {
	passed, total int
	threshold     float64
	pass          bool
}

func (c checkTallies) rate(idx int, threshold float64) rateOutcome {
	p, t := c.passed[idx], c.total[idx]
	return rateOutcome{
		passed:    p,
		total:     t,
		threshold: threshold,
		pass:      t > 0 && float64(p)/float64(t) >= threshold,
	}
}

// effectiveMinRate maps Criterion.MinPassRate's zero-means-strict convention
// onto an actual numeric threshold (0 → 1.0). Values > 0 pass through.
func effectiveMinRate(min float64) float64 {
	if min <= 0 {
		return 1.0
	}
	return min
}

// runOnce runs eval.Run once, applies the assertions if it succeeded, and
// returns the RunResult. A panic in eval.Run is recovered and surfaced as
// RunResult.Err so a misbehaving SUT can't take down the whole test process.
func runOnce[T any](ctx context.Context, eval Eval[T]) (rr RunResult[T]) {
	runCtx := ctx
	if eval.Timeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, eval.Timeout)
		defer cancel()
	}

	start := time.Now()
	defer func() {
		if r := recover(); r != nil {
			rr.Err = fmt.Errorf("SUT panic: %v", r)
			rr.Duration = time.Since(start)
		}
	}()

	output, err := eval.Run(runCtx)
	rr = RunResult[T]{
		Output:   output,
		Err:      err,
		Duration: time.Since(start),
	}
	if err != nil {
		return rr
	}

	rr.Pass = allPassed(applyAssertions(runCtx, eval.Assertions, output, &rr))

	if eval.Judge != nil && len(eval.Criteria) > 0 {
		rr.Criteria = applyJudge(runCtx, eval.Judge, eval.Criteria, output, serializerOf(eval))
		if !allCriteriaPassed(rr.Criteria) {
			rr.Pass = false
		}
	}
	return rr
}

// applyAssertions runs every assertion against output, appends results to
// rr.Assertions, and returns the per-assertion pass slice for aggregation
// in allPassed.
func applyAssertions[T any](ctx context.Context, asns []Assertion[T], output T, rr *RunResult[T]) []bool {
	results := make([]bool, len(asns))
	for i, a := range asns {
		ar := a.Check(ctx, output)
		rr.Assertions = append(rr.Assertions, ar)
		results[i] = ar.Pass
	}
	return results
}

func allPassed(passes []bool) bool {
	for _, p := range passes {
		if !p {
			return false
		}
	}
	return true
}

func allCriteriaPassed(verdicts []CriterionResult) bool {
	for _, v := range verdicts {
		if !v.Pass {
			return false
		}
	}
	return true
}

// applyJudge serializes the typed output via ser, calls the Judge, and
// normalises its response: a returned error, a serializer error, or a
// verdict count that doesn't match the criteria list are all surfaced as
// a uniform Fail across every criterion with an explanatory Reason.
func applyJudge[T any](ctx context.Context, judge Judge, criteria []Criterion, output T, ser func(T) (string, error)) []CriterionResult {
	outputStr, err := ser(output)
	if err != nil {
		return judgeErrorVerdicts(criteria, fmt.Sprintf("serializer error: %v", err))
	}
	verdicts, err := judge.Evaluate(ctx, outputStr, criteria)
	if err != nil {
		return judgeErrorVerdicts(criteria, fmt.Sprintf("judge error: %v", err))
	}
	if len(verdicts) != len(criteria) {
		return judgeErrorVerdicts(criteria,
			fmt.Sprintf("judge returned %d verdicts for %d criteria", len(verdicts), len(criteria)))
	}
	return verdicts
}

// judgeErrorVerdicts produces a Fail verdict for every criterion with the
// same Reason — used when the judge call itself errored or its output was
// unusable, so callers see the failure consistently across all criteria
// for the affected Run.
func judgeErrorVerdicts(criteria []Criterion, reason string) []CriterionResult {
	out := make([]CriterionResult, len(criteria))
	for i := range criteria {
		out[i] = CriterionResult{Pass: false, Reason: reason}
	}
	return out
}
