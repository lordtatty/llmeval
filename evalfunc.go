package llmeval

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// EvalFunc is the imperative alternative to Eval[T]. The adopter's Run
// closure does the work and returns its own AssertionResult list — no
// typed output flowing through, no built-in Assertion interface, no
// generics. The framework's job is just to iterate, aggregate by name,
// and run PostChecks.
//
// Pick EvalFunc when your eval has a few assertions that are easy to
// express inline against your SUT's natural return type. Pick Eval[T]
// when you want the declarative spec (one assertion list, framework
// iterates) or when you're reusing assertion helpers across evals.
//
// Both live in this package side-by-side so adopters can compare.
type EvalFunc struct {
	// Name identifies the eval in reports. Optional.
	Name string

	// Run executes the SUT and returns one AssertionResult per check the
	// user performed. The framework calls Run Repeat times and aggregates
	// by Name. An error short-circuits the repeat (no results recorded);
	// the run is logged with Err and skipped during aggregation.
	Run func(ctx context.Context) ([]AssertionResult, error)

	// Repeat is how many times to invoke Run. Defaults to 1.
	Repeat int

	// Concurrency caps how many Run invocations may be in flight at once.
	// Defaults to 1 (sequential). Set > 1 for parallel runs when each Run
	// is I/O-bound. Run must be safe to invoke concurrently when > 1.
	Concurrency int

	// Timeout caps each individual Run via context.WithTimeout. Zero
	// means no per-Run cap.
	Timeout time.Duration

	// MinPassRates lets you tolerate per-assertion failures across
	// repeats. Map keys are assertion Names; values are required pass
	// rates in [0, 1]. Assertions not in the map require 1.0 (strict).
	// Equivalent to wrapping the declarative Eval[T]'s assertion with
	// AtLeast(rate, ...).
	MinPassRates map[string]float64

	// PostChecks fire once after all runs complete. Same shape as Eval[T]
	// — operates on EvalFuncResult.Summary().
	PostChecks []PostCheck
}

// EvalFuncResult is the aggregate outcome of one EvalFunc execution.
type EvalFuncResult struct {
	Name       string                `json:"name,omitempty"`
	Runs       []EvalFuncRunResult   `json:"runs,omitempty"`
	Assertions []AssertionRate       `json:"assertions,omitempty"`
	Pass       bool                  `json:"pass"`
	Usage      []Usage               `json:"usage,omitempty"`
	PostChecks []PostCheckResult     `json:"postChecks,omitempty"`
}

// Summary returns the non-generic view of this result. PostChecks read
// this — same shape as EvalResult[T].Summary so MaxCost / MaxTokens / a
// custom PostCheck don't need to know which Eval shape produced it.
func (r EvalFuncResult) Summary() EvalSummary {
	return EvalSummary{
		Name:       r.Name,
		Pass:       r.Pass,
		Assertions: r.Assertions,
		Usage:      r.Usage,
		PostChecks: r.PostChecks,
	}
}

// EvalFuncRunResult is the outcome of one EvalFunc.Run invocation.
type EvalFuncRunResult struct {
	Pass bool `json:"pass"`
	// Err is the SUT error or recovered panic, when present. Rendered as
	// a string via MarshalJSON since error has no useful default
	// representation.
	Err error `json:"-"`
	// Duration is wall-clock time for this Run. Rendered as milliseconds
	// in JSON via MarshalJSON; Go's default (nanoseconds, untyped int) is
	// too coarse for dashboards.
	Duration   time.Duration     `json:"-"`
	Assertions []AssertionResult `json:"assertions,omitempty"`
}

// MarshalJSON renders EvalFuncRunResult with err as a string (omitted when
// nil) and duration in milliseconds, matching RunResult[T]'s shape.
func (r EvalFuncRunResult) MarshalJSON() ([]byte, error) {
	var errStr string
	if r.Err != nil {
		errStr = r.Err.Error()
	}
	return json.Marshal(struct {
		Assertions []AssertionResult `json:"assertions,omitempty"`
		Pass       bool              `json:"pass"`
		Err        string            `json:"err,omitempty"`
		DurationMS int64             `json:"durationMs"`
	}{
		Assertions: r.Assertions,
		Pass:       r.Pass,
		Err:        errStr,
		DurationMS: r.Duration.Milliseconds(),
	})
}

// RunFunc executes an EvalFunc and returns the aggregated result. Mirrors
// llmeval.Run[T]'s lifecycle (collector attach, repeat with concurrency,
// per-Run timeout, panic recovery, aggregate, PostChecks) but the
// per-Run output is a typed AssertionResult slice rather than a T.
func RunFunc(ctx context.Context, eval EvalFunc) EvalFuncResult {
	repeat := max(eval.Repeat, 1)
	concurrency := max(eval.Concurrency, 1)
	result := EvalFuncResult{Name: eval.Name}

	ctx, collector := NewUsageCtx(ctx)

	runs, ran := runAllFunc(ctx, eval, repeat, concurrency)

	agg := newNameAggregator(eval.MinPassRates)
	for i, rr := range runs {
		if !ran[i] {
			continue
		}
		result.Runs = append(result.Runs, rr)
		if rr.Err != nil {
			continue
		}
		for _, ar := range rr.Assertions {
			agg.add(ar.Name, ar.Pass)
		}
	}

	result.Assertions = agg.rates()
	result.Pass = anyRunSucceededFunc(result.Runs)
	for _, ar := range result.Assertions {
		if !ar.Pass {
			result.Pass = false
		}
	}
	result.Usage = collector.Aggregated()
	applyPostChecksFunc(&result, eval.PostChecks)
	return result
}

// anyRunSucceededFunc reports whether at least one run completed without
// error. An eval with zero successful runs has no positive evidence the
// SUT works and must fail regardless of how few assertions were declared.
func anyRunSucceededFunc(runs []EvalFuncRunResult) bool {
	for _, rr := range runs {
		if rr.Err == nil {
			return true
		}
	}
	return false
}

// applyPostChecksFunc runs each PostCheck against the result Summary.
// Same shape as Eval[T]'s applyPostChecks — kept separate so it compiles
// against EvalFuncResult specifically.
func applyPostChecksFunc(result *EvalFuncResult, checks []PostCheck) {
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

// runAllFunc spawns the per-repeat goroutines, same shape as runAll for
// Eval[T] but invoking the typed-result Run signature.
func runAllFunc(ctx context.Context, eval EvalFunc, repeat, concurrency int) ([]EvalFuncRunResult, []bool) {
	runs := make([]EvalFuncRunResult, repeat)
	ran := make([]bool, repeat)
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i := range repeat {
		if ctx.Err() != nil {
			break
		}
		sem <- struct{}{}
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()
			if ctx.Err() != nil {
				return
			}
			runs[idx] = runOnceFunc(ctx, eval)
			ran[idx] = true
		}(i)
	}
	wg.Wait()
	return runs, ran
}

// runOnceFunc executes a single EvalFunc.Run with the per-Run timeout and
// panic recovery. The recovered panic surfaces as RunResult.Err so a
// misbehaving SUT can't take down the test process.
func runOnceFunc(ctx context.Context, eval EvalFunc) (rr EvalFuncRunResult) {
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

	results, err := eval.Run(runCtx)
	rr = EvalFuncRunResult{
		Assertions: results,
		Err:        err,
		Duration:   time.Since(start),
	}
	if err != nil {
		return rr
	}

	rr.Pass = true
	for _, ar := range results {
		if !ar.Pass {
			rr.Pass = false
			break
		}
	}
	return rr
}

// nameAggregator groups AssertionResults across Runs by Name, computing
// per-name pass rates. Handles missing names (a Run that doesn't return
// assertion "x" simply doesn't contribute to x's Total) and first-seen
// ordering so the result slice is stable across re-runs of the same eval.
//
// Within a single Run, every occurrence of a given Name counts as an
// independent observation — returning the same Name twice with mixed
// Pass values produces total=2 with one pass. The run-level Pass is still
// false (any failing AssertionResult fails the run), but the per-name
// rate reflects every observation.
type nameAggregator struct {
	stats   map[string]*nameStats
	order   []string
	minRate map[string]float64
}

type nameStats struct {
	passed, total int
}

func newNameAggregator(minRates map[string]float64) *nameAggregator {
	return &nameAggregator{
		stats:   map[string]*nameStats{},
		minRate: minRates,
	}
}

func (a *nameAggregator) add(name string, pass bool) {
	s, ok := a.stats[name]
	if !ok {
		s = &nameStats{}
		a.stats[name] = s
		a.order = append(a.order, name)
	}
	s.total++
	if pass {
		s.passed++
	}
}

func (a *nameAggregator) rates() []AssertionRate {
	out := make([]AssertionRate, 0, len(a.order))
	for _, name := range a.order {
		s := a.stats[name]
		minRate := 1.0
		if r, ok := a.minRate[name]; ok {
			minRate = r
		}
		out = append(out, AssertionRate{
			Name:    name,
			Passed:  s.passed,
			Total:   s.total,
			MinRate: minRate,
			Pass:    s.total > 0 && float64(s.passed)/float64(s.total) >= minRate,
		})
	}
	return out
}

// JudgeAll calls judge.Evaluate(ctx, output, criteria) and converts the
// per-criterion verdicts into AssertionResults named after each Criterion's
// Description. Use inside EvalFunc.Run to wire an LLM judge without
// rebuilding the verdict-to-result plumbing every time.
//
//	results = append(results, llmeval.JudgeAll(ctx, judge, output, criteria)...)
//
// On judge error or verdict-count mismatch, every criterion is marked
// failed with the same reason — same fail-uniformly semantic Eval[T] uses
// inside applyJudge.
func JudgeAll(ctx context.Context, judge Judge, output string, criteria []Criterion) []AssertionResult {
	out := make([]AssertionResult, len(criteria))
	verdicts, err := judge.Evaluate(ctx, output, criteria)
	if err != nil {
		reason := fmt.Sprintf("judge error: %v", err)
		for i, c := range criteria {
			out[i] = AssertionResult{Name: c.Description, Pass: false, Reason: reason}
		}
		return out
	}
	if len(verdicts) != len(criteria) {
		reason := fmt.Sprintf("judge returned %d verdicts for %d criteria", len(verdicts), len(criteria))
		for i, c := range criteria {
			out[i] = AssertionResult{Name: c.Description, Pass: false, Reason: reason}
		}
		return out
	}
	for i, c := range criteria {
		out[i] = AssertionResult{
			Name:   c.Description,
			Pass:   verdicts[i].Pass,
			Reason: verdicts[i].Reason,
		}
	}
	return out
}
