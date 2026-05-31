package llmeval

import (
	"encoding/json"
	"fmt"
	"io"
)

// PrintText writes a human-readable summary of result to w. The shape is
// stable and suitable for log files, terminal output, and inclusion in
// test failure messages. Returns the first write error encountered (if
// any) from w.
func PrintText(w io.Writer, result EvalResult) error {
	e := &errWriter{w: w}
	e.printf("Eval: %s\nPass: %s (%d runs)\n",
		nameOrPlaceholder(result.Name),
		passLabel(result.Pass),
		len(result.Runs),
	)
	for i, run := range result.Runs {
		writeRun(e, i, run)
	}
	if len(result.Assertions) > 0 {
		e.printf("\nAssertion rates:\n")
		for _, a := range result.Assertions {
			e.printf("  %s  %s  %d/%d  (≥%.2f)\n",
				passLabel(a.Pass), a.Name, a.Passed, a.Total, a.MinRate,
			)
		}
	}
	if len(result.Criteria) > 0 {
		e.printf("\nCriterion rates:\n")
		for _, c := range result.Criteria {
			e.printf("  %s  %s  %d/%d  (≥%.2f)\n",
				passLabel(c.Pass), c.Description, c.Passed, c.Total, c.MinRate,
			)
		}
	}
	return e.err
}

// PrintJSON writes result as indented JSON to w. The shape matches
// EvalResult / RunResult / etc.'s struct tags — see those types for the
// authoritative field list. Err is rendered as a string; Duration as
// milliseconds (durationMs).
func PrintJSON(w io.Writer, result EvalResult) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

// errWriter centralises "stop on first write error" so callers can write
// many lines without checking err after every Fprintf.
type errWriter struct {
	w   io.Writer
	err error
}

func (e *errWriter) printf(format string, args ...any) {
	if e.err != nil {
		return
	}
	_, e.err = fmt.Fprintf(e.w, format, args...)
}

func writeRun(e *errWriter, idx int, run RunResult) {
	e.printf("\nRun %d  [%dms]  %s\n",
		idx, run.Duration.Milliseconds(), passLabel(run.Pass),
	)
	if run.Err != nil {
		e.printf("  Error: %s\n", run.Err)
		return
	}
	e.printf("  Output: %q\n", run.Output)
	if len(run.Assertions) > 0 {
		e.printf("  Assertions:\n")
		for _, a := range run.Assertions {
			e.printf("    %s%s\n", passLabel(a.Pass), reasonSuffix(a.Reason))
		}
	}
	if len(run.Criteria) > 0 {
		e.printf("  Criteria:\n")
		for _, c := range run.Criteria {
			e.printf("    %s%s\n", passLabel(c.Pass), reasonSuffix(c.Reason))
		}
	}
}

func passLabel(pass bool) string {
	if pass {
		return "PASS"
	}
	return "FAIL"
}

func nameOrPlaceholder(name string) string {
	if name == "" {
		return "(unnamed)"
	}
	return name
}

func reasonSuffix(reason string) string {
	if reason == "" {
		return ""
	}
	return "  — " + reason
}
