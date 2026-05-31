package llmeval_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
)

// sampleResult is a multi-run EvalResult covering both assertion and
// criterion paths, one passing run, one failing run, and one errored run.
// Used as the input for several report tests so the output shape is
// specified once.
func sampleResult() llmeval.EvalResult {
	return llmeval.EvalResult{
		Name: "sentiment classifier",
		Pass: false,
		Runs: []llmeval.RunResult{
			{
				Output:   "positive",
				Pass:     true,
				Duration: 1210 * time.Millisecond,
				Assertions: []llmeval.AssertionResult{
					{Pass: true},
					{Pass: true},
				},
				Criteria: []llmeval.CriterionResult{
					{Pass: true, Reason: "concise"},
				},
			},
			{
				Output:   "POSITIVE",
				Pass:     false,
				Duration: 940 * time.Millisecond,
				Assertions: []llmeval.AssertionResult{
					{Pass: false, Reason: `"POSITIVE" not in set`},
					{Pass: false, Reason: `got "POSITIVE"`},
				},
				Criteria: []llmeval.CriterionResult{
					{Pass: false, Reason: "shouty"},
				},
			},
			{
				Err:      errors.New("rate limited"),
				Duration: 50 * time.Millisecond,
			},
		},
		Assertions: []llmeval.AssertionRate{
			{Name: "one of: positive, negative, neutral", Passed: 2, Total: 2, MinRate: 1.0, Pass: true},
			{Name: `equal: "positive"`, Passed: 1, Total: 2, MinRate: 0.8, Pass: false},
		},
		Criteria: []llmeval.CriterionRate{
			{Description: "is concise", Passed: 1, Total: 2, MinRate: 1.0, Pass: false},
		},
	}
}

func TestPrintTextWritesHeaderWithNameAndPassStatus(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, sampleResult()))

	s := buf.String()
	assert.Contains(t, s, "sentiment classifier")
	assert.Contains(t, s, "FAIL")
}

func TestPrintTextWritesPerRunDetail(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, sampleResult()))

	s := buf.String()
	assert.Contains(t, s, `Output: "positive"`)
	assert.Contains(t, s, `Output: "POSITIVE"`)
	assert.Contains(t, s, "Run 0")
	assert.Contains(t, s, "Run 1")
}

func TestPrintTextSurfacesPerAssertionFailureReasons(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, sampleResult()))

	assert.Contains(t, buf.String(), `"POSITIVE" not in set`)
	assert.Contains(t, buf.String(), `got "POSITIVE"`)
}

func TestPrintTextSurfacesRunErrors(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, sampleResult()))

	assert.Contains(t, buf.String(), "rate limited")
}

func TestPrintTextWritesAssertionRatesSummary(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, sampleResult()))

	s := buf.String()
	assert.Contains(t, s, "Assertion rates")
	assert.Contains(t, s, "1/2") // failing assertion's passed/total
}

func TestPrintTextWritesCriterionRatesSummaryWhenPresent(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, sampleResult()))

	s := buf.String()
	assert.Contains(t, s, "Criterion rates")
	assert.Contains(t, s, "is concise")
}

func TestPrintTextOmitsCriterionRatesWhenAbsent(t *testing.T) {
	r := sampleResult()
	r.Criteria = nil
	for i := range r.Runs {
		r.Runs[i].Criteria = nil
	}

	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, r))

	assert.NotContains(t, buf.String(), "Criterion rates")
}

func TestPrintTextRendersAPassingResultCleanly(t *testing.T) {
	r := llmeval.EvalResult{
		Name: "happy path",
		Pass: true,
		Runs: []llmeval.RunResult{{Output: "hello", Pass: true, Duration: 100 * time.Millisecond,
			Assertions: []llmeval.AssertionResult{{Pass: true}}}},
		Assertions: []llmeval.AssertionRate{{Name: "equal: \"hello\"", Passed: 1, Total: 1, MinRate: 1.0, Pass: true}},
	}
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, r))

	s := buf.String()
	assert.Contains(t, s, "PASS")
	assert.NotContains(t, s, "FAIL")
}

// ─────────────────────────────────────────────────────────────────────────────
// PrintJSON
// ─────────────────────────────────────────────────────────────────────────────

func TestPrintJSONProducesValidJSON(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSON(&buf, sampleResult()))

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
}

func TestPrintJSONIncludesPassFlagAndName(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSON(&buf, sampleResult()))

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
	assert.Equal(t, false, decoded["pass"])
	assert.Equal(t, "sentiment classifier", decoded["name"])
}

func TestPrintJSONRendersErrAsString(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSON(&buf, sampleResult()))

	var decoded struct {
		Runs []struct {
			Err string `json:"err"`
		} `json:"runs"`
	}
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
	require.Len(t, decoded.Runs, 3)
	assert.Equal(t, "rate limited", decoded.Runs[2].Err)
}

func TestPrintJSONRendersDurationInMilliseconds(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSON(&buf, sampleResult()))

	var decoded struct {
		Runs []struct {
			DurationMS int64 `json:"durationMs"`
		} `json:"runs"`
	}
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
	require.Len(t, decoded.Runs, 3)
	assert.Equal(t, int64(1210), decoded.Runs[0].DurationMS)
	assert.Equal(t, int64(940), decoded.Runs[1].DurationMS)
	assert.Equal(t, int64(50), decoded.Runs[2].DurationMS)
}

func TestPrintJSONUsesIndentation(t *testing.T) {
	// Indented output makes log files / dashboards easier to skim. Verify
	// the renderer is using MarshalIndent (or equivalent) by checking for
	// newlines in the output.
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSON(&buf, sampleResult()))

	assert.Contains(t, buf.String(), "\n")
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

// failingWriter returns an error on every Write call — used to exercise
// PrintText's write-error path.
type failingWriter struct{}

func (failingWriter) Write(_ []byte) (int, error) { return 0, errors.New("boom") }

func TestPrintTextSurfacesWriteErrors(t *testing.T) {
	err := llmeval.PrintText(failingWriter{}, sampleResult())
	require.Error(t, err)
}

func TestPrintTextRendersUnnamedEvalsAsPlaceholderAndOmitsRateSections(t *testing.T) {
	// An EvalResult with nothing but a Pass flag exercises:
	//   - the empty-Name branch in PrintText's header,
	//   - the empty-Runs branch (the per-run loop has zero iterations),
	//   - both "omit the rates section" branches at the bottom.
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, llmeval.EvalResult{Pass: true}))

	s := buf.String()
	assert.Contains(t, s, "(unnamed)")
	assert.NotContains(t, s, "Assertion rates")
	assert.NotContains(t, s, "Criterion rates")
}

func TestPrintTextOmitsTheAssertionsSectionWhenARunHasNone(t *testing.T) {
	// A non-erroring run with zero assertions and zero criteria — exercises
	// the "skip both inner sections" branches of writeRun.
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, llmeval.EvalResult{
		Name: "assertion-less",
		Pass: true,
		Runs: []llmeval.RunResult{{Output: "hello", Pass: true, Duration: 5 * time.Millisecond}},
	}))

	s := buf.String()
	assert.Contains(t, s, "Output: ")
	assert.NotContains(t, s, "Assertions:")
	assert.NotContains(t, s, "Criteria:")
}

func TestPrintJSONSurfacesWriteErrors(t *testing.T) {
	err := llmeval.PrintJSON(failingWriter{}, sampleResult())
	require.Error(t, err)
}

// ─────────────────────────────────────────────────────────────────────────────
// Usage section
// ─────────────────────────────────────────────────────────────────────────────

func TestPrintTextWritesUsageSectionWhenPresent(t *testing.T) {
	r := sampleResult()
	r.Usage = []llmeval.Usage{
		{Provider: "openai", Model: "gpt-4.1-mini", InputTokens: 300, OutputTokens: 150},
		{Provider: "anthropic", Model: "claude-haiku-4-5", InputTokens: 200, OutputTokens: 80},
	}

	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, r))

	s := buf.String()
	assert.Contains(t, s, "Usage:")
	assert.Contains(t, s, "openai / gpt-4.1-mini")
	assert.Contains(t, s, "300 in / 150 out")
	assert.Contains(t, s, "anthropic / claude-haiku-4-5")
}

func TestPrintTextOmitsUsageSectionWhenAbsent(t *testing.T) {
	r := sampleResult()
	r.Usage = nil

	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintText(&buf, r))

	assert.NotContains(t, buf.String(), "Usage:")
}

func TestPrintJSONIncludesUsageField(t *testing.T) {
	r := sampleResult()
	r.Usage = []llmeval.Usage{{Provider: "openai", Model: "x", InputTokens: 1, OutputTokens: 2}}

	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSON(&buf, r))

	var decoded struct {
		Usage []struct {
			Provider     string `json:"provider"`
			Model        string `json:"model"`
			InputTokens  int    `json:"inputTokens"`
			OutputTokens int    `json:"outputTokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
	require.Len(t, decoded.Usage, 1)
	assert.Equal(t, "openai", decoded.Usage[0].Provider)
	assert.Equal(t, 1, decoded.Usage[0].InputTokens)
}
