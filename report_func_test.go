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

// sampleFuncResult is a multi-run EvalFuncResult covering passing,
// failing, and errored runs plus assertion rates, usage, and a
// PostCheck. Mirrors sampleResult() in report_test.go so the two
// reporters can be eyeballed against the same shape.
func sampleFuncResult() llmeval.EvalFuncResult {
	return llmeval.EvalFuncResult{
		Name: "sentiment classifier",
		Pass: false,
		Runs: []llmeval.EvalFuncRunResult{
			{
				Pass:     true,
				Duration: 1210 * time.Millisecond,
				Assertions: []llmeval.AssertionResult{
					{Name: "category", Pass: true},
					{Name: "confidence", Pass: true},
				},
			},
			{
				Pass:     false,
				Duration: 940 * time.Millisecond,
				Assertions: []llmeval.AssertionResult{
					{Name: "category", Pass: false, Reason: `"POSITIVE" not in set`},
					{Name: "confidence", Pass: false, Reason: "below threshold"},
				},
			},
			{
				Err:      errors.New("rate limited"),
				Duration: 50 * time.Millisecond,
			},
		},
		Assertions: []llmeval.AssertionRate{
			{Name: "category", Passed: 1, Total: 2, MinRate: 1.0, Pass: false},
			{Name: "confidence", Passed: 1, Total: 2, MinRate: 0.8, Pass: false},
		},
		Usage: []llmeval.Usage{
			{Provider: "openai", Model: "gpt-4.1-mini", InputTokens: 300, OutputTokens: 150},
		},
		PostChecks: []llmeval.PostCheckResult{
			{Name: "max cost: $0.10", Pass: true},
		},
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// PrintTextFunc
// ─────────────────────────────────────────────────────────────────────────────

func TestPrintTextFuncWritesHeaderWithNameAndPassStatus(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, sampleFuncResult()))

	s := buf.String()
	assert.Contains(t, s, "sentiment classifier")
	assert.Contains(t, s, "FAIL")
}

func TestPrintTextFuncWritesPerRunDetail(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, sampleFuncResult()))

	s := buf.String()
	assert.Contains(t, s, "Run 0")
	assert.Contains(t, s, "Run 1")
}

func TestPrintTextFuncSurfacesPerAssertionNameAndFailureReason(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, sampleFuncResult()))

	s := buf.String()
	assert.Contains(t, s, "category")
	assert.Contains(t, s, `"POSITIVE" not in set`)
	assert.Contains(t, s, "below threshold")
}

func TestPrintTextFuncSurfacesRunErrors(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, sampleFuncResult()))

	assert.Contains(t, buf.String(), "rate limited")
}

func TestPrintTextFuncWritesAssertionRatesSummary(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, sampleFuncResult()))

	s := buf.String()
	assert.Contains(t, s, "Assertion rates")
	assert.Contains(t, s, "1/2")
}

func TestPrintTextFuncWritesUsageSection(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, sampleFuncResult()))

	s := buf.String()
	assert.Contains(t, s, "Usage:")
	assert.Contains(t, s, "openai / gpt-4.1-mini")
	assert.Contains(t, s, "300 in / 150 out")
}

func TestPrintTextFuncOmitsUsageSectionWhenAbsent(t *testing.T) {
	r := sampleFuncResult()
	r.Usage = nil

	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, r))

	assert.NotContains(t, buf.String(), "Usage:")
}

func TestPrintTextFuncWritesPostChecksSection(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, sampleFuncResult()))

	s := buf.String()
	assert.Contains(t, s, "Post-checks:")
	assert.Contains(t, s, "max cost: $0.10")
}

func TestPrintTextFuncOmitsPostChecksSectionWhenAbsent(t *testing.T) {
	r := sampleFuncResult()
	r.PostChecks = nil

	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, r))

	assert.NotContains(t, buf.String(), "Post-checks:")
}

func TestPrintTextFuncRendersUnnamedEvalsAsPlaceholderAndOmitsRateSections(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, llmeval.EvalFuncResult{Pass: true}))

	s := buf.String()
	assert.Contains(t, s, "(unnamed)")
	assert.NotContains(t, s, "Assertion rates")
}

func TestPrintTextFuncOmitsAssertionsSectionWhenARunHasNone(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintTextFunc(&buf, llmeval.EvalFuncResult{
		Name: "no-assertions",
		Pass: true,
		Runs: []llmeval.EvalFuncRunResult{{Pass: true, Duration: 5 * time.Millisecond}},
	}))

	s := buf.String()
	assert.Contains(t, s, "Run 0")
	assert.NotContains(t, s, "Assertions:")
}

func TestPrintTextFuncSurfacesWriteErrors(t *testing.T) {
	err := llmeval.PrintTextFunc(failingWriter{}, sampleFuncResult())
	require.Error(t, err)
}

// ─────────────────────────────────────────────────────────────────────────────
// PrintJSONFunc
// ─────────────────────────────────────────────────────────────────────────────

func TestPrintJSONFuncProducesValidJSON(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSONFunc(&buf, sampleFuncResult()))

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
}

func TestPrintJSONFuncIncludesPassFlagAndName(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSONFunc(&buf, sampleFuncResult()))

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
	assert.Equal(t, false, decoded["pass"])
	assert.Equal(t, "sentiment classifier", decoded["name"])
}

func TestPrintJSONFuncIncludesAssertionRatesAndUsageAndPostChecks(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSONFunc(&buf, sampleFuncResult()))

	var decoded struct {
		Assertions []llmeval.AssertionRate   `json:"assertions"`
		Usage      []llmeval.Usage           `json:"usage"`
		PostChecks []llmeval.PostCheckResult `json:"postChecks"`
	}
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
	require.Len(t, decoded.Assertions, 2)
	assert.Equal(t, "category", decoded.Assertions[0].Name)
	require.Len(t, decoded.Usage, 1)
	assert.Equal(t, "openai", decoded.Usage[0].Provider)
	require.Len(t, decoded.PostChecks, 1)
	assert.Equal(t, "max cost: $0.10", decoded.PostChecks[0].Name)
}

func TestPrintJSONFuncSurfacesWriteErrors(t *testing.T) {
	err := llmeval.PrintJSONFunc(failingWriter{}, sampleFuncResult())
	require.Error(t, err)
}

func TestPrintJSONFuncRendersErrAsString(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSONFunc(&buf, sampleFuncResult()))

	var decoded struct {
		Runs []struct {
			Err string `json:"err"`
		} `json:"runs"`
	}
	require.NoError(t, json.Unmarshal(buf.Bytes(), &decoded))
	require.Len(t, decoded.Runs, 3)
	assert.Equal(t, "rate limited", decoded.Runs[2].Err)
}

func TestPrintJSONFuncRendersDurationInMilliseconds(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSONFunc(&buf, sampleFuncResult()))

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

func TestPrintJSONFuncUsesIndentation(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, llmeval.PrintJSONFunc(&buf, sampleFuncResult()))

	assert.Contains(t, buf.String(), "\n")
}
