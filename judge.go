package llmeval

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"text/template"
	"time"
)

// Judge is an LLM-driven verdict source. The runner calls Evaluate exactly
// once per Run (after local assertions), passing the SUT output and the full
// list of Criteria. Evaluate must return one CriterionResult per criterion,
// in the same order.
//
// Returning err, or returning a slice whose length doesn't match len(criteria),
// causes every criterion for that Run to be recorded as failed with a Reason
// explaining the judge error.
type Judge interface {
	Evaluate(ctx context.Context, output string, criteria []Criterion) ([]CriterionResult, error)
}

// Criterion is one rubric item the Judge should evaluate against the SUT output.
type Criterion struct {
	// Description is what the judge evaluates and what shows in reports.
	Description string

	// MinPassRate is the fraction of Repeat runs in which this criterion
	// must pass (verdict.Pass = true). Zero value means strict — must pass
	// every run. Set explicitly (e.g. 0.8) for tolerance.
	//
	// Note this inverts Assertion.MinPassRate()'s default (which is 1.0).
	// Zero-means-strict makes struct literals ergonomic for the common case.
	MinPassRate float64
}

// CriterionResult is the verdict for one Criterion on one Run.
type CriterionResult struct {
	Pass   bool   `json:"pass"`
	Reason string `json:"reason"`
}

// VerdictParser extracts CriterionResult verdicts from a judge LLM's raw
// reply. PromptedJudge uses it to decouple the prompt format from the
// response format — pick the parser that matches what your prompt asks for.
//
// Implementations should return an error when the parsed verdict count
// doesn't equal len(criteria) — that mismatch signals the runner to mark
// every criterion failed for that Run, surfacing the issue rather than
// silently misaligning verdicts to criteria.
type VerdictParser func(reply string, criteria []Criterion) ([]CriterionResult, error)

// PromptedJudge is a convenience Judge that builds a prompt from a template,
// sends it to a user-supplied LLM call, and parses verdicts from the reply.
// It's the default way to wire any LLM client (Anthropic, OpenAI, local
// model, mock) into the eval as a judge.
//
// The prompt format and the response format are decoupled via PromptTemplate
// and Parser. The shipped pairs are:
//
//   - Default PASS/FAIL prefix: leave both fields empty. Robust across
//     providers, works without structured-output APIs.
//   - JSON: set PromptTemplate=JSONPromptTemplate and Parser=JSONVerdictParser.
//     Pair this with a structured-output-enabled LLMFunc when possible.
//
// You can also bring your own template and/or parser.
type PromptedJudge struct {
	// LLMFunc is the bridge to your LLM client. Required.
	LLMFunc func(ctx context.Context, prompt string) (string, error)

	// PromptTemplate is a text/template source. Vars available to the template:
	//   {{.Output}}     — the SUT output being judged (string)
	//   {{.Criteria}}   — the criteria list ([]Criterion)
	// Empty uses the built-in default that asks for numbered PASS/FAIL lines.
	PromptTemplate string

	// Parser converts the LLM reply into verdicts. nil uses PrefixVerdictParser
	// (the PASS/FAIL line parser). Set to JSONVerdictParser for JSON replies,
	// or supply your own.
	Parser VerdictParser

	// Timeout, if non-zero, caps each LLMFunc call via context.WithTimeout.
	// Default 30s.
	Timeout time.Duration

	// compiled is the parsed template, lazily set on first Evaluate.
	compiled *template.Template
}

// DefaultPromptTemplate (alias of PrefixPromptTemplate) is the built-in prompt
// used when PromptedJudge.PromptTemplate is empty.
const DefaultPromptTemplate = PrefixPromptTemplate

// PrefixPromptTemplate pairs with PrefixVerdictParser. It asks the LLM to
// reply with one numbered PASS/FAIL line per criterion.
const PrefixPromptTemplate = `You are evaluating an LLM output against a list of criteria.

OUTPUT:
{{.Output}}

CRITERIA:
{{range $i, $c := .Criteria}}{{plus1 $i}}. {{$c.Description}}
{{end}}
For each criterion above, reply on its own line in EXACTLY this format:

<number>. PASS: <one-line reason>
or
<number>. FAIL: <one-line reason>

Reply with one line per criterion, in the same order, and nothing else.`

// JSONPromptTemplate pairs with JSONVerdictParser. It asks the LLM to reply
// with a single JSON object of the shape
// {"verdicts":[{"pass":bool,"reason":string},...]}.
// Use it together with Parser = JSONVerdictParser. Pair with a
// structured-output-enabled LLMFunc when possible so the reply is constrained
// to valid JSON by construction.
const JSONPromptTemplate = `You are evaluating an LLM output against a list of criteria.

OUTPUT:
{{.Output}}

CRITERIA:
{{range $i, $c := .Criteria}}{{plus1 $i}}. {{$c.Description}}
{{end}}
Reply with a single JSON object of EXACTLY this shape, and nothing else:

{"verdicts": [
  {"pass": true|false, "reason": "<one-line reason>"},
  ...
]}

The verdicts array must contain one entry per criterion above, in the same order.`

const defaultJudgeTimeout = 30 * time.Second

// verdictLine matches `<number>. PASS:` or `<number>. FAIL:` (case-insensitive)
// at the start of a trimmed line, capturing the verdict word and reason.
var verdictLine = regexp.MustCompile(`(?i)^\s*\d+\.\s*(PASS|FAIL)\s*:\s*(.*)$`)

// Evaluate implements Judge by rendering the prompt, calling LLMFunc, and
// parsing one verdict per criterion from the response.
func (p *PromptedJudge) Evaluate(ctx context.Context, output string, criteria []Criterion) ([]CriterionResult, error) {
	if err := p.ensureTemplate(); err != nil {
		return nil, fmt.Errorf("PromptedJudge: %w", err)
	}

	var buf strings.Builder
	if err := p.compiled.Execute(&buf, struct {
		Output   string
		Criteria []Criterion
	}{output, criteria}); err != nil {
		return nil, fmt.Errorf("PromptedJudge: render prompt: %w", err)
	}

	timeout := p.Timeout
	if timeout <= 0 {
		timeout = defaultJudgeTimeout
	}
	llmCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	reply, err := p.LLMFunc(llmCtx, buf.String())
	if err != nil {
		return nil, fmt.Errorf("PromptedJudge: LLM call: %w", err)
	}

	parser := p.Parser
	if parser == nil {
		parser = PrefixVerdictParser
	}
	return parser(reply, criteria)
}

// ensureTemplate parses the prompt template (default or user-supplied) once.
func (p *PromptedJudge) ensureTemplate() error {
	if p.compiled != nil {
		return nil
	}
	src := p.PromptTemplate
	if src == "" {
		src = DefaultPromptTemplate
	}
	tmpl, err := template.New("judge").Funcs(template.FuncMap{
		"plus1": func(i int) int { return i + 1 },
	}).Parse(src)
	if err != nil {
		return fmt.Errorf("parse template: %w", err)
	}
	p.compiled = tmpl
	return nil
}

// PrefixVerdictParser extracts PASS/FAIL verdicts from a reply containing one
// numbered line per criterion (e.g. `1. PASS: looks fine`). Tolerates
// preamble lines and case variations. Pair with PrefixPromptTemplate.
//
// Returns an error if the parsed verdict count doesn't equal len(criteria).
func PrefixVerdictParser(reply string, criteria []Criterion) ([]CriterionResult, error) {
	verdicts := make([]CriterionResult, 0, len(criteria))
	for line := range strings.Lines(reply) {
		m := verdictLine.FindStringSubmatch(strings.TrimRight(line, "\r\n"))
		if m == nil {
			continue
		}
		verdicts = append(verdicts, CriterionResult{
			Pass:   strings.EqualFold(m[1], "PASS"),
			Reason: strings.TrimSpace(m[2]),
		})
	}
	if len(verdicts) != len(criteria) {
		return nil, fmt.Errorf("PrefixVerdictParser: parsed %d verdict lines, expected %d", len(verdicts), len(criteria))
	}
	return verdicts, nil
}

// JSONVerdictParser parses a JSON reply of the shape
// {"verdicts":[{"pass":bool,"reason":string},...]}, tolerating leading or
// trailing whitespace and optional Markdown code fences (`json or bare`).
// Pair with JSONPromptTemplate.
//
// Returns an error if the JSON is malformed or if the verdict count doesn't
// equal len(criteria).
func JSONVerdictParser(reply string, criteria []Criterion) ([]CriterionResult, error) {
	body := stripCodeFences(strings.TrimSpace(reply))

	var wrapper struct {
		Verdicts []CriterionResult `json:"verdicts"`
	}
	if err := json.Unmarshal([]byte(body), &wrapper); err != nil {
		return nil, fmt.Errorf("JSONVerdictParser: %w", err)
	}
	if len(wrapper.Verdicts) != len(criteria) {
		return nil, fmt.Errorf("JSONVerdictParser: got %d verdicts, expected %d", len(wrapper.Verdicts), len(criteria))
	}
	return wrapper.Verdicts, nil
}

// stripCodeFences removes leading/trailing Markdown code fences (e.g. `json
// ... ` or bare `...`) that LLMs often wrap around JSON replies.
func stripCodeFences(s string) string {
	if !strings.HasPrefix(s, "```") {
		return s
	}
	// Drop the opening fence and any language tag on the same line.
	if nl := strings.IndexByte(s, '\n'); nl >= 0 {
		s = s[nl+1:]
	} else {
		s = strings.TrimPrefix(s, "```")
	}
	s = strings.TrimSuffix(strings.TrimRight(s, "\n"), "```")
	return strings.TrimSpace(s)
}
