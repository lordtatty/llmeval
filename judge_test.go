package llmeval_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
)

// fakeJudge is a test double that records its inputs and returns a
// pre-programmed verdict list.
type fakeJudge struct {
	gotOutput   string
	gotCriteria []llmeval.Criterion
	verdicts    []llmeval.CriterionResult
	err         error
	calls       int
}

func (f *fakeJudge) Evaluate(_ context.Context, output string, criteria []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
	f.calls++
	f.gotOutput = output
	f.gotCriteria = criteria
	return f.verdicts, f.err
}

func TestRun_Judge_CalledOncePerRunWithOutputAndCriteria(t *testing.T) {
	judge := &fakeJudge{verdicts: []llmeval.CriterionResult{{Pass: true}, {Pass: true}}}
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run: func(ctx context.Context) (string, error) { return "the summary", nil },
		Judge: judge,
		Criteria: []llmeval.Criterion{
			{Description: "is concise"},
			{Description: "mentions the topic"},
		},
	})

	assert.Equal(t, 1, judge.calls)
	assert.Equal(t, "the summary", judge.gotOutput)
	require.Len(t, judge.gotCriteria, 2)
	assert.Equal(t, "is concise", judge.gotCriteria[0].Description)
	assert.True(t, result.Pass, "result=%+v", result)
}

func TestRun_Judge_AggregatesAcrossRepeats(t *testing.T) {
	// 5 runs, judge returns Pass on every call → 5/5 for the single criterion.
	judge := &fakeJudge{verdicts: []llmeval.CriterionResult{{Pass: true}}}
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:      func(ctx context.Context) (string, error) { return "x", nil },
		Repeat:   5,
		Judge:    judge,
		Criteria: []llmeval.Criterion{{Description: "is fine"}},
	})

	assert.Equal(t, 5, judge.calls)
	require.Len(t, result.Criteria, 1)
	assert.Equal(t, "is fine", result.Criteria[0].Description)
	assert.Equal(t, 5, result.Criteria[0].Passed)
	assert.Equal(t, 5, result.Criteria[0].Total)
	assert.True(t, result.Criteria[0].Pass)
}

func TestRun_Judge_TolerantCriterionPassesAboveThreshold(t *testing.T) {
	// Alternate verdicts: pass,fail,pass,fail,pass → 3/5 = 60% ≥ 50%
	calls := 0
	judge := &programmableJudge{
		fn: func() []llmeval.CriterionResult {
			calls++
			return []llmeval.CriterionResult{{Pass: calls%2 == 1}}
		},
	}
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:      func(ctx context.Context) (string, error) { return "x", nil },
		Repeat:   5,
		Judge:    judge,
		Criteria: []llmeval.Criterion{{Description: "passes 50% of time", MinPassRate: 0.5}},
	})

	require.Len(t, result.Criteria, 1)
	assert.Equal(t, 3, result.Criteria[0].Passed)
	assert.Equal(t, 5, result.Criteria[0].Total)
	assert.True(t, result.Criteria[0].Pass)
	assert.True(t, result.Pass)
}

func TestRun_Judge_StrictCriterionFailsOnAnyMiss(t *testing.T) {
	// 4 pass, 1 fail → strict requires 5/5 → criterion fails → eval fails.
	calls := 0
	judge := &programmableJudge{
		fn: func() []llmeval.CriterionResult {
			calls++
			return []llmeval.CriterionResult{{Pass: calls != 3}} // call 3 fails
		},
	}
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:      func(ctx context.Context) (string, error) { return "x", nil },
		Repeat:   5,
		Judge:    judge,
		Criteria: []llmeval.Criterion{{Description: "must always hold"}}, // MinPassRate 0 = strict
	})

	require.Len(t, result.Criteria, 1)
	assert.Equal(t, 4, result.Criteria[0].Passed)
	assert.False(t, result.Criteria[0].Pass)
	assert.False(t, result.Pass)
}

func TestRun_Judge_ErrorCausesAllCriteriaToFailThatRun(t *testing.T) {
	judge := &fakeJudge{err: errors.New("LLM unreachable")}
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:      func(ctx context.Context) (string, error) { return "x", nil },
		Repeat:   3,
		Judge:    judge,
		Criteria: []llmeval.Criterion{{Description: "is fine"}, {Description: "is also fine"}},
	})

	require.Len(t, result.Criteria, 2)
	// Every run errored at the judge step → 0 passes for each criterion.
	for _, c := range result.Criteria {
		assert.Equal(t, 0, c.Passed)
		assert.Equal(t, 3, c.Total)
		assert.False(t, c.Pass)
	}
	assert.False(t, result.Pass)
}

func TestRun_Judge_VerdictCountMismatchTreatedAsJudgeError(t *testing.T) {
	// Judge returns 1 verdict but we have 2 criteria → both fail with parse error.
	judge := &fakeJudge{verdicts: []llmeval.CriterionResult{{Pass: true}}}
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:      func(ctx context.Context) (string, error) { return "x", nil },
		Judge:    judge,
		Criteria: []llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	})

	require.Len(t, result.Criteria, 2)
	assert.False(t, result.Criteria[0].Pass)
	assert.False(t, result.Criteria[1].Pass)
	assert.False(t, result.Pass)
}

func TestRun_NoJudge_NoCriteriaInResult(t *testing.T) {
	// Sanity: without Judge+Criteria, result.Criteria stays nil.
	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:        func(ctx context.Context) (string, error) { return "x", nil },
		Assertions: []llmeval.Assertion{llmeval.Equal("x")},
	})
	assert.Nil(t, result.Criteria)
	assert.True(t, result.Pass)
}

// programmableJudge lets a test compute verdicts dynamically per call.
type programmableJudge struct {
	fn func() []llmeval.CriterionResult
}

func (p *programmableJudge) Evaluate(_ context.Context, _ string, _ []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
	return p.fn(), nil
}

// --- PromptedJudge tests ---

func TestPromptedJudge_DefaultPromptIncludesOutputAndCriteria(t *testing.T) {
	var gotPrompt string
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(ctx context.Context, prompt string) (string, error) {
			gotPrompt = prompt
			return "1. PASS: looks fine\n2. PASS: also fine\n", nil
		},
	}
	_, err := judge.Evaluate(context.Background(), "the SUT output here", []llmeval.Criterion{
		{Description: "is concise"},
		{Description: "mentions topic X"},
	})
	require.NoError(t, err)
	assert.Contains(t, gotPrompt, "the SUT output here")
	assert.Contains(t, gotPrompt, "is concise")
	assert.Contains(t, gotPrompt, "mentions topic X")
}

func TestPromptedJudge_ParsesPassFailLines(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, _ string) (string, error) {
			return "1. PASS: yes\n2. FAIL: missing the topic\n", nil
		},
	}
	verdicts, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{
		{Description: "a"}, {Description: "b"},
	})
	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.Equal(t, "yes", verdicts[0].Reason)
	assert.False(t, verdicts[1].Pass)
	assert.Equal(t, "missing the topic", verdicts[1].Reason)
}

func TestPromptedJudge_PassFailParseIsCaseInsensitive(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, _ string) (string, error) {
			return "1. pass: ok\n2. Fail: nope\n", nil
		},
	}
	verdicts, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{
		{Description: "a"}, {Description: "b"},
	})
	require.NoError(t, err)
	assert.True(t, verdicts[0].Pass)
	assert.False(t, verdicts[1].Pass)
}

func TestPromptedJudge_VerdictCountMismatchReturnsError(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, _ string) (string, error) {
			return "1. PASS: only one line\n", nil
		},
	}
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{
		{Description: "a"}, {Description: "b"},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected 2")
}

func TestPromptedJudge_LLMFuncErrorIsReturned(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, _ string) (string, error) {
			return "", errors.New("network down")
		},
	}
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "network down")
}

func TestPromptedJudge_CustomTemplateIsUsed(t *testing.T) {
	var gotPrompt string
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, prompt string) (string, error) {
			gotPrompt = prompt
			return "1. PASS: ok\n", nil
		},
		PromptTemplate: "JUDGE [{{.Output}}]: {{range $i, $c := .Criteria}}{{$i}}={{$c.Description}};{{end}}",
	}
	_, err := judge.Evaluate(context.Background(), "X", []llmeval.Criterion{{Description: "alpha"}})
	require.NoError(t, err)
	assert.Equal(t, "JUDGE [X]: 0=alpha;", gotPrompt)
}

func TestPromptedJudge_IgnoresPreambleLinesFromLLM(t *testing.T) {
	// LLMs often add preamble like "Here are the verdicts:". The parser
	// should skip lines that don't match the verdict format.
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, _ string) (string, error) {
			return "Sure, here are my verdicts:\n\n1. PASS: looks fine\n2. FAIL: missing X\n", nil
		},
	}
	verdicts, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{
		{Description: "a"}, {Description: "b"},
	})
	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.False(t, verdicts[1].Pass)
}

func TestPromptedJudge_BadTemplateReturnsError(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		PromptTemplate: "{{", // malformed
		LLMFunc:        func(_ context.Context, _ string) (string, error) { return "", nil },
	}
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "parse template")
}

func TestPromptedJudge_TemplateExecuteErrorReturnsError(t *testing.T) {
	// Template parses but fails at execute (calls a method that returns error).
	judge := &llmeval.PromptedJudge{
		PromptTemplate: "{{.Missing.Field}}", // .Output has no .Missing field
		LLMFunc:        func(_ context.Context, _ string) (string, error) { return "", nil },
	}
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "render prompt")
}

func TestPromptedJudge_CustomParserIsUsed(t *testing.T) {
	// A custom parser ignores the reply text and just returns canned verdicts.
	called := false
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, _ string) (string, error) {
			return "this would not parse as PASS/FAIL", nil
		},
		Parser: func(_ string, _ []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
			called = true
			return []llmeval.CriterionResult{{Pass: true, Reason: "via custom parser"}}, nil
		},
	}
	verdicts, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)
	assert.True(t, called)
	require.Len(t, verdicts, 1)
	assert.Equal(t, "via custom parser", verdicts[0].Reason)
}

// --- Built-in parsers ---

func TestPrefixVerdictParser_ParsesPassFailLines(t *testing.T) {
	verdicts, err := llmeval.PrefixVerdictParser(
		"1. PASS: yes\n2. FAIL: missing topic\n",
		[]llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	)
	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.Equal(t, "yes", verdicts[0].Reason)
	assert.False(t, verdicts[1].Pass)
}

func TestJSONVerdictParser_ParsesPlainJSON(t *testing.T) {
	reply := `{"verdicts":[{"pass":true,"reason":"ok"},{"pass":false,"reason":"missing X"}]}`
	verdicts, err := llmeval.JSONVerdictParser(reply, []llmeval.Criterion{
		{Description: "a"}, {Description: "b"},
	})
	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.Equal(t, "ok", verdicts[0].Reason)
	assert.False(t, verdicts[1].Pass)
	assert.Equal(t, "missing X", verdicts[1].Reason)
}

func TestJSONVerdictParser_StripsCodeFences(t *testing.T) {
	reply := "```json\n{\"verdicts\":[{\"pass\":true,\"reason\":\"ok\"}]}\n```\n"
	verdicts, err := llmeval.JSONVerdictParser(reply, []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)
	require.Len(t, verdicts, 1)
	assert.True(t, verdicts[0].Pass)
}

func TestJSONVerdictParser_BareCodeFenceWithoutLang(t *testing.T) {
	reply := "```\n{\"verdicts\":[{\"pass\":false,\"reason\":\"nope\"}]}\n```"
	verdicts, err := llmeval.JSONVerdictParser(reply, []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)
	require.Len(t, verdicts, 1)
	assert.False(t, verdicts[0].Pass)
	assert.Equal(t, "nope", verdicts[0].Reason)
}

func TestJSONVerdictParser_CodeFenceWithoutNewline(t *testing.T) {
	// Pathological but possible: an LLM emits a fence-without-newline before
	// the JSON. The parser strips the opening fence even when no newline
	// follows it.
	reply := "```{\"verdicts\":[{\"pass\":true,\"reason\":\"ok\"}]}```"
	verdicts, err := llmeval.JSONVerdictParser(reply, []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)
	require.Len(t, verdicts, 1)
	assert.True(t, verdicts[0].Pass)
}

func TestJSONVerdictParser_MalformedJSONReturnsError(t *testing.T) {
	_, err := llmeval.JSONVerdictParser(`{"verdicts": [`, []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
}

func TestJSONVerdictParser_VerdictCountMismatchReturnsError(t *testing.T) {
	_, err := llmeval.JSONVerdictParser(
		`{"verdicts":[{"pass":true,"reason":"only one"}]}`,
		[]llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected 2")
}

func TestPromptedJudge_JSONPair_FullRoundTrip(t *testing.T) {
	// The shipped pair: JSONPromptTemplate + JSONVerdictParser.
	judge := &llmeval.PromptedJudge{
		PromptTemplate: llmeval.JSONPromptTemplate,
		Parser:         llmeval.JSONVerdictParser,
		LLMFunc: func(_ context.Context, prompt string) (string, error) {
			// Sanity: the JSON template asks for a JSON reply.
			assert.Contains(t, prompt, "JSON")
			return `{"verdicts":[{"pass":true,"reason":"fine"},{"pass":true,"reason":"also fine"}]}`, nil
		},
	}
	verdicts, err := judge.Evaluate(context.Background(), "the output",
		[]llmeval.Criterion{{Description: "is concise"}, {Description: "on topic"}})
	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.True(t, verdicts[1].Pass)
}

func TestPromptedJudge_ReusedAcrossEvaluatesCachesTemplate(t *testing.T) {
	// Same PromptedJudge invoked twice should work — template is parsed once
	// and reused (the second call hits the early-return in ensureTemplate).
	calls := 0
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, _ string) (string, error) {
			calls++
			return "1. PASS: fine\n", nil
		},
	}
	criteria := []llmeval.Criterion{{Description: "a"}}
	_, err := judge.Evaluate(context.Background(), "x", criteria)
	require.NoError(t, err)
	_, err = judge.Evaluate(context.Background(), "y", criteria)
	require.NoError(t, err)
	assert.Equal(t, 2, calls)
}

func TestPromptedJudge_TimeoutIsAppliedToLLMFunc(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		Timeout: 5 * time.Millisecond,
		LLMFunc: func(ctx context.Context, _ string) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		},
	}
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}
