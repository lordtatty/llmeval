package llmeval_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/mocks"
)

// ─────────────────────────────────────────────────────────────────────────────
// Judge / Criteria contract — how the runner invokes the judge.
// ─────────────────────────────────────────────────────────────────────────────

func TestJudgeIsInvokedOncePerRunWithTheSUTOutputAndAllCriteria(t *testing.T) {
	criteria := []llmeval.Criterion{
		{Description: "is concise"},
		{Description: "mentions the topic"},
	}
	judge := mocks.NewMockJudge(t)
	judge.EXPECT().
		Evaluate(mock.Anything, "the summary", criteria).
		Return([]llmeval.CriterionResult{{Pass: true}, {Pass: true}}, nil).
		Once()

	result := llmeval.Run(context.Background(), llmeval.Eval[string]{
		Run:      func(context.Context) (string, error) { return "the summary", nil },
		Judge:    judge,
		Criteria: criteria,
	})

	assert.True(t, result.Pass, "result=%+v", result)
}

func TestJudgeIsInvokedOnceForEveryRepeatRun(t *testing.T) {
	judge := mocks.NewMockJudge(t)
	judge.EXPECT().
		Evaluate(mock.Anything, "x", mock.Anything).
		Return([]llmeval.CriterionResult{{Pass: true}}, nil).
		Times(5)

	result := llmeval.Run(context.Background(), llmeval.Eval[string]{
		Run:      func(context.Context) (string, error) { return "x", nil },
		Repeat:   5,
		Judge:    judge,
		Criteria: []llmeval.Criterion{{Description: "is fine"}},
	})

	assert.Equal(t, 5, result.Criteria[0].Passed)
}

func TestTolerantCriterionPassesWhenJudgeAgreesOnEnoughRuns(t *testing.T) {
	// 3 of 5 verdicts say PASS → 60%, meeting the ≥50% threshold.
	judge := mocks.NewMockJudge(t)
	expectNextVerdict := func(pass bool) {
		judge.EXPECT().Evaluate(mock.Anything, mock.Anything, mock.Anything).
			Return([]llmeval.CriterionResult{{Pass: pass}}, nil).Once()
	}
	expectNextVerdict(true)
	expectNextVerdict(false)
	expectNextVerdict(true)
	expectNextVerdict(false)
	expectNextVerdict(true)

	result := llmeval.Run(context.Background(), llmeval.Eval[string]{
		Run:      func(context.Context) (string, error) { return "x", nil },
		Repeat:   5,
		Judge:    judge,
		Criteria: []llmeval.Criterion{{Description: "passes most of the time", MinPassRate: 0.5}},
	})

	assert.True(t, result.Pass, "result=%+v", result)
	assert.Equal(t, 3, result.Criteria[0].Passed)
}

func TestStrictCriterionFailsIfAnyRunFailsTheJudge(t *testing.T) {
	// Criterion.MinPassRate=0 means strict — every run must pass.
	judge := mocks.NewMockJudge(t)
	expectNextVerdict := func(pass bool) {
		judge.EXPECT().Evaluate(mock.Anything, mock.Anything, mock.Anything).
			Return([]llmeval.CriterionResult{{Pass: pass}}, nil).Once()
	}
	expectNextVerdict(true)
	expectNextVerdict(true)
	expectNextVerdict(false)
	expectNextVerdict(true)
	expectNextVerdict(true)

	result := llmeval.Run(context.Background(), llmeval.Eval[string]{
		Run:      func(context.Context) (string, error) { return "x", nil },
		Repeat:   5,
		Judge:    judge,
		Criteria: []llmeval.Criterion{{Description: "must always hold"}},
	})

	assert.False(t, result.Pass, "result=%+v", result)
	assert.Equal(t, 4, result.Criteria[0].Passed)
}

func TestJudgeErrorMarksEveryCriterionFailedForThatRun(t *testing.T) {
	judge := mocks.NewMockJudge(t)
	judge.EXPECT().
		Evaluate(mock.Anything, mock.Anything, mock.Anything).
		Return(nil, errors.New("LLM unreachable")).
		Times(3)

	result := llmeval.Run(context.Background(), llmeval.Eval[string]{
		Run:    func(context.Context) (string, error) { return "x", nil },
		Repeat: 3,
		Judge:  judge,
		Criteria: []llmeval.Criterion{
			{Description: "is fine"},
			{Description: "is also fine"},
		},
	})

	assert.False(t, result.Pass, "result=%+v", result)
	assert.Equal(t, 0, result.Criteria[0].Passed)
	assert.Equal(t, 0, result.Criteria[1].Passed)
}

func TestVerdictCountMismatchIsTreatedAsAJudgeError(t *testing.T) {
	// A misbehaving judge that returns the wrong number of verdicts
	// would silently misalign criteria if the runner trusted it.
	judge := mocks.NewMockJudge(t)
	judge.EXPECT().
		Evaluate(mock.Anything, mock.Anything, mock.Anything).
		Return([]llmeval.CriterionResult{{Pass: true}}, nil). // 1 verdict
		Once()

	result := llmeval.Run(context.Background(), llmeval.Eval[string]{
		Run:   func(context.Context) (string, error) { return "x", nil },
		Judge: judge,
		Criteria: []llmeval.Criterion{
			{Description: "a"},
			{Description: "b"}, // 2 criteria
		},
	})

	assert.False(t, result.Pass, "result=%+v", result)
}

func TestNoJudgeMeansNoCriteriaAreRecorded(t *testing.T) {
	result := llmeval.Run(context.Background(), llmeval.Eval[string]{
		Run:        func(context.Context) (string, error) { return "x", nil },
		Assertions: []llmeval.Assertion[string]{llmeval.Equal("x")},
	})

	assert.True(t, result.Pass, "result=%+v", result)
	assert.Empty(t, result.Criteria)
}

// ─────────────────────────────────────────────────────────────────────────────
// PromptedJudge — the convenience wrapper around an LLMFunc.
//
// These tests use a plain closure as the LLMFunc (capturing whatever the
// PromptedJudge passes in); a Judge mock would only be useful if we were
// stubbing out PromptedJudge itself.
// ─────────────────────────────────────────────────────────────────────────────

func TestPromptedJudgeSendsTheSUTOutputAndCriteriaToTheLLM(t *testing.T) {
	var prompt string
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(_ context.Context, p string) (string, error) {
			prompt = p
			return "1. PASS: looks fine\n2. PASS: also fine\n", nil
		},
	}

	_, err := judge.Evaluate(context.Background(), "the SUT output here", []llmeval.Criterion{
		{Description: "is concise"},
		{Description: "mentions topic X"},
	})
	require.NoError(t, err)

	assert.Contains(t, prompt, "the SUT output here")
	assert.Contains(t, prompt, "is concise")
	assert.Contains(t, prompt, "mentions topic X")
}

func TestPromptedJudgePropagatesLLMErrorsToTheCaller(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(context.Context, string) (string, error) {
			return "", errors.New("network down")
		},
	}

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "network down")
}

func TestPromptedJudgeAppliesTimeoutToTheLLMCall(t *testing.T) {
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

func TestPromptedJudgeAcceptsACustomPromptTemplate(t *testing.T) {
	var prompt string
	judge := &llmeval.PromptedJudge{
		PromptTemplate: "JUDGE [{{.Output}}]: {{range $i, $c := .Criteria}}{{$i}}={{$c.Description}};{{end}}",
		LLMFunc: func(_ context.Context, p string) (string, error) {
			prompt = p
			return "1. PASS: ok\n", nil
		},
	}

	_, err := judge.Evaluate(context.Background(), "X", []llmeval.Criterion{{Description: "alpha"}})
	require.NoError(t, err)
	assert.Equal(t, "JUDGE [X]: 0=alpha;", prompt)
}

func TestPromptedJudgeReportsTemplateParseErrorsClearly(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		PromptTemplate: "{{", // malformed
		LLMFunc:        func(context.Context, string) (string, error) { return "", nil },
	}

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "parse template")
}

func TestPromptedJudgeReportsTemplateExecuteErrors(t *testing.T) {
	judge := &llmeval.PromptedJudge{
		PromptTemplate: "{{.Missing.Field}}",
		LLMFunc:        func(context.Context, string) (string, error) { return "", nil },
	}

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "render prompt")
}

func TestPromptedJudgeCanBeReusedAcrossEvaluateCalls(t *testing.T) {
	// Users will reuse the same judge across many evals.
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(context.Context, string) (string, error) { return "1. PASS: fine\n", nil },
	}
	criteria := []llmeval.Criterion{{Description: "a"}}

	_, err1 := judge.Evaluate(context.Background(), "first", criteria)
	require.NoError(t, err1)
	_, err2 := judge.Evaluate(context.Background(), "second", criteria)
	require.NoError(t, err2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Verdict parsers — direct function tests, no helpers needed.
// ─────────────────────────────────────────────────────────────────────────────

func TestPrefixVerdictParserExtractsPassAndFailLines(t *testing.T) {
	verdicts, err := llmeval.PrefixVerdictParser(
		"1. PASS: yes\n2. FAIL: missing topic\n",
		[]llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	)
	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.Equal(t, "yes", verdicts[0].Reason)
	assert.False(t, verdicts[1].Pass)
	assert.Equal(t, "missing topic", verdicts[1].Reason)
}

func TestPrefixVerdictParserIsCaseInsensitive(t *testing.T) {
	verdicts, err := llmeval.PrefixVerdictParser(
		"1. pass: ok\n2. Fail: nope\n",
		[]llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	)
	require.NoError(t, err)
	assert.True(t, verdicts[0].Pass)
	assert.False(t, verdicts[1].Pass)
}

func TestPrefixVerdictParserSkipsPreambleLinesFromTheLLM(t *testing.T) {
	verdicts, err := llmeval.PrefixVerdictParser(
		"Sure, here are my verdicts:\n\n1. PASS: looks fine\n2. FAIL: missing X\n",
		[]llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	)
	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.False(t, verdicts[1].Pass)
}

func TestPrefixVerdictParserReportsVerdictCountMismatch(t *testing.T) {
	_, err := llmeval.PrefixVerdictParser(
		"1. PASS: only one line\n",
		[]llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected 2")
}

func TestJSONVerdictParserExtractsVerdictsFromAJSONReply(t *testing.T) {
	verdicts, err := llmeval.JSONVerdictParser(
		`{"verdicts":[{"pass":true,"reason":"ok"},{"pass":false,"reason":"missing X"}]}`,
		[]llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	)
	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.Equal(t, "ok", verdicts[0].Reason)
	assert.False(t, verdicts[1].Pass)
	assert.Equal(t, "missing X", verdicts[1].Reason)
}

func TestJSONVerdictParserStripsLanguageTaggedMarkdownCodeFences(t *testing.T) {
	verdicts, err := llmeval.JSONVerdictParser(
		"```json\n{\"verdicts\":[{\"pass\":true,\"reason\":\"ok\"}]}\n```\n",
		[]llmeval.Criterion{{Description: "a"}},
	)
	require.NoError(t, err)
	require.Len(t, verdicts, 1)
	assert.True(t, verdicts[0].Pass)
}

func TestJSONVerdictParserStripsBareMarkdownCodeFences(t *testing.T) {
	verdicts, err := llmeval.JSONVerdictParser(
		"```\n{\"verdicts\":[{\"pass\":false,\"reason\":\"nope\"}]}\n```",
		[]llmeval.Criterion{{Description: "a"}},
	)
	require.NoError(t, err)
	require.Len(t, verdicts, 1)
	assert.False(t, verdicts[0].Pass)
}

func TestJSONVerdictParserHandlesAFenceWithNoTrailingNewline(t *testing.T) {
	verdicts, err := llmeval.JSONVerdictParser(
		"```{\"verdicts\":[{\"pass\":true,\"reason\":\"ok\"}]}```",
		[]llmeval.Criterion{{Description: "a"}},
	)
	require.NoError(t, err)
	require.Len(t, verdicts, 1)
	assert.True(t, verdicts[0].Pass)
}

func TestJSONVerdictParserReportsMalformedJSON(t *testing.T) {
	_, err := llmeval.JSONVerdictParser(`{"verdicts": [`, []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
}

func TestJSONVerdictParserReportsVerdictCountMismatch(t *testing.T) {
	_, err := llmeval.JSONVerdictParser(
		`{"verdicts":[{"pass":true,"reason":"only one"}]}`,
		[]llmeval.Criterion{{Description: "a"}, {Description: "b"}},
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expected 2")
}

// ─────────────────────────────────────────────────────────────────────────────
// Pluggable parser — Parser field on PromptedJudge.
// ─────────────────────────────────────────────────────────────────────────────

func TestPromptedJudgeUsesACustomParserWhenOneIsProvided(t *testing.T) {
	called := false
	judge := &llmeval.PromptedJudge{
		LLMFunc: func(context.Context, string) (string, error) {
			return "this would not parse as PASS/FAIL", nil
		},
		Parser: func(string, []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
			called = true
			return []llmeval.CriterionResult{{Pass: true, Reason: "via custom parser"}}, nil
		},
	}

	verdicts, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})

	require.NoError(t, err)
	assert.True(t, called)
	assert.Equal(t, "via custom parser", verdicts[0].Reason)
}

func TestPromptedJudgeJSONPairWorksEndToEnd(t *testing.T) {
	var prompt string
	judge := &llmeval.PromptedJudge{
		PromptTemplate: llmeval.JSONPromptTemplate,
		Parser:         llmeval.JSONVerdictParser,
		LLMFunc: func(_ context.Context, p string) (string, error) {
			prompt = p
			return `{"verdicts":[{"pass":true,"reason":"fine"},{"pass":true,"reason":"also fine"}]}`, nil
		},
	}

	verdicts, err := judge.Evaluate(context.Background(), "the output",
		[]llmeval.Criterion{{Description: "is concise"}, {Description: "on topic"}})

	require.NoError(t, err)
	require.Len(t, verdicts, 2)
	assert.True(t, verdicts[0].Pass)
	assert.True(t, verdicts[1].Pass)
	assert.Contains(t, prompt, "JSON")
}
