package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/anthropic"
)

// newFakeAPI returns a test server that captures the most recent request
// body and replies with a canned Anthropic-shaped response. Tests use it
// to specify the wiring without burning real API credits.
type fakeAPI struct {
	server         *httptest.Server
	lastRequestRaw []byte
	reply          string
	status         int
}

func newFakeAPI() *fakeAPI {
	f := &fakeAPI{
		reply:  "1. PASS: looks fine\n",
		status: http.StatusOK,
	}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.lastRequestRaw, _ = io.ReadAll(r.Body)
		if f.status != http.StatusOK {
			http.Error(w, "boom", f.status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":    "msg_test",
			"type":  "message",
			"role":  "assistant",
			"model": "claude-haiku-4-5",
			"content": []map[string]any{
				{"type": "text", "text": f.reply},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
			},
		})
	}))
	return f
}

// lastRequest returns the most recent captured request body as a map for
// assertion-by-key tests (model, max_tokens, messages, etc.).
func (f *fakeAPI) lastRequest(t *testing.T) map[string]any {
	t.Helper()
	require.NotEmpty(t, f.lastRequestRaw, "no captured request")
	var body map[string]any
	require.NoError(t, json.Unmarshal(f.lastRequestRaw, &body))
	return body
}

// clientPointingAt returns an Anthropic client wired to the fake API. The
// API key is dummy — the fake server doesn't validate it — but the real
// SDK requires one to be set.
func clientPointingAt(t *testing.T, f *fakeAPI) *anthropicsdk.Client {
	t.Helper()
	c := anthropicsdk.NewClient(
		option.WithBaseURL(f.server.URL),
		option.WithAPIKey("test-key"),
	)
	return &c
}

// ─────────────────────────────────────────────────────────────────────────────
// Wiring contract — what the LLMFunc actually sends to Anthropic.
// ─────────────────────────────────────────────────────────────────────────────

func TestNewJudgeSendsThePromptAsAUserMessage(t *testing.T) {
	fake := newFakeAPI()
	defer fake.server.Close()

	judge := anthropic.NewJudge(clientPointingAt(t, fake))

	_, err := judge.Evaluate(context.Background(), "the SUT output", []llmeval.Criterion{
		{Description: "is concise"},
	})
	require.NoError(t, err)

	body := fake.lastRequest(t)
	messages := body["messages"].([]any)
	require.Len(t, messages, 1)
	msg := messages[0].(map[string]any)
	assert.Equal(t, "user", msg["role"])
	// The prompt built by PromptedJudge mentions the SUT output verbatim,
	// regardless of the surrounding rubric template.
	content, _ := json.Marshal(msg["content"])
	assert.Contains(t, string(content), "the SUT output")
}

func TestNewJudgeUsesDefaultModelAndMaxTokens(t *testing.T) {
	fake := newFakeAPI()
	defer fake.server.Close()

	judge := anthropic.NewJudge(clientPointingAt(t, fake))
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)

	body := fake.lastRequest(t)
	assert.Equal(t, string(anthropic.DefaultModel), body["model"])
	assert.Equal(t, float64(anthropic.DefaultMaxTokens), body["max_tokens"])
}

// ─────────────────────────────────────────────────────────────────────────────
// Options override the defaults.
// ─────────────────────────────────────────────────────────────────────────────

func TestWithModelOverridesTheDefaultModel(t *testing.T) {
	fake := newFakeAPI()
	defer fake.server.Close()

	judge := anthropic.NewJudge(
		clientPointingAt(t, fake),
		anthropic.WithModel(anthropicsdk.ModelClaudeOpus4_5),
	)
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)

	assert.Equal(t, string(anthropicsdk.ModelClaudeOpus4_5), fake.lastRequest(t)["model"])
}

func TestWithMaxTokensOverridesTheDefault(t *testing.T) {
	fake := newFakeAPI()
	defer fake.server.Close()

	judge := anthropic.NewJudge(
		clientPointingAt(t, fake),
		anthropic.WithMaxTokens(8192),
	)
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)

	assert.Equal(t, float64(8192), fake.lastRequest(t)["max_tokens"])
}

func TestWithTimeoutOverridesTheDefault(t *testing.T) {
	// Set a 5ms timeout against a server that delays for 100ms — the call
	// must hit the deadline.
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer slow.Close()

	c := anthropicsdk.NewClient(
		option.WithBaseURL(slow.URL),
		option.WithAPIKey("test-key"),
	)
	judge := anthropic.NewJudge(&c, anthropic.WithTimeout(5*time.Millisecond))

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	// Either a deadline error or the SDK wrapping one — the point is the
	// request didn't complete normally.
}

// ─────────────────────────────────────────────────────────────────────────────
// Error paths.
// ─────────────────────────────────────────────────────────────────────────────

func TestNewJudgePropagatesAPIErrors(t *testing.T) {
	fake := newFakeAPI()
	fake.status = http.StatusInternalServerError
	defer fake.server.Close()

	judge := anthropic.NewJudge(clientPointingAt(t, fake))

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
}

func TestNewJudgeErrorsWhenAPIReturnsEmptyContent(t *testing.T) {
	// A successful but content-less response should error rather than
	// returning an empty verdict to the parser, which would silently fail
	// with an unhelpful count-mismatch.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":          "msg_test",
			"type":        "message",
			"role":        "assistant",
			"model":       "claude-haiku-4-5",
			"content":     []map[string]any{}, // empty
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 1, "output_tokens": 0},
		})
	}))
	defer server.Close()

	c := anthropicsdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test-key"))
	judge := anthropic.NewJudge(&c)

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty response")
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end: a passing eval through the real PromptedJudge pipeline.
// ─────────────────────────────────────────────────────────────────────────────

func TestJudgeReturnsParsedVerdictsToTheEvalRunner(t *testing.T) {
	// Two criteria, two PASS verdicts in the API reply, the prompted-judge
	// default parser splits them, the criterion rates aggregate to pass.
	fake := newFakeAPI()
	fake.reply = "1. PASS: looks fine\n2. PASS: also fine\n"
	defer fake.server.Close()

	judge := anthropic.NewJudge(clientPointingAt(t, fake))

	result := llmeval.Run(context.Background(), llmeval.Eval{
		Run:   func(context.Context) (string, error) { return "the output", nil },
		Judge: judge,
		Criteria: []llmeval.Criterion{
			{Description: "is concise"},
			{Description: "mentions the topic"},
		},
	})

	assert.True(t, result.Pass, "result=%+v", result)
	require.Len(t, result.Criteria, 2)
	assert.Equal(t, 1, result.Criteria[0].Passed)
	assert.Equal(t, 1, result.Criteria[1].Passed)
}
