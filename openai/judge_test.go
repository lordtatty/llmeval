package openai_test

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

	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/openai"
)

// fakeAPI captures the last request body and replies with a canned
// Chat-Completions-shaped response — enough to specify our wiring without
// burning real API credits.
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
			"id":      "chatcmpl_test",
			"object":  "chat.completion",
			"created": 0,
			"model":   "gpt-4.1-mini",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message": map[string]any{
					"role":    "assistant",
					"content": f.reply,
				},
			}},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		})
	}))
	return f
}

func (f *fakeAPI) lastRequest(t *testing.T) map[string]any {
	t.Helper()
	require.NotEmpty(t, f.lastRequestRaw, "no captured request")
	var body map[string]any
	require.NoError(t, json.Unmarshal(f.lastRequestRaw, &body))
	return body
}

// clientPointingAt returns an OpenAI client wired to the fake API. API key
// is dummy — fake server doesn't validate it — but the SDK requires one.
func clientPointingAt(t *testing.T, f *fakeAPI) *openaisdk.Client {
	t.Helper()
	c := openaisdk.NewClient(
		option.WithBaseURL(f.server.URL),
		option.WithAPIKey("test-key"),
	)
	return &c
}

// ─────────────────────────────────────────────────────────────────────────────
// Wiring contract — what the LLMFunc actually sends to OpenAI.
// ─────────────────────────────────────────────────────────────────────────────

func TestNewJudgeSendsThePromptAsAUserMessage(t *testing.T) {
	fake := newFakeAPI()
	defer fake.server.Close()

	judge := openai.NewJudge(clientPointingAt(t, fake))

	_, err := judge.Evaluate(context.Background(), "the SUT output", []llmeval.Criterion{
		{Description: "is concise"},
	})
	require.NoError(t, err)

	body := fake.lastRequest(t)
	messages := body["messages"].([]any)
	require.Len(t, messages, 1)
	msg := messages[0].(map[string]any)
	assert.Equal(t, "user", msg["role"])
	assert.Contains(t, msg["content"].(string), "the SUT output")
}

func TestNewJudgeUsesDefaultModelAndMaxTokens(t *testing.T) {
	fake := newFakeAPI()
	defer fake.server.Close()

	judge := openai.NewJudge(clientPointingAt(t, fake))
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)

	body := fake.lastRequest(t)
	assert.Equal(t, string(openai.DefaultModel), body["model"])
	assert.Equal(t, float64(openai.DefaultMaxTokens), body["max_completion_tokens"])
}

// ─────────────────────────────────────────────────────────────────────────────
// Options override the defaults.
// ─────────────────────────────────────────────────────────────────────────────

func TestWithModelOverridesTheDefaultModel(t *testing.T) {
	fake := newFakeAPI()
	defer fake.server.Close()

	judge := openai.NewJudge(
		clientPointingAt(t, fake),
		openai.WithModel(shared.ChatModelO4Mini),
	)
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)

	assert.Equal(t, string(shared.ChatModelO4Mini), fake.lastRequest(t)["model"])
}

func TestWithMaxTokensOverridesTheDefault(t *testing.T) {
	fake := newFakeAPI()
	defer fake.server.Close()

	judge := openai.NewJudge(
		clientPointingAt(t, fake),
		openai.WithMaxTokens(8192),
	)
	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.NoError(t, err)

	assert.Equal(t, float64(8192), fake.lastRequest(t)["max_completion_tokens"])
}

func TestWithTimeoutOverridesTheDefault(t *testing.T) {
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer slow.Close()

	c := openaisdk.NewClient(
		option.WithBaseURL(slow.URL),
		option.WithAPIKey("test-key"),
	)
	judge := openai.NewJudge(&c, openai.WithTimeout(5*time.Millisecond))

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
}

// ─────────────────────────────────────────────────────────────────────────────
// Error paths.
// ─────────────────────────────────────────────────────────────────────────────

func TestNewJudgePropagatesAPIErrors(t *testing.T) {
	fake := newFakeAPI()
	fake.status = http.StatusInternalServerError
	defer fake.server.Close()

	judge := openai.NewJudge(clientPointingAt(t, fake))

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
}

func TestNewJudgeErrorsWhenAPIReturnsEmptyChoices(t *testing.T) {
	// A successful but choice-less response should error rather than
	// silently returning empty content.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl_test",
			"object":  "chat.completion",
			"created": 0,
			"model":   "gpt-4.1-mini",
			"choices": []map[string]any{}, // empty
			"usage": map[string]any{
				"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1,
			},
		})
	}))
	defer server.Close()

	c := openaisdk.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test-key"))
	judge := openai.NewJudge(&c)

	_, err := judge.Evaluate(context.Background(), "x", []llmeval.Criterion{{Description: "a"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty choices")
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end: a passing eval through the real PromptedJudge pipeline.
// ─────────────────────────────────────────────────────────────────────────────

func TestJudgeReturnsParsedVerdictsToTheEvalRunner(t *testing.T) {
	fake := newFakeAPI()
	fake.reply = "1. PASS: looks fine\n2. PASS: also fine\n"
	defer fake.server.Close()

	judge := openai.NewJudge(clientPointingAt(t, fake))

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
