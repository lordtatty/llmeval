// Package openai ships a pre-wired llmeval.PromptedJudge backed by OpenAI's
// official Go SDK. It lives in its own module so the main llmeval package
// stays SDK-free; consumers who use a different provider don't pay the
// dependency cost.
//
// Typical usage:
//
//	import (
//	    "os"
//
//	    openaisdk "github.com/openai/openai-go"
//	    "github.com/openai/openai-go/option"
//
//	    "github.com/lordtatty/llmeval"
//	    "github.com/lordtatty/llmeval/openai"
//	    "github.com/lordtatty/llmeval/llmevaltest"
//	)
//
//	func TestMyApp(t *testing.T) {
//	    client := openaisdk.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
//	    judge := openai.NewJudge(&client)
//
//	    llmevaltest.Run(t, llmeval.Eval{
//	        Run:      myLLMCall,
//	        Judge:    judge,
//	        Criteria: []llmeval.Criterion{{Description: "is concise"}},
//	    })
//	}
package openai

import (
	"context"
	"errors"
	"time"

	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"

	"github.com/lordtatty/llmeval"
)

// Defaults used by NewJudge when the corresponding Option isn't supplied.
const (
	DefaultModel     = shared.ChatModelGPT4_1Mini
	DefaultMaxTokens = int64(1024)
	DefaultTimeout   = 30 * time.Second
)

// Option configures a NewJudge call.
type Option func(*config)

type config struct {
	model     shared.ChatModel
	maxTokens int64
	timeout   time.Duration
}

// WithModel overrides the chat model used for judging. Defaults to
// DefaultModel (gpt-4.1-mini — the cheap fast tier; judges run many
// times per suite so the default favours cost). Bump to gpt-4.1 or a
// reasoning model when the rubric demands stronger judgement.
func WithModel(model shared.ChatModel) Option {
	return func(c *config) { c.model = model }
}

// WithMaxTokens overrides the max-completion-tokens cap on each judge
// response. Defaults to DefaultMaxTokens (1024).
func WithMaxTokens(n int64) Option {
	return func(c *config) { c.maxTokens = n }
}

// WithTimeout overrides the per-Evaluate timeout PromptedJudge applies to
// the LLM call. Defaults to DefaultTimeout (30s).
func WithTimeout(d time.Duration) Option {
	return func(c *config) { c.timeout = d }
}

// NewJudge returns an llmeval.PromptedJudge wired to call OpenAI's Chat
// Completions API via the official Go SDK. The judge uses the package
// defaults unless overridden via Option.
//
// The returned PromptedJudge uses llmeval's default Prefix prompt and
// PrefixVerdictParser. Swap PromptTemplate + Parser yourself if you want
// the JSON pair.
func NewJudge(client *openaisdk.Client, opts ...Option) *llmeval.PromptedJudge {
	cfg := config{
		model:     DefaultModel,
		maxTokens: DefaultMaxTokens,
		timeout:   DefaultTimeout,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &llmeval.PromptedJudge{
		Timeout: cfg.timeout,
		LLMFunc: func(ctx context.Context, prompt string) (string, error) {
			resp, err := client.Chat.Completions.New(ctx, openaisdk.ChatCompletionNewParams{
				Model:               cfg.model,
				MaxCompletionTokens: param.NewOpt(cfg.maxTokens),
				Messages: []openaisdk.ChatCompletionMessageParamUnion{
					openaisdk.UserMessage(prompt),
				},
			})
			if err != nil {
				return "", err
			}
			if len(resp.Choices) == 0 {
				return "", errors.New("openai: empty choices from chat completion")
			}
			return resp.Choices[0].Message.Content, nil
		},
	}
}
