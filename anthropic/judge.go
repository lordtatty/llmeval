// Package anthropic ships a pre-wired llmeval.PromptedJudge backed by
// Anthropic's official Go SDK. It lives in its own module so the main
// llmeval package stays SDK-free; consumers who use a different provider
// don't pay the dependency cost.
//
// Typical usage:
//
//	import (
//	    "os"
//
//	    "github.com/anthropics/anthropic-sdk-go"
//	    "github.com/anthropics/anthropic-sdk-go/option"
//
//	    "github.com/lordtatty/llmeval"
//	    "github.com/lordtatty/llmeval/anthropic"
//	    "github.com/lordtatty/llmeval/llmevaltest"
//	)
//
//	func TestMyApp(t *testing.T) {
//	    client := anthropicsdk.NewClient(option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
//	    judge := anthropic.NewJudge(&client)
//
//	    llmevaltest.Run(t, llmeval.Eval{
//	        Run:      myLLMCall,
//	        Judge:    judge,
//	        Criteria: []llmeval.Criterion{{Description: "is concise"}},
//	    })
//	}
package anthropic

import (
	"context"
	"errors"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/lordtatty/llmeval"
)

// Defaults used by NewJudge when the corresponding Option isn't supplied.
const (
	DefaultModel     = anthropicsdk.ModelClaudeHaiku4_5
	DefaultMaxTokens = int64(1024)
	DefaultTimeout   = 30 * time.Second
)

// Option configures a NewJudge call.
type Option func(*config)

// config holds the merged defaults+overrides for one NewJudge.
type config struct {
	model     anthropicsdk.Model
	maxTokens int64
	timeout   time.Duration
}

// WithModel overrides the Claude model used for judging. Defaults to
// DefaultModel (Haiku — the cheap fast tier; judges run many times per
// suite so the default favours cost). Bump to Sonnet or Opus when the
// rubric demands stronger reasoning.
func WithModel(model anthropicsdk.Model) Option {
	return func(c *config) { c.model = model }
}

// WithMaxTokens overrides the max-tokens cap on each judge response.
// Defaults to DefaultMaxTokens (1024); enough for several PASS/FAIL verdicts
// with one-line reasons.
func WithMaxTokens(n int64) Option {
	return func(c *config) { c.maxTokens = n }
}

// WithTimeout overrides the per-Evaluate timeout PromptedJudge applies to
// the LLM call. Defaults to DefaultTimeout (30s).
func WithTimeout(d time.Duration) Option {
	return func(c *config) { c.timeout = d }
}

// NewJudge returns an llmeval.PromptedJudge wired to call Anthropic's
// Messages API via the official Go SDK. The judge uses the package
// defaults (model, max tokens, timeout) unless overridden via Option.
//
// The returned PromptedJudge uses llmeval's default Prefix prompt and
// PrefixVerdictParser. Swap PromptTemplate + Parser yourself if you want
// the JSON pair.
func NewJudge(client *anthropicsdk.Client, opts ...Option) *llmeval.PromptedJudge {
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
			resp, err := client.Messages.New(ctx, anthropicsdk.MessageNewParams{
				Model:     cfg.model,
				MaxTokens: cfg.maxTokens,
				Messages: []anthropicsdk.MessageParam{
					anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock(prompt)),
				},
			})
			if err != nil {
				return "", err
			}
			if len(resp.Content) == 0 {
				return "", errors.New("anthropic: empty response from Claude")
			}
			return resp.Content[0].Text, nil
		},
	}
}
