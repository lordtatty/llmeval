// Package anthropic ships pre-wired llmeval.PromptedJudge constructors
// backed by Anthropic's official Go SDK. It lives in its own module so the
// main llmeval package stays SDK-free; consumers who use a different
// provider don't pay the dependency cost.
//
// Start with NewDefaultJudge — the recommended entry point. For
// structured-output workflows use NewJSONJudge, which is identical except
// it pre-installs llmeval's JSON prompt template and verdict parser.
//
// Typical usage:
//
//	import (
//	    "os"
//
//	    anthropicsdk "github.com/anthropics/anthropic-sdk-go"
//	    "github.com/anthropics/anthropic-sdk-go/option"
//
//	    "github.com/lordtatty/llmeval"
//	    "github.com/lordtatty/llmeval/anthropic"
//	    "github.com/lordtatty/llmeval/llmevaltest"
//	)
//
//	func TestMyApp(t *testing.T) {
//	    client := anthropicsdk.NewClient(option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
//	    judge := anthropic.NewDefaultJudge(&client)
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

// Defaults used by NewDefaultJudge when the corresponding Option isn't
// supplied.
//
// The default model is deliberately a cheap, fast tier — judges typically
// run many times per eval, so a small model keeps per-suite cost low.
// Override with WithModel to use a stronger judge when the criteria
// warrant it.
const (
	DefaultModel     = anthropicsdk.ModelClaudeHaiku4_5
	DefaultMaxTokens = int64(1024)
	DefaultTimeout   = 30 * time.Second
)

// Option configures a NewDefaultJudge or NewJSONJudge call.
type Option func(*config)

// config holds the merged defaults+overrides for one constructor call.
type config struct {
	model     anthropicsdk.Model
	maxTokens int64
	timeout   time.Duration
	template  string
	parser    llmeval.VerdictParser
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

// WithJSONFormat installs llmeval's JSON prompt template and verdict
// parser, replacing the Prefix-format defaults. NewJSONJudge applies this
// for you; use this option directly when you want to flip an otherwise
// default-configured judge to JSON.
func WithJSONFormat() Option {
	return func(c *config) {
		c.template = llmeval.JSONPromptTemplate
		c.parser = llmeval.JSONVerdictParser
	}
}

// NewDefaultJudge returns an llmeval.PromptedJudge wired to call
// Anthropic's Messages API via the official Go SDK. The judge uses the
// package defaults (model, max tokens, timeout) and llmeval's default
// Prefix prompt template + parser unless overridden via Option.
func NewDefaultJudge(client *anthropicsdk.Client, opts ...Option) *llmeval.PromptedJudge {
	cfg := config{
		model:     DefaultModel,
		maxTokens: DefaultMaxTokens,
		timeout:   DefaultTimeout,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	return &llmeval.PromptedJudge{
		Timeout:        cfg.timeout,
		PromptTemplate: cfg.template,
		Parser:         cfg.parser,
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

// NewJSONJudge returns a judge pre-configured for llmeval's JSON prompt
// template + verdict parser pair — use it when the underlying model
// supports structured output and you want the judge to demand JSON
// replies. It's equivalent to NewDefaultJudge(client, WithJSONFormat()),
// with later opts able to override the JSON defaults if needed.
func NewJSONJudge(client *anthropicsdk.Client, opts ...Option) *llmeval.PromptedJudge {
	return NewDefaultJudge(client, append([]Option{WithJSONFormat()}, opts...)...)
}
