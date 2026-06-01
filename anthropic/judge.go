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
//	    llmevaltest.Run(t, llmeval.Eval[string]{
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

// ProviderName tags llmeval.Usage records emitted by judges built from
// this package, so Pricer and llmeval.TotalCost can route correctly
// across multi-provider setups.
const ProviderName = "anthropic"

// prices holds Anthropic's published per-million-token rates as of 2025-11.
// Kept unexported so consumers can't mutate it concurrently with Pricer
// reads; to use different rates, supply your own llmeval.Pricer ahead of
// anthropic.Pricer() in TotalCost — first match wins.
//
// Both alias names and date-stamped variants are listed so the API's
// returned Model string matches regardless of which form the SDK sent.
//
// Covered models: claude-haiku-4-5, claude-sonnet-4-5, claude-sonnet-4-6,
// claude-opus-4-5. See https://anthropic.com/pricing for current rates.
var prices = map[anthropicsdk.Model]llmeval.Price{
	anthropicsdk.ModelClaudeHaiku4_5:           {Input: 1.00, Output: 5.00},
	anthropicsdk.ModelClaudeHaiku4_5_20251001:  {Input: 1.00, Output: 5.00},
	anthropicsdk.ModelClaudeSonnet4_5:          {Input: 3.00, Output: 15.00},
	anthropicsdk.ModelClaudeSonnet4_5_20250929: {Input: 3.00, Output: 15.00},
	anthropicsdk.ModelClaudeSonnet4_6:          {Input: 3.00, Output: 15.00},
	anthropicsdk.ModelClaudeOpus4_5:            {Input: 15.00, Output: 75.00},
	anthropicsdk.ModelClaudeOpus4_5_20251101:   {Input: 15.00, Output: 75.00},
}

// Pricer returns an llmeval.Pricer that knows Anthropic's current model
// prices. Compose with other providers' pricers via llmeval.TotalCost; a
// custom pricer placed earlier in the TotalCost call overrides ours.
func Pricer() llmeval.Pricer {
	return llmeval.NewPricer(ProviderName, func(model string) (llmeval.Price, bool) {
		p, ok := prices[anthropicsdk.Model(model)]
		return p, ok
	})
}

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
			llmeval.RecordUsage(ctx, llmeval.Usage{
				Provider:     ProviderName,
				Model:        string(resp.Model),
				InputTokens:  int(resp.Usage.InputTokens),
				OutputTokens: int(resp.Usage.OutputTokens),
			})
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
