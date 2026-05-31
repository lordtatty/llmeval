// Package openai ships pre-wired llmeval.PromptedJudge constructors backed
// by OpenAI's official Go SDK. It lives in its own module so the main
// llmeval package stays SDK-free; consumers who use a different provider
// don't pay the dependency cost.
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
//	    judge := openai.NewDefaultJudge(&client)
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

// Defaults used by NewDefaultJudge when the corresponding Option isn't
// supplied.
const (
	DefaultModel     = shared.ChatModelGPT4_1Mini
	DefaultMaxTokens = int64(1024)
	DefaultTimeout   = 30 * time.Second
)

// ProviderName tags llmeval.Usage records emitted by judges built from
// this package, so Pricer and llmeval.TotalCost can route correctly
// across multi-provider setups.
const ProviderName = "openai"

// prices holds OpenAI's published per-million-token rates as of 2025-11.
// Kept unexported so consumers can't mutate it concurrently with Pricer
// reads; to use different rates, supply your own llmeval.Pricer ahead of
// openai.Pricer() in TotalCost — first match wins.
//
// Both alias names and date-stamped variants are listed so the API's
// returned Model string matches regardless of which form the SDK sent.
//
// Covered models: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano. See
// https://openai.com/pricing for current rates.
var prices = map[shared.ChatModel]llmeval.Price{
	shared.ChatModelGPT4_1:               {Input: 2.00, Output: 8.00},
	shared.ChatModelGPT4_1_2025_04_14:    {Input: 2.00, Output: 8.00},
	shared.ChatModelGPT4_1Mini:           {Input: 0.40, Output: 1.60},
	shared.ChatModelGPT4_1Mini2025_04_14: {Input: 0.40, Output: 1.60},
	shared.ChatModelGPT4_1Nano:           {Input: 0.10, Output: 0.40},
	shared.ChatModelGPT4_1Nano2025_04_14: {Input: 0.10, Output: 0.40},
}

// Pricer returns an llmeval.Pricer that knows OpenAI's current model
// prices. Compose with other providers' pricers via llmeval.TotalCost; a
// custom pricer placed earlier in the TotalCost call overrides ours.
func Pricer() llmeval.Pricer {
	return llmeval.NewPricer(ProviderName, func(model string) (llmeval.Price, bool) {
		p, ok := prices[shared.ChatModel(model)]
		return p, ok
	})
}

// Option configures a NewDefaultJudge or NewJSONJudge call.
type Option func(*config)

type config struct {
	model     shared.ChatModel
	maxTokens int64
	timeout   time.Duration
	template  string
	parser    llmeval.VerdictParser
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

// NewDefaultJudge returns an llmeval.PromptedJudge wired to call OpenAI's
// Chat Completions API via the official Go SDK. The judge uses the package
// defaults (model, max tokens, timeout) and llmeval's default Prefix
// prompt template + parser unless overridden via Option.
func NewDefaultJudge(client *openaisdk.Client, opts ...Option) *llmeval.PromptedJudge {
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
			llmeval.RecordUsage(ctx, llmeval.Usage{
				Provider:     ProviderName,
				Model:        string(resp.Model),
				InputTokens:  int(resp.Usage.PromptTokens),
				OutputTokens: int(resp.Usage.CompletionTokens),
			})
			return resp.Choices[0].Message.Content, nil
		},
	}
}

// NewJSONJudge returns a judge pre-configured for llmeval's JSON prompt
// template + verdict parser pair — use it when the underlying model
// supports structured output and you want the judge to demand JSON
// replies. It's equivalent to NewDefaultJudge(client, WithJSONFormat()),
// with later opts able to override the JSON defaults if needed.
func NewJSONJudge(client *openaisdk.Client, opts ...Option) *llmeval.PromptedJudge {
	return NewDefaultJudge(client, append([]Option{WithJSONFormat()}, opts...)...)
}
