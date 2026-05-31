//go:build llmlive

// This file runs real LLM calls against OpenAI's API. Build-tagged
// (`-tags=llmlive`) so it never fires under a normal `go test`.
// Requires OPENAI_API_KEY in the environment or in ../.env / ../.env.local.

package openai_test

import (
	"context"
	"os"
	"testing"

	"github.com/joho/godotenv"

	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/judgetest"
	"github.com/lordtatty/llmeval/openai"
)

func TestMain(m *testing.M) {
	// .env.local overrides .env (loaded first because godotenv.Load
	// doesn't override variables already in the env).
	_ = godotenv.Load("../.env.local")
	_ = godotenv.Load("../.env")
	os.Exit(m.Run())
}

func TestJudgeProducesExpectedVerdictsWithDefaultFormat(t *testing.T) {
	ctx, collector := llmeval.NewUsageCtx(context.Background())
	defer logUsageAndCost(t, collector)

	judge := openai.NewDefaultJudge(newClient(t))
	for _, c := range judgetest.Cases {
		t.Run(c.Name, func(t *testing.T) {
			judgetest.AssertCase(ctx, t, judge, c)
		})
	}
}

func TestJudgeProducesExpectedVerdictsWithJSONFormat(t *testing.T) {
	ctx, collector := llmeval.NewUsageCtx(context.Background())
	defer logUsageAndCost(t, collector)

	judge := openai.NewJSONJudge(newClient(t))
	for _, c := range judgetest.Cases {
		t.Run(c.Name, func(t *testing.T) {
			judgetest.AssertCase(ctx, t, judge, c)
		})
	}
}

// newClient fails the test loudly if OPENAI_API_KEY is missing — silent
// skips were hiding misconfigured local setups.
func newClient(t *testing.T) *openaisdk.Client {
	t.Helper()
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Fatal("OPENAI_API_KEY required for live tests (see .env.example)")
	}
	c := openaisdk.NewClient(option.WithAPIKey(key))
	return &c
}

// logUsageAndCost surfaces the token usage and an estimated dollar cost
// for the test that just ran, so `make test-live -v` shows what each
// suite is costing over time.
func logUsageAndCost(t *testing.T, c *llmeval.UsageCollector) {
	t.Helper()
	usages := c.Aggregated()
	if len(usages) == 0 {
		return
	}
	for _, u := range usages {
		t.Logf("usage: %s/%s  %d in / %d out",
			u.Provider, u.Model, u.InputTokens, u.OutputTokens)
	}
	t.Logf("estimated cost: $%.4f", llmeval.TotalCost(usages, openai.Pricer()))
}
