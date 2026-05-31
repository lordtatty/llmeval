//go:build llmlive

// This file runs real LLM calls against Anthropic's API. Build-tagged
// (`-tags=llmlive`) so it never fires under a normal `go test`.
// Requires ANTHROPIC_API_KEY in the environment or in ../.env / ../.env.local.

package anthropic_test

import (
	"context"
	"os"
	"testing"

	"github.com/joho/godotenv"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/anthropic"
	"github.com/lordtatty/llmeval/judgetest"
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

	judge := anthropic.NewDefaultJudge(newClient(t))
	for _, c := range judgetest.Cases {
		t.Run(c.Name, func(t *testing.T) {
			judgetest.AssertCase(ctx, t, judge, c)
		})
	}
}

func TestJudgeProducesExpectedVerdictsWithJSONFormat(t *testing.T) {
	ctx, collector := llmeval.NewUsageCtx(context.Background())
	defer logUsageAndCost(t, collector)

	judge := anthropic.NewJSONJudge(newClient(t))
	for _, c := range judgetest.Cases {
		t.Run(c.Name, func(t *testing.T) {
			judgetest.AssertCase(ctx, t, judge, c)
		})
	}
}

// newClient fails the test loudly if ANTHROPIC_API_KEY is missing —
// silent skips were hiding misconfigured local setups.
func newClient(t *testing.T) *anthropicsdk.Client {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Fatal("ANTHROPIC_API_KEY required for live tests (see .env.example)")
	}
	c := anthropicsdk.NewClient(option.WithAPIKey(key))
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
	t.Logf("estimated cost: $%.4f", llmeval.TotalCost(usages, anthropic.Pricer()))
}
