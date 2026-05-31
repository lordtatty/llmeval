//go:build llmlive

// This file runs real LLM calls against OpenAI's API. Build-tagged
// (`-tags=llmlive`) so it never fires under a normal `go test`.
// Requires OPENAI_API_KEY in the environment or in ../.env / ../.env.local.

package openai_test

import (
	"os"
	"testing"

	"github.com/joho/godotenv"

	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

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
	judge := openai.NewDefaultJudge(newClient(t))
	for _, c := range judgetest.Cases {
		t.Run(c.Name, func(t *testing.T) {
			judgetest.AssertCase(t, judge, c)
		})
	}
}

func TestJudgeProducesExpectedVerdictsWithJSONFormat(t *testing.T) {
	judge := openai.NewJSONJudge(newClient(t))
	for _, c := range judgetest.Cases {
		t.Run(c.Name, func(t *testing.T) {
			judgetest.AssertCase(t, judge, c)
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
