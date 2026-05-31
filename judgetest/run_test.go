package judgetest_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/lordtatty/llmeval"
	"github.com/lordtatty/llmeval/judgetest"
	"github.com/lordtatty/llmeval/mocks"
)

// fakeT records Errorf calls so we can assert what AssertCase reported.
// It satisfies judgetest.TestingT (Helper + Errorf only).
type fakeT struct{ msgs []string }

func (f *fakeT) Helper() {}

func (f *fakeT) Errorf(format string, args ...any) {
	f.msgs = append(f.msgs, fmt.Sprintf(format, args...))
}

// stubJudge returns a canned reply for a single Evaluate call. The unit
// tests below use one stubJudge per case, mirroring how AssertCase is
// invoked per case in real callers.
type stubJudge struct {
	reply []llmeval.CriterionResult
	err   error
}

func (s *stubJudge) Evaluate(_ context.Context, _ string, _ []llmeval.Criterion) ([]llmeval.CriterionResult, error) {
	return s.reply, s.err
}

// allMatching builds a reply that matches c.Wants exactly.
func allMatching(c judgetest.Case) []llmeval.CriterionResult {
	out := make([]llmeval.CriterionResult, len(c.Wants))
	for i, want := range c.Wants {
		out[i] = llmeval.CriterionResult{Pass: want, Reason: "stub"}
	}
	return out
}

func TestAssertCaseReportsNoFailuresWhenEveryVerdictMatches(t *testing.T) {
	c := judgetest.Cases[0]
	fake := &fakeT{}

	judgetest.AssertCase(fake, &stubJudge{reply: allMatching(c)}, c)

	assert.Empty(t, fake.msgs)
}

func TestAssertCaseReportsAFailureWhenAVerdictDisagrees(t *testing.T) {
	c := judgetest.Cases[0]
	reply := allMatching(c)
	reply[0].Pass = !c.Wants[0]
	fake := &fakeT{}

	judgetest.AssertCase(fake, &stubJudge{reply: reply}, c)

	require.Len(t, fake.msgs, 1)
	assert.Contains(t, fake.msgs[0], "criterion 0")
}

func TestAssertCaseReportsAFailureWhenTheJudgeErrors(t *testing.T) {
	c := judgetest.Cases[0]
	fake := &fakeT{}

	judgetest.AssertCase(fake, &stubJudge{err: errors.New("rate limited")}, c)

	require.Len(t, fake.msgs, 1)
	assert.Contains(t, fake.msgs[0], "rate limited")
}

func TestAssertCaseReportsAFailureWhenVerdictCountMismatchesCriteria(t *testing.T) {
	// Use a multi-criterion case so dropping one verdict creates a real
	// count mismatch.
	var c judgetest.Case
	for _, cand := range judgetest.Cases {
		if len(cand.Criteria) > 1 {
			c = cand
			break
		}
	}
	require.NotEmpty(t, c.Criteria, "test setup: needed a multi-criterion case in Cases")

	reply := allMatching(c)[:len(c.Wants)-1]
	fake := &fakeT{}

	judgetest.AssertCase(fake, &stubJudge{reply: reply}, c)

	require.Len(t, fake.msgs, 1)
	assert.Contains(t, fake.msgs[0], "verdict")
}

// Verify *testing.T satisfies TestingT (compile-time check via t.Run).
func TestAssertCaseAcceptsRealTestingT(t *testing.T) {
	c := judgetest.Cases[0]
	t.Run("nested", func(nt *testing.T) {
		judgetest.AssertCase(nt, &stubJudge{reply: allMatching(c)}, c)
	})
}

// Guard against the generated mock silently falling out of sync with the
// Judge interface.
func TestGeneratedJudgeMockSatisfiesTheInterface(t *testing.T) {
	var _ llmeval.Judge = mocks.NewMockJudge(t)
}
