# Discoverable entry points for contributors. CI runs the same commands —
# see .github/workflows/ci.yml.
#
# Bare `make` prints this list. Each module is its own Go module (workspace
# at the repo root is gitignored — use `go work init . ./anthropic ./openai`
# locally if you want cross-module edits without per-module re-publish).

MODULES := . anthropic openai

# Override if you have golangci-lint somewhere other than $GOPATH/bin/.
GOLANGCI_LINT ?= $(shell go env GOPATH)/bin/golangci-lint

.PHONY: help test test-examples test-live check lint coverage tidy mocks

help:  ## List available targets.
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  %-16s %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

test:  ## Unit tests in every module (race detector on).
	@for m in $(MODULES); do \
		echo "==> test $$m"; \
		(cd $$m && go test -race ./...) || exit 1; \
	done

test-examples:  ## Build-tagged example evals (stub LLM, no API key needed).
	go test -tags=llmeval ./examples/...

test-live:  ## Live LLM prompt-calibration tests. Costs money; needs ANTHROPIC_API_KEY / OPENAI_API_KEY (env or .env / .env.local). Tests skip themselves if a key is missing.
	go test -tags=llmlive -count=1 ./anthropic ./openai

check: test test-examples coverage lint  ## Everything CI runs except the paid live tests — pre-PR sanity.

lint:  ## golangci-lint in every module.
	@for m in $(MODULES); do \
		echo "==> lint $$m"; \
		(cd $$m && $(GOLANGCI_LINT) run ./...) || exit 1; \
	done

coverage:  ## Verify every module is at 100% on non-mock, non-example files (matches CI gate).
	@for m in $(MODULES); do \
		(cd $$m && go test -coverprofile=coverage.out ./...) || exit 1; \
		gaps=$$(cd $$m && go tool cover -func=coverage.out | grep -E '^github\.com/lordtatty/llmeval' | grep -v '/examples/' | grep -v '/mocks/' | grep -v '100\.0%') || true; \
		if [ -n "$$gaps" ]; then echo "$$m below 100%:"; echo "$$gaps"; exit 1; fi; \
	done
	@echo "All modules at 100% coverage."

tidy:  ## go mod tidy in every module.
	@for m in $(MODULES); do \
		echo "==> tidy $$m"; \
		(cd $$m && go mod tidy) || exit 1; \
	done

mocks:  ## Regenerate mockery mocks from .mockery.yml.
	mockery
