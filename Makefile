# No CI: run `make test` locally before committing.
.PHONY: test build

build:
	python3 scripts/build-llm-calc-data.py

test:
	python3 scripts/build-llm-calc-data.py --check || { echo "calculator/llm-calculator.js is stale: run 'make build'"; exit 1; }
	node tests/test_llm_calculator.cjs
