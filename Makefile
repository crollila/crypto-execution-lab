.PHONY: install test lint reproduce quick capture

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

reproduce:
	python -m execlab reproduce

quick:
	python -m execlab reproduce --quick --out /tmp/execlab-results --figures /tmp/execlab-figures

capture:
	python -m execlab capture --seconds 1800 --out data/samples/new_capture.jsonl.gz
