.PHONY: test bench

PY ?= .venv/bin/python

test:
	$(PY) -m pytest -q

bench:
	$(PY) -m bench.score --rows 10000 --seed 7 --rate 0.015 --out bench/data --results bench/RESULTS.md
