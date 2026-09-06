PY := .venv/bin/python
ifeq ($(OS),Windows_NT)
	PY := .venv/Scripts/python.exe
endif

.PHONY: setup samples ingest run cli-ask test lint eval clean

setup:              ## create venv + install
	py -3.12 -m venv .venv || python3.12 -m venv .venv
	$(PY) -m pip install -e ".[dev]" -q
	[ -f .env ] || cp .env.example .env

samples:            ## generate PDF/DOCX/XLSX/PNG demo files
	$(PY) scripts/make_samples.py

ingest: samples     ## ingest the whole sample corpus
	$(PY) -m grounded.cli ingest "data/samples/*"

run:                ## launch the Streamlit app
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src/ tests/ app/ eval/ scripts/

typecheck:
	$(PY) -m mypy src/

eval:               ## run the gold-set evaluation
	$(PY) eval/run_eval.py

clean:              ## drop the local index / DuckDB
	rm -rf .grounded
