PYTHON ?= python3
DB     := data/lakehouse.duckdb

.PHONY: setup data build quality demo ask test eval app mcp clean

setup:            ## install Python dependencies
	$(PYTHON) -m pip install --break-system-packages -r requirements.txt

data:             ## download the Olist CSVs into data/raw (skips files already present)
	$(PYTHON) scripts/download_data.py

build:            ## build bronze/silver/gold into data/lakehouse.duckdb
	$(PYTHON) -m lakehouse_agent.build_lakehouse

$(DB):
	$(MAKE) data build

quality: $(DB)    ## data-quality report (PK uniqueness, FK integrity, null rates)
	$(PYTHON) -m lakehouse_agent.quality

demo: $(DB)       ## run sample questions offline and print answers
	LLM_MODEL=mock $(PYTHON) -m lakehouse_agent.cli --demo

ask: $(DB)        ## ask one question: make ask Q="Top 5 sellers by revenue?"
	$(PYTHON) -m lakehouse_agent.cli "$(Q)"

test:             ## offline unit tests (tiny fixtures, no download needed)
	$(PYTHON) -m pytest

eval: $(DB)       ## verify gold SQL, then score the agent (mock unless a key is set)
	$(PYTHON) -m evals.run_eval --check-gold
	$(PYTHON) -m evals.run_eval

app: $(DB)        ## Streamlit chat UI
	$(PYTHON) -m streamlit run app.py

mcp: $(DB)        ## MCP server over stdio
	$(PYTHON) mcp_server.py

clean:
	rm -rf data/lakehouse.duckdb data/build_manifest.json data/*.json data/demo_output.md .pytest_cache
