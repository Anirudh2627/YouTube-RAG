.PHONY: install test test-slow api ui demo eval eval-ci ingest clean

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

test:                 ## offline test suite (no model downloads)
	pytest -m "not slow"

test-slow:            ## full suite incl. real embedding/reranker models
	pytest

api:                  ## run the backend on :8000
	uvicorn app.api.main:app --reload --port 8000

ui:                   ## run the Streamlit frontend on :8501
	streamlit run frontend/streamlit_app.py

demo:                 ## ingest the offline demo fixture + chat in the terminal
	python scripts/chat.py fixture:data/fixtures/demo_lecture.json

ingest:               ## usage: make ingest URL="https://youtube.com/watch?v=..."
	python scripts/ingest.py "$(URL)"

eval:                 ## production-config evaluation (downloads models once)
	python scripts/evaluate.py --embedding bge --tag prod

eval-ci:              ## fast offline evaluation (hashing embedder, no rerank)
	python scripts/evaluate.py --embedding hashing --no-rerank --tag ci

clean:
	rm -rf data/cache data/vectorstore .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
