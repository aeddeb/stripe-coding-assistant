# Common commands. Everything data-related talks to the Postgres from
# docker-compose (or to DATABASE_URL when set — e.g. cloud Postgres).

POSTGRES_USER ?= app
POSTGRES_DB   ?= stripe_assistant
RUN = uv run --env-file .env

.PHONY: up down schema model ingest chunk embed corpus experiments app psql

up:            ## start postgres (and future services)
	docker compose up -d

down:
	docker compose down

schema: up     ## apply db/schema.sql + read-only Grafana role (idempotent)
	docker compose exec -T postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) < db/schema.sql
	$(RUN) python -m db.grafana_role

model:         ## download the ONNX embedding model (~90 MB, once)
	$(RUN) python -m services.embedder

ingest:        ## fetch all Stripe docs pages into postgres (via dlt)
	$(RUN) python -m ingestion.pipeline

chunk:         ## rebuild rag.chunks from ingested pages
	$(RUN) python -m ingestion.chunking

embed:         ## fill missing embeddings in rag.chunks
	$(RUN) python -m ingestion.embed

corpus: chunk embed  ## chunk + embed in one go

eval:          ## run the retrieval eval grid, log to rag.experiments
	$(RUN) python -m evals.run_retrieval_eval

llm-eval:      ## judge answer-prompt variants, log to rag.experiments
	$(RUN) python -m evals.run_llm_eval

experiments:   ## list recent eval runs
	$(RUN) python -m evals.experiments

app: up        ## run the Streamlit chat UI
	$(RUN) streamlit run app/app.py

psql:          ## open a psql shell in the postgres container
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)
