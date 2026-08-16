# Common commands. Everything data-related talks to the Postgres from
# docker-compose (or to DATABASE_URL when set — e.g. cloud Postgres).

POSTGRES_USER ?= app
POSTGRES_DB   ?= stripe_assistant
RUN = uv run --env-file .env

.PHONY: up down schema model ingest chunk embed corpus experiments psql

up:            ## start postgres (and future services)
	docker compose up -d

down:
	docker compose down

schema: up     ## apply db/schema.sql (idempotent)
	docker compose exec -T postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) < db/schema.sql

model:         ## download the ONNX embedding model (~90 MB, once)
	$(RUN) python -m services.embedder

ingest:        ## fetch all Stripe docs pages into postgres (via dlt)
	$(RUN) python -m ingestion.pipeline

chunk:         ## rebuild rag.chunks from ingested pages
	$(RUN) python -m ingestion.chunking

embed:         ## fill missing embeddings in rag.chunks
	$(RUN) python -m ingestion.embed

corpus: chunk embed  ## chunk + embed in one go

experiments:   ## list recent eval runs
	$(RUN) python -m evals.experiments

psql:          ## open a psql shell in the postgres container
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)
