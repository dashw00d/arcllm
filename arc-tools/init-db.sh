#!/bin/bash
# Create databases for churner and ghostgraph (Temporal uses its own)
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE churner OWNER $POSTGRES_USER;
    CREATE DATABASE ghostgraph OWNER $POSTGRES_USER;
EOSQL
# Enable pgvector in ghostgraph
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname ghostgraph <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL
