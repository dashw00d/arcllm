#!/bin/bash
# Create a separate database for the churner app (Temporal uses its own)
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE churner OWNER $POSTGRES_USER;
EOSQL
