-- ============================================================
-- 🗄️ PostgreSQL Storage Overview — Running Agent Project
-- ============================================================
-- Purpose:
--   Quickly inspect how much data is stored in your PostgreSQL
--   database and where it's physically located.
--
-- Usage:
--   Run this in pgAdmin's Query Tool or via psql.
-- ============================================================

-- 1️⃣ Database Size Summary
SELECT
    current_database() AS database_name,
    pg_size_pretty(pg_database_size(current_database())) AS total_size;

-- 2️⃣ Table Storage Breakdown
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_only,
    pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS indexes
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

-- 3️⃣ Disk Location of Data Directory
SHOW data_directory;

-- 4️⃣ Optional: Row Counts per Table
SELECT
    relname AS table_name,
    n_live_tup AS approx_rows
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

