-- Two roles, as recommended for real use:
--   shdi_reader can only SELECT the source (SHDI_PG_DSN)
--   shdi_writer creates and writes the staging and quarantine tables (SHDI_PG_SINK_DSN)
CREATE ROLE shdi_reader LOGIN PASSWORD 'reader';
CREATE ROLE shdi_writer LOGIN PASSWORD 'writer';
GRANT CONNECT ON DATABASE shdi TO shdi_reader, shdi_writer;
GRANT CREATE ON DATABASE shdi TO shdi_writer;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
