CREATE TABLE fuzzymatcher.house
(
    name text COLLATE pg_catalog."default",
    yearsinoffice numeric(8,0),
    party text COLLATE pg_catalog."default",
    state text COLLATE pg_catalog."default",
    district text COLLATE pg_catalog."default"
)
WITH (
    OIDS = FALSE
)
