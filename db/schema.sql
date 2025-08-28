-- Core tables for ARGO
CREATE TABLE IF NOT EXISTS profiles (
    profile_id SERIAL PRIMARY KEY,
    platform_number VARCHAR(16) NOT NULL,
    cycle_number INTEGER,
    juld TIMESTAMP WITH TIME ZONE,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    direction CHAR(1),
    data_mode CHAR(1),
    region TEXT,
    geom GEOGRAPHY(POINT, 4326)
);

CREATE INDEX IF NOT EXISTS idx_profiles_geom ON profiles USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_profiles_time ON profiles(juld);

CREATE TABLE IF NOT EXISTS levels (
    level_id BIGSERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(profile_id) ON DELETE CASCADE,
    pres_dbar DOUBLE PRECISION,
    temp_degC DOUBLE PRECISION,
    psal_psu DOUBLE PRECISION,
    pres_qc CHAR(1),
    temp_qc CHAR(1),
    psal_qc CHAR(1)
);

CREATE INDEX IF NOT EXISTS idx_levels_profile ON levels(profile_id);
CREATE INDEX IF NOT EXISTS idx_levels_depth ON levels(pres_dbar);
