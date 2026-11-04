-- AgriSight-BD modeling database.
-- One observation is one district, one year, and one crop.
-- Climate and soil are shared by every crop in a district and year.
-- Season rainfall, season temperature, and NDVI stay on the observation
-- because each crop uses its own window.

DROP VIEW IF EXISTS crop_dataset;
DROP VIEW IF EXISTS yield_record;

DROP TABLE IF EXISTS trained_models;
DROP TABLE IF EXISTS observations;
DROP TABLE IF EXISTS climate;
DROP TABLE IF EXISTS soil;
DROP TABLE IF EXISTS crops;
DROP TABLE IF EXISTS districts;
DROP TABLE IF EXISTS divisions;

CREATE TABLE divisions (
    id SMALLINT UNSIGNED NOT NULL,
    name VARCHAR(64) NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_divisions_name (name)
) ENGINE=InnoDB;

CREATE TABLE districts (
    id SMALLINT UNSIGNED NOT NULL,
    name VARCHAR(64) NOT NULL,
    division_id SMALLINT UNSIGNED NOT NULL,
    latitude DECIMAL(8, 4) NOT NULL,
    longitude DECIMAL(8, 4) NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_districts_name (name),
    CONSTRAINT fk_districts_division
        FOREIGN KEY (division_id) REFERENCES divisions (id)
) ENGINE=InnoDB;

CREATE TABLE crops (
    id VARCHAR(32) NOT NULL,
    label VARCHAR(64) NOT NULL,
    crop_group VARCHAR(32) NOT NULL,
    season VARCHAR(160) NOT NULL,
    yield_source ENUM('bbs', 'sample') NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

CREATE TABLE soil (
    district_id SMALLINT UNSIGNED NOT NULL,
    year SMALLINT NOT NULL,
    ph DOUBLE NOT NULL,
    soc DOUBLE NOT NULL,
    nitrogen DOUBLE NOT NULL,
    cec DOUBLE NOT NULL,
    clay DOUBLE NOT NULL,
    sand DOUBLE NOT NULL,
    PRIMARY KEY (district_id, year),
    CONSTRAINT fk_soil_district
        FOREIGN KEY (district_id) REFERENCES districts (id)
) ENGINE=InnoDB;

CREATE TABLE climate (
    district_id SMALLINT UNSIGNED NOT NULL,
    year SMALLINT NOT NULL,
    temp_max_mean DOUBLE NOT NULL,
    temp_min_mean DOUBLE NOT NULL,
    rainfall_total DOUBLE NOT NULL,
    humidity_mean DOUBLE NOT NULL,
    solar_radiation_mean DOUBLE NOT NULL,
    PRIMARY KEY (district_id, year),
    CONSTRAINT fk_climate_district
        FOREIGN KEY (district_id) REFERENCES districts (id)
) ENGINE=InnoDB;

CREATE TABLE observations (
    district_id SMALLINT UNSIGNED NOT NULL,
    crop_id VARCHAR(32) NOT NULL,
    year SMALLINT NOT NULL,
    yield_ton_ha DOUBLE NOT NULL,
    rainfall_mar_jun DOUBLE NOT NULL,
    temp_max_mar_jun DOUBLE NOT NULL,
    ndvi_mean DOUBLE NOT NULL,
    ndvi_peak DOUBLE NOT NULL,
    ndvi_std DOUBLE NOT NULL,
    PRIMARY KEY (district_id, crop_id, year),
    KEY ix_observations_crop_year (crop_id, year),
    CONSTRAINT fk_observations_district
        FOREIGN KEY (district_id) REFERENCES districts (id),
    CONSTRAINT fk_observations_crop
        FOREIGN KEY (crop_id) REFERENCES crops (id),
    CONSTRAINT fk_observations_climate
        FOREIGN KEY (district_id, year) REFERENCES climate (district_id, year)
) ENGINE=InnoDB;

CREATE TABLE trained_models (
    crop_id VARCHAR(32) NOT NULL,
    best_model VARCHAR(32) NOT NULL,
    feature_names JSON NOT NULL,
    results JSON NOT NULL,
    n_samples INT UNSIGNED NOT NULL,
    year_start SMALLINT NOT NULL,
    year_end SMALLINT NOT NULL,
    mean_yield DOUBLE NOT NULL,
    artifact_dir VARCHAR(255) NOT NULL,
    PRIMARY KEY (crop_id),
    CONSTRAINT fk_trained_models_crop
        FOREIGN KEY (crop_id) REFERENCES crops (id)
) ENGINE=InnoDB;

CREATE VIEW yield_record AS
SELECT
    o.year,
    d.name AS district,
    dv.name AS division,
    d.latitude,
    d.longitude,
    o.crop_id AS crop,
    o.yield_ton_ha
FROM observations AS o
INNER JOIN districts AS d ON d.id = o.district_id
INNER JOIN divisions AS dv ON dv.id = d.division_id;

CREATE VIEW crop_dataset AS
SELECT
    o.year,
    d.name AS district,
    dv.name AS division,
    d.latitude,
    d.longitude,
    o.crop_id AS crop,
    o.yield_ton_ha,
    c.temp_max_mean,
    c.temp_min_mean,
    c.rainfall_total,
    c.humidity_mean,
    c.solar_radiation_mean,
    o.rainfall_mar_jun,
    o.temp_max_mar_jun,
    s.cec AS soil_cec,
    s.clay AS soil_clay,
    s.nitrogen AS soil_nitrogen,
    s.ph AS soil_ph,
    s.sand AS soil_sand,
    s.soc AS soil_soc,
    o.ndvi_mean,
    o.ndvi_peak,
    o.ndvi_std
FROM observations AS o
INNER JOIN districts AS d ON d.id = o.district_id
INNER JOIN divisions AS dv ON dv.id = d.division_id
INNER JOIN climate AS c
    ON c.district_id = o.district_id
    AND c.year = o.year
INNER JOIN soil AS s
    ON s.district_id = o.district_id
    AND s.year = o.year;
