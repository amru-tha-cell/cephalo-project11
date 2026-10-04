-- Reference DDL matching backend/app/models/db_models.py (SQLAlchemy generates
-- this automatically on startup; this file is for direct inspection / manual
-- provisioning on PostgreSQL).

CREATE TABLE cases (
    id               SERIAL PRIMARY KEY,
    patient_ref      VARCHAR,
    image_path       VARCHAR NOT NULL,
    image_width      INTEGER NOT NULL,
    image_height     INTEGER NOT NULL,
    pixel_spacing_mm FLOAT DEFAULT 0.1,
    status           VARCHAR DEFAULT 'uploaded',
    has_ground_truth INTEGER DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ground_truth_landmarks (
    id             SERIAL PRIMARY KEY,
    case_id        INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    landmark_index INTEGER NOT NULL,
    name           VARCHAR NOT NULL,
    x              FLOAT NOT NULL,
    y              FLOAT NOT NULL
);

CREATE TABLE predicted_landmarks (
    id                  SERIAL PRIMARY KEY,
    case_id             INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    landmark_index      INTEGER NOT NULL,
    name                VARCHAR NOT NULL,
    x                   FLOAT NOT NULL,
    y                   FLOAT NOT NULL,
    confidence          FLOAT NOT NULL,
    uncertainty_std_px  FLOAT NOT NULL,
    uncertainty_std_mm  FLOAT NOT NULL,
    reliability         VARCHAR NOT NULL,   -- High | Moderate | Low
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE corrected_landmarks (
    id             SERIAL PRIMARY KEY,
    case_id        INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    landmark_index INTEGER NOT NULL,
    name           VARCHAR NOT NULL,
    x              FLOAT NOT NULL,
    y              FLOAT NOT NULL,
    original_x     FLOAT,
    original_y     FLOAT,
    corrected_by   VARCHAR DEFAULT 'clinician',
    corrected_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE measurements (
    id              SERIAL PRIMARY KEY,
    case_id         INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    source          VARCHAR NOT NULL,   -- predicted | corrected | ground_truth
    name            VARCHAR NOT NULL,   -- SNA | SNB | ANB | Facial Angle | Y-Axis
    value           FLOAT NOT NULL,
    unit            VARCHAR DEFAULT 'deg',
    normal_low      FLOAT,
    normal_high     FLOAT,
    interpretation  VARCHAR,
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE evaluation_results (
    id                          SERIAL PRIMARY KEY,
    case_id                     INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    source                      VARCHAR DEFAULT 'predicted',
    mre_mm                      FLOAT NOT NULL,
    sdr_json                    JSON NOT NULL,   -- {"2.0mm": 0.81, "2.5mm": 0.88, ...}
    per_landmark_error_json     JSON NOT NULL,   -- {"Sella": 1.2, "Nasion": 0.8, ...}
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE case_events (
    id            SERIAL PRIMARY KEY,
    case_id       INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    event_type    VARCHAR NOT NULL,  -- upload | inference | correction | recalculation | report
    details_json  JSON,
    timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pred_case  ON predicted_landmarks(case_id);
CREATE INDEX idx_corr_case  ON corrected_landmarks(case_id);
CREATE INDEX idx_meas_case  ON measurements(case_id);
CREATE INDEX idx_eval_case  ON evaluation_results(case_id);
CREATE INDEX idx_event_case ON case_events(case_id);
