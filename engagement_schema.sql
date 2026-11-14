-- Additive schema only. These tables do not modify attendance or student tables.
CREATE TABLE IF NOT EXISTS engagement_sessions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    teacher_id INT NOT NULL,
    class_id INT NOT NULL,
    subject VARCHAR(150) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    duration_seconds INT NOT NULL,
    started_at DATETIME NULL,
    scheduled_end_at DATETIME NULL,
    ended_at DATETIME NULL,
    actual_duration_seconds INT NOT NULL DEFAULT 0,
    emotion_counts LONGTEXT NOT NULL,
    total_face_observations INT NOT NULL DEFAULT 0,
    frame_samples INT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_engagement_teacher (teacher_id, created_at),
    KEY idx_engagement_class (class_id, created_at),
    CONSTRAINT fk_engagement_teacher FOREIGN KEY (teacher_id)
        REFERENCES teacher_users(id) ON DELETE CASCADE,
    CONSTRAINT fk_engagement_class FOREIGN KEY (class_id)
        REFERENCES classes(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS engagement_buckets (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    bucket_number INT NOT NULL,
    emotion_counts LONGTEXT NOT NULL,
    face_observations INT NOT NULL DEFAULT 0,
    UNIQUE KEY uq_engagement_bucket (session_id, bucket_number),
    CONSTRAINT fk_engagement_bucket_session FOREIGN KEY (session_id)
        REFERENCES engagement_sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB;
