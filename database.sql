CREATE DATABASE IF NOT EXISTS smart_attendance CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE smart_attendance;

CREATE TABLE IF NOT EXISTS teacher_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    class_name VARCHAR(100) NOT NULL DEFAULT 'My Class',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS classes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_id INT NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    department VARCHAR(100) NOT NULL,
    academic_year VARCHAR(30) NOT NULL,
    section VARCHAR(50) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_classes_teacher FOREIGN KEY (teacher_id)
        REFERENCES teacher_users(id) ON DELETE CASCADE,
    CONSTRAINT uq_teacher_class_details UNIQUE
        (teacher_id, class_name, department, academic_year, section)
);

CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    embedding LONGBLOB NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    teacher_id INT NOT NULL,
    CONSTRAINT fk_students_teacher FOREIGN KEY (teacher_id)
        REFERENCES teacher_users(id) ON DELETE CASCADE
);

-- Explicit class membership. Existing students are intentionally not assigned
-- to classes by this schema setup; memberships are added only by user action.
CREATE TABLE IF NOT EXISTS class_students (
    class_id INT NOT NULL,
    student_id INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (class_id, student_id),
    KEY idx_class_students_student (student_id),
    CONSTRAINT fk_class_students_class FOREIGN KEY (class_id)
        REFERENCES classes(id) ON DELETE CASCADE,
    CONSTRAINT fk_class_students_student FOREIGN KEY (student_id)
        REFERENCES students(id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS attendance_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher_id INT NOT NULL,
    class_id INT NOT NULL,
    session_date DATE NOT NULL,
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    duration_seconds INT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_sessions_class_date (class_id, session_date),
    CONSTRAINT fk_sessions_teacher FOREIGN KEY (teacher_id)
        REFERENCES teacher_users(id) ON DELETE CASCADE,
    CONSTRAINT fk_sessions_class FOREIGN KEY (class_id)
        REFERENCES classes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    class_id INT NULL,
    session_id INT NULL,
    attendance_date DATE NOT NULL,
    attendance_time TIME NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'Present',
    KEY idx_attendance_class_day (class_id, attendance_date),
    CONSTRAINT fk_attendance_student FOREIGN KEY (student_id)
        REFERENCES students(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_attendance_class FOREIGN KEY (class_id)
        REFERENCES classes(id) ON DELETE SET NULL,
    CONSTRAINT uq_student_class_day UNIQUE (student_id, class_id, attendance_date)
);

-- Teacher account is created automatically by app.py on first run:
-- username: teacher
-- password: teacher123
