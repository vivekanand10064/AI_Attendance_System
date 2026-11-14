import csv
from io import StringIO

import os
import urllib.request
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path

BASE = Path(__file__).resolve().parent

try:
    import cv2
except ModuleNotFoundError as exc:
    venv_python = BASE / ".venv" / "Scripts" / "python.exe"
    print(
        "Missing runtime dependencies. Use the project virtual environment first.\n"
        f"Run: {venv_python} -m pip install -r requirements.txt\n"
        f"Then start the app with: {venv_python} app.py\n"
        "Or activate the venv and run: .\\.venv\\Scripts\\activate"
    )
    raise SystemExit(1) from exc

import mysql.connector
import numpy as np
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

load_dotenv(BASE / ".env")

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "dev-secret-change-me"
)

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv(
        "MYSQL_DATABASE",
        "smart_attendance"
    ),
}

# Additive engagement module configuration. Existing attendance routes and
# database tables are not changed by this registration.
app.config["DB_CONFIG"] = DB_CONFIG
from engagement import engagement_bp, ensure_engagement_tables
app.register_blueprint(engagement_bp)


# ============================================================
# FACE RECOGNITION MODEL CONFIGURATION
# ============================================================

MODELS = BASE / "models"

YUNET = MODELS / (
    "face_detection_yunet_2023mar.onnx"
)

SFACE = MODELS / (
    "face_recognition_sface_2021dec.onnx"
)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)

# Cosine similarity threshold
MATCH_THRESHOLD = 0.363


# ============================================================
# DATABASE
# ============================================================

def get_db():
    """
    Create and return a MySQL database connection.
    """
    return mysql.connector.connect(**DB_CONFIG)


# ============================================================
# TEACHER ACCOUNT
# ============================================================

def ensure_teacher():
    """
    Creates the default teacher account if it doesn't exist.

    Username:
        teacher

    Password:
        teacher123

    The password is stored as a secure hash.
    """

    conn = get_db()
    cur = conn.cursor()

    # Existing installations predate teacher registration. Add the class
    # field and ownership column in place so their data remains available.
    cur.execute(
        """
        SELECT column_default
        FROM information_schema.columns
        WHERE table_schema=DATABASE()
          AND table_name='teacher_users'
          AND column_name='class_name'
        """
    )
    legacy_class_column = cur.fetchone()
    if legacy_class_column is None:
        cur.execute(
            "ALTER TABLE teacher_users "
            "ADD COLUMN class_name VARCHAR(100) NOT NULL DEFAULT 'My Class'"
        )
    elif legacy_class_column[0] != "My Class":
        # Keep legacy values but make the field optional for new registrations.
        cur.execute(
            "ALTER TABLE teacher_users MODIFY class_name "
            "VARCHAR(100) NOT NULL DEFAULT 'My Class'"
        )

    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema=DATABASE()
          AND table_name='students'
          AND column_name='teacher_id'
        """
    )
    if cur.fetchone()[0] == 0:
        cur.execute(
            "ALTER TABLE students ADD COLUMN teacher_id INT NULL"
        )

    cur.execute(
        """
        SELECT id
        FROM teacher_users
        WHERE username=%s
        """,
        ("teacher",)
    )

    existing = cur.fetchone()

    if not existing:
        password_hash = generate_password_hash(
            "teacher123"
        )

        cur.execute(
            """
            INSERT INTO teacher_users
                (username, password_hash)
            VALUES
                (%s, %s)
            """,
            (
                "teacher",
                password_hash
            )
        )

        conn.commit()

    cur.execute("SELECT id FROM teacher_users WHERE username=%s", ("teacher",))
    default_teacher_id = cur.fetchone()[0]
    cur.execute(
        "UPDATE students SET teacher_id=%s WHERE teacher_id IS NULL",
        (default_teacher_id,)
    )
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.table_constraints
        WHERE table_schema=DATABASE()
          AND table_name='students'
          AND constraint_name='fk_students_teacher'
          AND constraint_type='FOREIGN KEY'
        """
    )
    if cur.fetchone()[0] == 0:
        cur.execute("ALTER TABLE students ADD INDEX idx_students_teacher (teacher_id)")
        cur.execute(
            """
            ALTER TABLE students
            ADD CONSTRAINT fk_students_teacher
            FOREIGN KEY (teacher_id) REFERENCES teacher_users(id)
            ON DELETE CASCADE
            """
        )
    cur.execute(
        "ALTER TABLE students MODIFY teacher_id INT NOT NULL"
    )

    # Phase 1 adds teacher-owned class records without linking or moving
    # any existing students or attendance history.
    cur.execute(
        """
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
        )
        """
    )

    # Explicit memberships allow a student record (and its single face
    # embedding) to be shared across classes without duplicating attendance.
    # No existing student is assigned automatically.
    cur.execute(
        """
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
        )
        """
    )

    # Classroom sessions support start/pause/resume/finish and session reports.
    cur.execute(
        """CREATE TABLE IF NOT EXISTS attendance_sessions (
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
        )"""
    )

    # Keep legacy attendance rows class-unassigned. New class sessions point
    # to the authorized class; the nullable FK leaves historical rows intact.
    cur.execute(
        """SELECT COUNT(*) FROM information_schema.columns
           WHERE table_schema=DATABASE() AND table_name='attendance'
             AND column_name='class_id'"""
    )
    if cur.fetchone()[0] == 0:
        cur.execute("ALTER TABLE attendance ADD COLUMN class_id INT NULL AFTER student_id")

    cur.execute(
        """SELECT COUNT(*) FROM information_schema.columns
           WHERE table_schema=DATABASE() AND table_name='attendance'
             AND column_name='session_id'"""
    )
    if cur.fetchone()[0] == 0:
        cur.execute("ALTER TABLE attendance ADD COLUMN session_id INT NULL AFTER class_id")

    cur.execute(
        """SELECT COUNT(*) FROM information_schema.table_constraints
           WHERE table_schema=DATABASE() AND table_name='attendance'
             AND constraint_name='fk_attendance_class' AND constraint_type='FOREIGN KEY'"""
    )
    if cur.fetchone()[0] == 0:
        cur.execute(
            """ALTER TABLE attendance ADD CONSTRAINT fk_attendance_class
               FOREIGN KEY (class_id) REFERENCES classes(id) ON DELETE SET NULL"""
        )

    cur.execute(
        """SELECT COUNT(*) FROM information_schema.statistics
           WHERE table_schema=DATABASE() AND table_name='attendance'
             AND index_name='idx_attendance_student'"""
    )
    if cur.fetchone()[0] == 0:
        # The previous unique key also supported the student foreign key.
        cur.execute("ALTER TABLE attendance ADD INDEX idx_attendance_student (student_id)")

    cur.execute(
        """SELECT COUNT(*) FROM information_schema.statistics
           WHERE table_schema=DATABASE() AND table_name='attendance'
             AND index_name='uq_student_day'"""
    )
    if cur.fetchone()[0] > 0:
        cur.execute("ALTER TABLE attendance DROP INDEX uq_student_day")

    cur.execute(
        """SELECT COUNT(*) FROM information_schema.statistics
           WHERE table_schema=DATABASE() AND table_name='attendance'
             AND index_name='uq_student_class_day'"""
    )
    if cur.fetchone()[0] == 0:
        cur.execute(
            """ALTER TABLE attendance ADD UNIQUE KEY uq_student_class_day
               (student_id, class_id, attendance_date)"""
        )
    cur.execute(
        """SELECT COUNT(*) FROM information_schema.statistics
           WHERE table_schema=DATABASE() AND table_name='attendance'
             AND index_name='idx_attendance_class_day'"""
    )
    if cur.fetchone()[0] == 0:
        cur.execute("ALTER TABLE attendance ADD INDEX idx_attendance_class_day (class_id, attendance_date)")

    cur.close()
    conn.close()


# ============================================================
# DOWNLOAD FACE MODELS
# ============================================================

def download_models():
    """
    Download YuNet and SFace models if they don't exist.
    """

    MODELS.mkdir(exist_ok=True)

    models = [
        (YUNET, YUNET_URL),
        (SFACE, SFACE_URL)
    ]

    for path, url in models:

        # YuNet is a compact ~233 KB model; the former 1 MB cutoff made
        # the app try downloading it on every startup.
        minimum_size = 100_000 if path == YUNET else 1_000_000

        if (
            not path.exists()
            or path.stat().st_size < minimum_size
        ):
            print(
                f"Downloading {path.name}..."
            )

            urllib.request.urlretrieve(
                url,
                path
            )


# ============================================================
# FACE DETECTOR
# ============================================================

def detector():

    return cv2.FaceDetectorYN.create(
        str(YUNET),
        "",
        (320, 320),
        0.80,
        0.30,
        5000
    )


# ============================================================
# FACE RECOGNIZER
# ============================================================

def recognizer():

    return cv2.FaceRecognizerSF.create(
        str(SFACE),
        ""
    )


# ============================================================
# FACE DETECTION
# ============================================================

def detect(det, frame):

    det.setInputSize(
        (
            frame.shape[1],
            frame.shape[0]
        )
    )

    _, faces = det.detect(frame)

    if faces is None:
        return np.empty(
            (0, 15),
            dtype=np.float32
        )

    return faces


# ============================================================
# CREATE FACE EMBEDDING
# ============================================================

def feature(rec, frame, face):

    aligned = rec.alignCrop(
        frame,
        face
    )

    return rec.feature(
        aligned
    ).astype(np.float32)


# ============================================================
# TEACHER AUTHORIZATION DECORATOR
# ============================================================

def teacher_required(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):

        if not session.get("teacher_id"):
            return redirect(
                url_for("teacher_login")
            )

        return fn(*args, **kwargs)

    return wrapped


def get_teacher_class_name():
    if session.get("class_name"):
        return session["class_name"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT class_name FROM teacher_users WHERE id=%s",
        (session.get("teacher_id"),)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else "My Class"


# ============================================================
# GET ALL STUDENTS
# ============================================================

def all_students():

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    teacher_id = session.get("teacher_id")
    if teacher_id:
        cur.execute(
            """
        SELECT
            id,
            student_id,
            name,
            embedding,
            created_at
        FROM students
        WHERE teacher_id=%s
        ORDER BY name
            """,
            (teacher_id,)
        )
    else:
        cur.execute(
            """
            SELECT id, student_id, name, embedding, created_at
            FROM students
            ORDER BY name
            """
        )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


# ============================================================
# GET ATTENDANCE FOR A PARTICULAR DAY
# ============================================================

def attendance_for_day(day, class_id=None):

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    if class_id is not None:
        cur.execute(
            """SELECT s.id, s.student_id, s.name,
                      COALESCE(a.status,'Absent') AS status,
                      a.attendance_time, c.class_name
               FROM class_students cs
               JOIN students s ON s.id=cs.student_id
               JOIN classes c ON c.id=cs.class_id
               LEFT JOIN attendance a
                 ON a.student_id=s.id AND a.class_id=cs.class_id
                AND a.attendance_date=%s
               WHERE cs.class_id=%s AND s.teacher_id=%s AND c.teacher_id=%s
               ORDER BY s.name""",
            (day, class_id, session["teacher_id"], session["teacher_id"]),
        )
    else:
        # A teacher-wide view is one row per explicit class membership. Legacy
        # students without a class keep their historical null-class records.
        cur.execute(
        """
        SELECT
            s.id,
            s.student_id,
            s.name,
            c.class_name,
            COALESCE(
                a.status,
                'Absent'
            ) AS status,
            a.attendance_time

        FROM students s
        LEFT JOIN class_students cs ON cs.student_id=s.id
        LEFT JOIN classes c ON c.id=cs.class_id AND c.teacher_id=%s
        LEFT JOIN attendance a
            ON a.student_id = s.id AND a.attendance_date = %s
           AND a.class_id <=> cs.class_id
        WHERE s.teacher_id=%s
          AND (cs.class_id IS NULL OR c.id IS NOT NULL)
        UNION ALL
        SELECT s.id,s.student_id,s.name,'Unassigned' AS class_name,
               a.status,a.attendance_time
        FROM attendance a JOIN students s ON s.id=a.student_id
        WHERE s.teacher_id=%s AND a.attendance_date=%s AND a.class_id IS NULL
          AND EXISTS (SELECT 1 FROM class_students cs2 JOIN classes c2 ON c2.id=cs2.class_id
                      WHERE cs2.student_id=s.id AND c2.teacher_id=%s)
        ORDER BY name,class_name
        """,
        (session["teacher_id"], day, session["teacher_id"],day,
         session["teacher_id"],session["teacher_id"]))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


# ============================================================
# PARSE FLOAT
# ============================================================

def parse_float(value):
    """Return a float or None for an empty/invalid value."""
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_percentage_filter(value, field_name):
    """
    Validate an attendance percentage filter.

    Returns:
        (number, error_message)
    """
    raw = (value or "").strip()

    if raw == "":
        return None, None

    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None, f"{field_name} must be a valid number."

    if not np.isfinite(number):
        return None, f"{field_name} must be a valid number."

    if number < 0 or number > 100:
        return None, f"{field_name} must be between 0 and 100."

    return number, None


# ============================================================
# STUDENT ATTENDANCE STATISTICS
# ============================================================

def attendance_stats_for_student(
    student_id,
    month=None
):

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    if month:

        cur.execute(
            """
            SELECT
                COUNT(
                    DISTINCT attendance_date
                ) AS present_days,

                MIN(attendance_date)
                    AS first_date,

                MAX(attendance_date)
                    AS last_date

            FROM attendance

            WHERE student_id=%s

            AND attendance_date >= %s

            AND attendance_date <
                DATE_ADD(
                    %s,
                    INTERVAL 1 MONTH
                )
            """,
            (
                student_id,
                month,
                month
            )
        )

    else:

        cur.execute(
            """
            SELECT

                COUNT(
                    DISTINCT attendance_date
                ) AS present_days,

                MIN(attendance_date)
                    AS first_date,

                MAX(attendance_date)
                    AS last_date

            FROM attendance

            WHERE student_id=%s
            """,
            (student_id,)
        )

    result = (
        cur.fetchone()
        or {}
    )

    cur.close()
    conn.close()

    return result


# ============================================================
# MONTHLY STUDENT STATISTICS
# ============================================================

def monthly_student_stats(
    student_id,
    months=12,
    class_id=None,
):
    """
    Return the student's attendance percentage for the last N months.

    The denominator is the number of distinct attendance dates recorded
    for the whole class in each month, not the number of calendar days.
    This keeps the percentage consistent with the Analytics page.
    """

    current_month = date.today().replace(day=1)

    month_starts = []
    year = current_month.year
    month = current_month.month

    for _ in range(months):
        month_starts.append(date(year, month, 1))
        month -= 1

        if month == 0:
            month = 12
            year -= 1

    start = month_starts[-1]

    if current_month.month == 12:
        end = date(current_month.year + 1, 1, 1)
    else:
        end = date(
            current_month.year,
            current_month.month + 1,
            1
        )

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    # Student's present days by month.
    if class_id is None:
        cur.execute(
        """
        SELECT
            DATE_FORMAT(
                attendance_date,
                '%%Y-%%m'
            ) AS month,
            COUNT(DISTINCT attendance_date) AS present_days
        FROM attendance
        WHERE student_id=%s
          AND attendance_date >= %s
          AND attendance_date < %s
          AND status='Present'
        GROUP BY DATE_FORMAT(
            attendance_date,
            '%%Y-%%m'
        )
        """,
        (student_id, start, end)
        )
    else:
        cur.execute(
            """SELECT DATE_FORMAT(a.attendance_date,'%%Y-%%m') AS month,
                      COUNT(DISTINCT a.attendance_date) AS present_days
               FROM attendance a JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
               JOIN students s ON s.id=a.student_id
               WHERE a.student_id=%s AND a.class_id=%s AND s.teacher_id=%s
                 AND a.attendance_date >= %s AND a.attendance_date < %s AND a.status='Present'
               GROUP BY DATE_FORMAT(a.attendance_date,'%%Y-%%m')""",
            (student_id, class_id, session["teacher_id"], start, end),
        )
    student_rows = cur.fetchall()

    # Number of actual attendance-recorded/class days by month.
    if class_id is None:
        cur.execute(
        """
        SELECT
            DATE_FORMAT(
                attendance_date,
                '%%Y-%%m'
            ) AS month,
            COUNT(DISTINCT attendance_date) AS recorded_days
        FROM attendance
        JOIN students s ON s.id=attendance.student_id
        WHERE s.teacher_id=%s
          AND attendance_date >= %s
          AND attendance_date < %s
        GROUP BY DATE_FORMAT(
            attendance_date,
            '%%Y-%%m'
        )
        """,
        (session["teacher_id"], start, end)
        )
    else:
        cur.execute(
            """SELECT DATE_FORMAT(a.attendance_date,'%%Y-%%m') AS month,
                      COUNT(DISTINCT a.attendance_date) AS recorded_days
               FROM attendance a JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
               JOIN students s ON s.id=a.student_id
               WHERE a.class_id=%s AND s.teacher_id=%s AND a.attendance_date >= %s AND a.attendance_date < %s
               GROUP BY DATE_FORMAT(a.attendance_date,'%%Y-%%m')""",
            (class_id, session["teacher_id"], start, end),
        )
    recorded_rows = cur.fetchall()

    cur.close()
    conn.close()

    present_by_month = {
        row["month"]: int(row["present_days"] or 0)
        for row in student_rows
    }

    recorded_by_month = {
        row["month"]: int(row["recorded_days"] or 0)
        for row in recorded_rows
    }

    output = []

    # Oldest -> newest for the chart.
    for month_start in reversed(month_starts):
        key = month_start.strftime("%Y-%m")
        present = present_by_month.get(key, 0)
        recorded_days = recorded_by_month.get(key, 0)

        percentage = (
            round(present * 100 / recorded_days, 1)
            if recorded_days
            else None
        )

        output.append({
            "month": key,
            "present_days": present,
            "recorded_days": recorded_days,
            "percentage": percentage
        })

    return output


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return redirect(
        url_for(
            "student_attendance"
        )
    )


# ============================================================
# STUDENT ATTENDANCE PAGE
# ============================================================

@app.get("/attendance")
@teacher_required
def student_attendance():
    class_id_raw = request.args.get("class_id")
    class_id = request.args.get("class_id", type=int)
    if class_id_raw and class_id is None:
        return "Invalid class", 400

    conn = get_db()
    cur = conn.cursor(dictionary=True)
    if class_id is not None:
        class_record = owned_class(class_id, session["teacher_id"], cur)
        cur.close(); conn.close()
        if not class_record:
            return "Class not found", 404
        return render_template("student_attendance.html", class_record=class_record)

    cur.execute(
        """SELECT id, class_name, department, academic_year, section
           FROM classes WHERE teacher_id=%s
           ORDER BY class_name, academic_year DESC, section""",
        (session["teacher_id"],),
    )
    classes = cur.fetchall()
    cur.close(); conn.close()
    return render_template("attendance_class_select.html", classes=classes)


@app.get("/classes/<int:class_id>/attendance")
@teacher_required
def class_attendance(class_id):
    conn = get_db(); cur = conn.cursor(dictionary=True)
    class_record = owned_class(class_id, session["teacher_id"], cur)
    cur.close(); conn.close()
    if not class_record:
        return "Class not found", 404
    return render_template("student_attendance.html", class_record=class_record)


# ============================================================
# CLASSROOM SESSION MANAGEMENT
# ============================================================

@app.get("/api/classes/<int:class_id>/session")
@teacher_required
def get_attendance_session(class_id):
    conn = get_db(); cur = conn.cursor(dictionary=True)
    class_record = owned_class(class_id, session["teacher_id"], cur)
    if not class_record:
        cur.close(); conn.close()
        return jsonify(ok=False, message="Class not found."), 404
    cur.execute(
        "SELECT id, status, started_at, ended_at, duration_seconds FROM attendance_sessions "
        "WHERE class_id=%s AND teacher_id=%s AND session_date=%s "
        "ORDER BY id DESC LIMIT 1",
        (class_id, session["teacher_id"], date.today()),
    )
    current = cur.fetchone()
    cur.execute(
        "SELECT COUNT(*) AS total FROM class_students WHERE class_id=%s", (class_id,)
    )
    total = cur.fetchone()["total"]
    cur.execute(
        "SELECT COUNT(*) AS present FROM attendance WHERE class_id=%s AND attendance_date=%s AND status='Present'",
        (class_id, date.today()),
    )
    present = cur.fetchone()["present"]
    cur.close(); conn.close()
    if current:
        for key in ("started_at", "ended_at"):
            if current.get(key):
                current[key] = current[key].isoformat(sep=" ")
    return jsonify(ok=True, session=current, total_students=total, present_count=present,
                   absent_count=max(total-present, 0))


@app.post("/api/classes/<int:class_id>/session/start")
@teacher_required
def start_attendance_session(class_id):
    conn = get_db(); cur = conn.cursor(dictionary=True)
    if not owned_class(class_id, session["teacher_id"], cur):
        cur.close(); conn.close()
        return jsonify(ok=False, message="Class not found."), 404
    cur.execute(
        "SELECT id, status FROM attendance_sessions WHERE class_id=%s AND teacher_id=%s "
        "AND session_date=%s ORDER BY id DESC LIMIT 1",
        (class_id, session["teacher_id"], date.today()),
    )
    existing = cur.fetchone()
    if existing and existing["status"] == "finished":
        cur.close(); conn.close()
        return jsonify(ok=False, message="This class already has a finished attendance session today."), 409
    if existing:
        if existing["status"] == "paused":
            cur.execute("UPDATE attendance_sessions SET status='active',started_at=NOW() WHERE id=%s", (existing["id"],))
            conn.commit()
            result = {"id": existing["id"], "status": "active"}
        else:
            result = existing
    else:
        cur.execute(
            "INSERT INTO attendance_sessions (teacher_id,class_id,session_date,status) VALUES (%s,%s,%s,'active')",
            (session["teacher_id"], class_id, date.today()),
        )
        conn.commit()
        result = {"id": cur.lastrowid, "status": "active"}
    cur.close(); conn.close()
    return jsonify(ok=True, session=result)


@app.post("/api/classes/<int:class_id>/session/pause")
@teacher_required
def pause_attendance_session(class_id):
    return set_attendance_session_state(class_id, "paused")


@app.post("/api/classes/<int:class_id>/session/resume")
@teacher_required
def resume_attendance_session(class_id):
    return set_attendance_session_state(class_id, "active")


def set_attendance_session_state(class_id, new_state):
    conn = get_db(); cur = conn.cursor(dictionary=True)
    if not owned_class(class_id, session["teacher_id"], cur):
        cur.close(); conn.close()
        return jsonify(ok=False, message="Class not found."), 404
    if new_state == "paused":
        cur.execute(
            "UPDATE attendance_sessions SET status='paused',duration_seconds=duration_seconds+"
            "GREATEST(TIMESTAMPDIFF(SECOND,started_at,NOW()),0) "
            "WHERE class_id=%s AND teacher_id=%s AND session_date=%s AND status='active' "
            "ORDER BY id DESC LIMIT 1",
            (class_id, session["teacher_id"], date.today()),
        )
    else:
        cur.execute(
            "UPDATE attendance_sessions SET status='active',started_at=NOW() "
            "WHERE class_id=%s AND teacher_id=%s AND session_date=%s AND status='paused' "
            "ORDER BY id DESC LIMIT 1",
            (class_id, session["teacher_id"], date.today()),
        )
    changed = cur.rowcount
    conn.commit(); cur.close(); conn.close()
    if not changed:
        return jsonify(ok=False, message="No open attendance session found."), 409
    return jsonify(ok=True, status=new_state)


@app.post("/api/classes/<int:class_id>/session/finish")
@teacher_required
def finish_attendance_session(class_id):
    conn = get_db(); cur = conn.cursor(dictionary=True)
    if not owned_class(class_id, session["teacher_id"], cur):
        cur.close(); conn.close()
        return jsonify(ok=False, message="Class not found."), 404
    cur.execute(
        "SELECT id, started_at, duration_seconds, status FROM attendance_sessions "
        "WHERE class_id=%s AND teacher_id=%s AND session_date=%s "
        "AND status IN ('active','paused') ORDER BY id DESC LIMIT 1",
        (class_id, session["teacher_id"], date.today()),
    )
    active = cur.fetchone()
    if not active:
        cur.close(); conn.close()
        return jsonify(ok=False, message="There is no active session to finish."), 409
    now = datetime.now()
    started = active["started_at"]
    duration = int(active.get("duration_seconds", 0))
    if active.get("status") == "active":
        duration += max(int((now - started).total_seconds()), 0)
    today = date.today(); now_time = now.time().replace(microsecond=0)
    try:
        # Create explicit Absent rows for roster members not marked Present today.
        cur.execute(
            """INSERT IGNORE INTO attendance
               (student_id,class_id,session_id,attendance_date,attendance_time,status)
               SELECT cs.student_id,cs.class_id,%s,%s,%s,'Absent'
               FROM class_students cs
               WHERE cs.class_id=%s""",
            (active["id"], today, now_time, class_id),
        )
        cur.execute(
            "UPDATE attendance_sessions SET status='finished',ended_at=%s,duration_seconds=%s WHERE id=%s",
            (now, duration, active["id"]),
        )
        conn.commit()
    except Exception:
        conn.rollback(); app.logger.exception("Could not finish attendance session")
        cur.close(); conn.close()
        return jsonify(ok=False, message="Could not finalize attendance."), 500
    cur.execute(
        "SELECT status,COUNT(*) AS n FROM attendance WHERE class_id=%s AND attendance_date=%s GROUP BY status",
        (class_id, today),
    )
    counts = {r["status"]: r["n"] for r in cur.fetchall()}
    cur.close(); conn.close()
    return jsonify(ok=True, status="finished", duration_seconds=duration,
                   present_count=counts.get("Present", 0), absent_count=counts.get("Absent", 0))


@app.get("/api/classes/<int:class_id>/session/report")
@teacher_required
def attendance_session_report(class_id):
    conn = get_db(); cur = conn.cursor(dictionary=True)
    if not owned_class(class_id, session["teacher_id"], cur):
        cur.close(); conn.close()
        return jsonify(ok=False, message="Class not found."), 404
    cur.execute(
        """SELECT s.id AS student_pk,s.student_id,s.name,COALESCE(a.status,'Absent') AS status,
                  a.attendance_time
           FROM class_students cs JOIN students s ON s.id=cs.student_id
           LEFT JOIN attendance a ON a.student_id=s.id AND a.class_id=cs.class_id
                AND a.attendance_date=%s
           WHERE cs.class_id=%s AND s.teacher_id=%s ORDER BY s.name""",
        (date.today(), class_id, session["teacher_id"]),
    )
    rows = cur.fetchall(); cur.close(); conn.close()
    return jsonify(ok=True, students=[{
        "student_pk": r["student_pk"], "uid": r["student_id"], "name": r["name"],
        "status": r["status"], "attendance_time": str(r["attendance_time"]) if r["attendance_time"] else None
    } for r in rows])


@app.post("/api/classes/<int:class_id>/attendance/manual")
@teacher_required
def manual_attendance_update(class_id):
    payload = request.get_json(silent=True) or {}
    student_pk = payload.get("student_id")
    status = payload.get("status")
    if not isinstance(student_pk, int) or status not in ("Present", "Absent"):
        return jsonify(ok=False, message="Valid student_id and Present/Absent status are required."), 400
    conn = get_db(); cur = conn.cursor(dictionary=True)
    if not owned_class(class_id, session["teacher_id"], cur):
        cur.close(); conn.close()
        return jsonify(ok=False, message="Class not found."), 404
    cur.execute(
        "SELECT s.id FROM students s JOIN class_students cs ON cs.student_id=s.id "
        "WHERE s.id=%s AND s.teacher_id=%s AND cs.class_id=%s",
        (student_pk, session["teacher_id"], class_id),
    )
    if not cur.fetchone():
        cur.close(); conn.close()
        return jsonify(ok=False, message="Student is not in this class."), 404
    cur.execute(
        "SELECT id FROM attendance_sessions WHERE class_id=%s AND teacher_id=%s AND session_date=%s "
        "ORDER BY id DESC LIMIT 1", (class_id, session["teacher_id"], date.today())
    )
    session_row = cur.fetchone()
    cur.execute(
        "INSERT INTO attendance (student_id,class_id,session_id,attendance_date,attendance_time,status) "
        "VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE status=VALUES(status),attendance_time=VALUES(attendance_time),session_id=VALUES(session_id)",
        (student_pk, class_id, session_row["id"] if session_row else None,
         date.today(), datetime.now().time().replace(microsecond=0), status),
    )
    conn.commit(); cur.close(); conn.close()
    return jsonify(ok=True, message=f"Attendance updated to {status}.")


# ============================================================
# FACE RECOGNITION + ATTENDANCE API
# ============================================================

@app.post("/api/recognize")
def api_recognize():
    """Recognize every visible face in one camera frame and mark attendance."""
    class_id_raw = request.form.get("class_id")
    class_id = request.form.get("class_id", type=int)

    if class_id_raw and class_id is None:
        return jsonify(ok=False, state="error", message="Invalid class."), 400
    if not session.get("teacher_id"):
        return jsonify(ok=False, state="error", message="Teacher login is required."), 401
    if class_id is None:
        return jsonify(ok=False, state="error", message="Select a class before marking attendance."), 400

    teacher_id = session["teacher_id"]
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    class_record = owned_class(class_id, teacher_id, cur)
    cur.close()
    conn.close()
    if not class_record:
        return jsonify(ok=False, state="error", message="Class not found."), 404

    # Attendance is only recorded while a teacher-controlled session is active.
    conn = get_db(); cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id FROM attendance_sessions WHERE class_id=%s AND teacher_id=%s "
        "AND session_date=%s AND status='active' ORDER BY id DESC LIMIT 1",
        (class_id, teacher_id, date.today()),
    )
    active_session = cur.fetchone()
    cur.close(); conn.close()
    if not active_session:
        return jsonify(ok=False, state="session_inactive",
                       message="Start or resume the classroom attendance session before scanning."), 409

    uploaded_frame = request.files.get("frame")
    if not uploaded_frame:
        return jsonify(ok=False, state="error", message="No camera frame received.", faces=[]), 400

    frame = cv2.imdecode(np.frombuffer(uploaded_frame.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(ok=False, state="error", message="Invalid camera frame.", faces=[]), 400

    det = detector()
    rec = recognizer()
    detected_faces = detect(det, frame)

    if len(detected_faces) == 0:
        return jsonify(
            ok=True, state="no_face",
            message="No faces detected. Make sure the classroom is visible to the camera.",
            faces=[], marked_count=0, already_count=0, unknown_count=0
        )

    students_data = students_in_class(class_id, teacher_id)
    if not students_data:
        return jsonify(
            ok=True, state="no_students",
            message="No students are registered in this class.",
            faces=[], marked_count=0, already_count=0, unknown_count=len(detected_faces)
        )

    today = date.today()
    now_time = datetime.now().time().replace(microsecond=0)
    results = []
    matched_student_ids = set()

    # A single database connection is reused for all faces in this frame.
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    try:
        for face in detected_faces:
            x, y, w, h = map(int, face[:4])
            result = {"x": x, "y": y, "w": w, "h": h}

            try:
                embedding = feature(rec, frame, face)
            except Exception:
                app.logger.exception("Could not create an embedding for a detected face")
                result.update(state="unknown", name="Unclear face", message="Face was too small or unclear.")
                results.append(result)
                continue

            best_student = None
            best_score = -1.0

            for student in students_data:
                try:
                    known = np.frombuffer(student["embedding"], dtype=np.float32).reshape(1, -1)
                    score = float(rec.match(embedding, known, cv2.FaceRecognizerSF_FR_COSINE))
                except (TypeError, ValueError, cv2.error):
                    continue

                if score > best_score:
                    best_student = student
                    best_score = score

            if best_student is None or best_score < MATCH_THRESHOLD:
                result.update(
                    state="unknown", name="Unknown Student",
                    confidence=round(max(best_score, 0) * 100, 1),
                    message="Not recognized; attendance was not marked."
                )
                results.append(result)
                continue

            student_pk = best_student["id"]
            if student_pk in matched_student_ids:
                result.update(
                    state="duplicate_face", name=best_student["name"],
                    uid=best_student["student_id"],
                    confidence=round(best_score * 100, 1),
                    message="This student was recognized more than once in this frame."
                )
                results.append(result)
                continue

            matched_student_ids.add(student_pk)

            # INSERT IGNORE relies on the database unique key
            # (student_id, class_id, attendance_date), preventing duplicates
            # even if two camera requests arrive at nearly the same time.
            cur.execute(
                """INSERT IGNORE INTO attendance
                   (student_id, class_id, session_id, attendance_date, attendance_time, status)
                   VALUES (%s, %s, %s, %s, %s, 'Present')""",
                (student_pk, class_id, active_session["id"], today, now_time)
            )

            if cur.rowcount == 1:
                result.update(
                    state="marked", name=best_student["name"],
                    uid=best_student["student_id"],
                    confidence=round(best_score * 100, 1),
                    message="Attendance marked."
                )
            else:
                cur.execute(
                    """SELECT attendance_time,status FROM attendance
                       WHERE student_id=%s AND class_id=%s AND attendance_date=%s""",
                    (student_pk, class_id, today)
                )
                existing = cur.fetchone()
                if existing and existing.get("status") == "Absent":
                    cur.execute(
                        "UPDATE attendance SET status='Present',session_id=%s,attendance_time=%s WHERE student_id=%s AND class_id=%s AND attendance_date=%s",
                        (active_session["id"], now_time, student_pk, class_id, today),
                    )
                    result.update(
                        state="marked", name=best_student["name"], uid=best_student["student_id"],
                        confidence=round(best_score * 100, 1), message="Attendance corrected to Present."
                    )
                else:
                    result.update(
                        state="already", name=best_student["name"],
                        uid=best_student["student_id"],
                        confidence=round(best_score * 100, 1),
                        attendance_time=str(existing["attendance_time"]) if existing else None,
                        message="Attendance was already marked today."
                    )
            results.append(result)

        conn.commit()
    except Exception:
        conn.rollback()
        app.logger.exception("Bulk face attendance processing failed")
        return jsonify(ok=False, state="error", message="Could not save attendance. Please retry.", faces=results), 500
    finally:
        cur.close()
        conn.close()

    marked_count = sum(1 for item in results if item["state"] == "marked")
    already_count = sum(1 for item in results if item["state"] == "already")
    unknown_count = sum(1 for item in results if item["state"] in ("unknown", "duplicate_face"))
    recognized_count = marked_count + already_count

    return jsonify(
        ok=True,
        state="processed",
        message=(
            f"Detected {len(detected_faces)} face(s): "
            f"{marked_count} newly marked, {already_count} already present, "
            f"{unknown_count} not matched."
        ),
        faces=results,
        detected_count=len(detected_faces),
        recognized_count=recognized_count,
        marked_count=marked_count,
        already_count=already_count,
        unknown_count=unknown_count
    )


# ============================================================
# TEACHER LOGIN
# ============================================================

@app.route(
    "/teacher/login",
    methods=["GET", "POST"]
)
def teacher_login():

    if request.method == "POST":

        username = (
            request.form
            .get(
                "username",
                ""
            )
            .strip()
        )

        password = (
            request.form
            .get(
                "password",
                ""
            )
        )

        conn = get_db()

        cur = conn.cursor(
            dictionary=True
        )

        cur.execute(
            """
            SELECT
                id,
                username,
                password_hash,
                class_name

            FROM teacher_users

            WHERE username=%s
            """,
            (username,)
        )

        teacher = cur.fetchone()

        cur.close()
        conn.close()

        if (
            teacher
            and
            check_password_hash(
                teacher["password_hash"],
                password
            )
        ):

            session.clear()

            session["teacher_id"] = (
                teacher["id"]
            )

            session["teacher_username"] = (
                teacher["username"]
            )
            session["class_name"] = teacher["class_name"]

            return redirect(url_for("assigned_classes"))

        flash(
            "Invalid username or password.",
            "error"
        )

    return render_template(
        "login.html"
    )


@app.route("/teacher/register", methods=["GET", "POST"])
def teacher_register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password or not confirm_password:
            flash("Username, password and confirmation are required.", "error")
        elif len(username) > 50:
            flash("Username must be 50 characters or fewer.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            conn = get_db()
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO teacher_users (username, password_hash)
                    VALUES (%s, %s)
                    """,
                    (username, generate_password_hash(password))
                )
                conn.commit()
                flash("Account created. Please log in.", "success")
                return redirect(url_for("teacher_login"))
            except mysql.connector.IntegrityError:
                conn.rollback()
                flash("That username is already taken.", "error")
            finally:
                cur.close()
                conn.close()

    return render_template("teacher_register.html")


@app.get("/classes")
@teacher_required
def assigned_classes():
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT id, class_name, department, academic_year, section, created_at
        FROM classes
        WHERE teacher_id=%s
        ORDER BY class_name, academic_year DESC, section
        """,
        (session["teacher_id"],)
    )
    classes = cur.fetchall()
    cur.close()
    conn.close()
    return render_template("classes_dashboard.html", classes=classes)


@app.route("/classes/add", methods=["GET", "POST"])
@teacher_required
def add_class():
    if request.method == "POST":
        class_name = request.form.get("class_name", "").strip()
        department = request.form.get("department", "").strip()
        academic_year = request.form.get("academic_year", "").strip()
        section = request.form.get("section", "").strip()
        values = (class_name, department, academic_year, section)

        if not all(values):
            flash("All class details are required.", "error")
        elif len(class_name) > 100 or len(department) > 100 or len(academic_year) > 30 or len(section) > 50:
            flash("One or more class details exceed the allowed length.", "error")
        else:
            conn = get_db()
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO classes
                        (teacher_id, class_name, department, academic_year, section)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session["teacher_id"], *values)
                )
                conn.commit()
                flash("Class added successfully.", "success")
                return redirect(url_for("assigned_classes"))
            except mysql.connector.IntegrityError:
                conn.rollback()
                flash("This class already exists in your account.", "error")
            finally:
                cur.close()
                conn.close()

    return render_template("add_class.html")


@app.get("/classes/<int:class_id>")
@teacher_required
def open_class(class_id):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT id, class_name, department, academic_year, section
        FROM classes
        WHERE id=%s AND teacher_id=%s
        """,
        (class_id, session["teacher_id"])
    )
    class_record = cur.fetchone()
    cur.close()
    conn.close()
    if not class_record:
        return "Class not found", 404
    return render_template("class_detail.html", class_record=class_record)


def owned_class(class_id, teacher_id, cur):
    """Return a class only when it belongs to the signed-in teacher."""
    cur.execute(
        "SELECT id, class_name, department, academic_year, section "
        "FROM classes WHERE id=%s AND teacher_id=%s",
        (class_id, teacher_id),
    )
    return cur.fetchone()


# ============================================================
# DELETE CLASS
# ============================================================

@app.post("/classes/<int:class_id>/delete")
@teacher_required
def delete_class(class_id):

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    try:
        # Verify that the class belongs to the logged-in teacher.
        class_record = owned_class(
            class_id,
            session["teacher_id"],
            cur
        )

        if not class_record:
            cur.close()
            conn.close()
            return "Class not found or unauthorized.", 404

        class_name = class_record["class_name"]

        # Delete only the selected teacher-owned class.
        # Database foreign keys handle related records:
        # class_students -> CASCADE
        # attendance_sessions -> CASCADE
        # attendance.class_id -> SET NULL
        cur.execute(
            """
            DELETE FROM classes
            WHERE id=%s AND teacher_id=%s
            """,
            (class_id, session["teacher_id"])
        )

        if cur.rowcount == 0:
            conn.rollback()
            flash("Class could not be deleted.", "error")
        else:
            conn.commit()
            flash(
                f"Class '{class_name}' deleted successfully.",
                "success"
            )

    except Exception:
        conn.rollback()
        app.logger.exception("Class deletion failed")
        flash("Unable to delete class. Please try again.", "error")

    finally:
        cur.close()
        conn.close()

    return redirect(url_for("assigned_classes"))

def students_in_class(class_id, teacher_id):
    """Fetch recognition records for a class owned by this teacher."""
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT s.id, s.student_id, s.name, s.embedding, s.created_at
           FROM class_students cs JOIN students s ON s.id=cs.student_id
           JOIN classes c ON c.id=cs.class_id
           WHERE cs.class_id=%s AND s.teacher_id=%s AND c.teacher_id=%s
           ORDER BY s.name""",
        (class_id, teacher_id, teacher_id),
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows


# ============================================================
# TEACHER LOGOUT
# ============================================================

@app.get("/teacher/logout")
def teacher_logout():

    session.clear()

    return redirect(
        url_for(
            "teacher_login"
        )
    )


# ============================================================
# TEACHER DASHBOARD
# ============================================================

@app.get("/dashboard")
@teacher_required
def dashboard():
    today = date.today()
    class_id = request.args.get("class_id", type=int)
    conn = get_db(); cur = conn.cursor(dictionary=True)
    class_record = None
    if class_id is not None:
        class_record = owned_class(class_id, session["teacher_id"], cur)
        if not class_record:
            cur.close(); conn.close()
            return "Class not found", 404
        cur.execute("SELECT COUNT(*) AS n FROM class_students cs JOIN students s ON s.id=cs.student_id WHERE cs.class_id=%s AND s.teacher_id=%s",
                    (class_id, session["teacher_id"]))
        total_students = int(cur.fetchone()["n"] or 0)
        cur.execute("""SELECT COUNT(DISTINCT a.student_id) AS n FROM attendance a
                       JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                       JOIN students s ON s.id=a.student_id
                       WHERE a.class_id=%s AND s.teacher_id=%s AND a.attendance_date=%s AND a.status='Present'""",
                    (class_id,session["teacher_id"],today))
        present_today = int(cur.fetchone()["n"] or 0)
        cur.execute("SELECT COUNT(*) AS n FROM attendance WHERE class_id=%s AND attendance_date=%s",(class_id,today))
        today_has_records=bool(cur.fetchone()["n"])
        cur.execute("""SELECT COUNT(DISTINCT a.attendance_date) AS n FROM attendance a
                       JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                       JOIN students s ON s.id=a.student_id
                       WHERE a.class_id=%s AND s.teacher_id=%s""",
                    (class_id,session["teacher_id"]))
        total_recorded_days = int(cur.fetchone()["n"] or 0)
        cur.execute("""SELECT a.attendance_date,COUNT(DISTINCT CASE WHEN a.status='Present' THEN a.student_id END) AS present
                       FROM attendance a JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                       JOIN students s ON s.id=a.student_id
                       WHERE a.class_id=%s AND s.teacher_id=%s AND a.attendance_date BETWEEN %s AND %s
                         GROUP BY a.attendance_date""",
                    (class_id,session["teacher_id"],today-timedelta(days=6),today))
        trend_rows = cur.fetchall()
        cur.execute("""SELECT s.id,s.student_id,s.name,
                          COUNT(DISTINCT CASE WHEN a.status='Present' THEN a.attendance_date END) AS present_days,
                          MAX(a.attendance_date) AS last_attendance
                       FROM class_students cs JOIN students s ON s.id=cs.student_id
                       LEFT JOIN attendance a ON a.student_id=s.id AND a.class_id=cs.class_id
                       WHERE cs.class_id=%s AND s.teacher_id=%s
                       GROUP BY s.id,s.student_id,s.name ORDER BY s.name""",
                    (class_id, session["teacher_id"]))
        student_stats = cur.fetchall()
        classes = []
    else:
        cur.execute("""SELECT c.id,c.class_name,c.department,c.academic_year,c.section,
                          COUNT(DISTINCT s.id) AS student_count
                       FROM classes c LEFT JOIN class_students cs ON cs.class_id=c.id
                       LEFT JOIN students s ON s.id=cs.student_id AND s.teacher_id=c.teacher_id
                       WHERE c.teacher_id=%s
                       GROUP BY c.id,c.class_name,c.department,c.academic_year,c.section
                       ORDER BY c.class_name,c.section""", (session["teacher_id"],))
        classes = cur.fetchall()
        total_students = len({r["student_id"] for r in students_in_teacher_classes(session["teacher_id"])})
        cur.execute("""SELECT COUNT(DISTINCT a.student_id) AS n FROM attendance a
                       JOIN classes c ON c.id=a.class_id
                       JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                       JOIN students s ON s.id=a.student_id
                       WHERE c.teacher_id=%s AND s.teacher_id=%s AND a.attendance_date=%s AND a.status='Present'""",
                    (session["teacher_id"],session["teacher_id"],today))
        present_today = int(cur.fetchone()["n"] or 0)
        cur.execute("""SELECT COUNT(*) AS n FROM attendance a JOIN classes c ON c.id=a.class_id
                       JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                       WHERE c.teacher_id=%s AND a.attendance_date=%s""",
                    (session["teacher_id"],today))
        today_has_records=bool(cur.fetchone()["n"])
        cur.execute("""SELECT COUNT(DISTINCT a.attendance_date) AS n FROM attendance a
                       JOIN classes c ON c.id=a.class_id
                       JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                       JOIN students s ON s.id=a.student_id
                       WHERE c.teacher_id=%s AND s.teacher_id=%s""",
                    (session["teacher_id"],session["teacher_id"]))
        total_recorded_days = int(cur.fetchone()["n"] or 0)
        cur.execute("""SELECT a.attendance_date,COUNT(DISTINCT CASE WHEN a.status='Present' THEN a.student_id END) AS present
                       FROM attendance a JOIN classes c ON c.id=a.class_id
                       JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                       JOIN students s ON s.id=a.student_id
                       WHERE c.teacher_id=%s AND s.teacher_id=%s AND a.attendance_date BETWEEN %s AND %s
                         GROUP BY a.attendance_date""",
                    (session["teacher_id"],session["teacher_id"],today-timedelta(days=6),today))
        trend_rows=cur.fetchall()
        cur.execute("""SELECT s.id,s.student_id,s.name,
                          COUNT(DISTINCT CASE WHEN a.status='Present' THEN a.attendance_date END) AS present_days,
                          MAX(a.attendance_date) AS last_attendance
                       FROM class_students cs JOIN classes c ON c.id=cs.class_id
                       JOIN students s ON s.id=cs.student_id
                       LEFT JOIN attendance a ON a.student_id=s.id AND a.class_id=cs.class_id
                       WHERE c.teacher_id=%s AND s.teacher_id=%s
                       GROUP BY s.id,s.student_id,s.name ORDER BY s.name""",
                    (session["teacher_id"],session["teacher_id"]))
        student_stats=cur.fetchall()

    cur.close(); conn.close()
    absent_today=max(total_students-present_today,0) if today_has_records else None
    attendance_rate=round(present_today*100/total_students,1) if total_students and today_has_records else None
    trend_map={r["attendance_date"]:int(r["present"] or 0) for r in trend_rows}
    trend_labels=[]; trend_values=[]; day=today-timedelta(days=6)
    while day<=today:
        trend_labels.append(day.strftime("%d %b"))
        trend_values.append(round(trend_map[day]*100/total_students,1) if day in trend_map and total_students else None)
        day+=timedelta(days=1)
    at_risk_students=[]
    if total_recorded_days:
        for student in student_stats:
            present_days=int(student["present_days"] or 0)
            percentage=round(present_days*100/total_recorded_days,1)
            if percentage<75:
                at_risk_students.append({"id":student["id"],"student_id":student["student_id"],
                    "name":student["name"],"present_days":present_days,
                    "attendance_pct":percentage,"last_attendance":student["last_attendance"]})
    rows=attendance_for_day(today,class_id)
    if class_id is None:
        rows=[row for row in rows if row["class_name"] is not None]
    return render_template("dashboard.html",today=today,
        class_name=class_record["class_name"] if class_record else "All assigned classes",
        class_record=class_record,class_id=class_id,classes=classes,
        total_classes=len(classes) if class_id is None else 1,
        total_students=total_students,present_today=present_today,
        absent_today=absent_today,attendance_rate=attendance_rate,rows=rows,
        trend_labels=trend_labels,trend_values=trend_values,
        at_risk_students=at_risk_students,total_recorded_days=total_recorded_days)


def students_in_teacher_classes(teacher_id):
    """Return distinct student rows explicitly assigned to a teacher's classes."""
    conn=get_db(); cur=conn.cursor(dictionary=True)
    cur.execute("""SELECT DISTINCT s.id,s.student_id FROM class_students cs
                   JOIN classes c ON c.id=cs.class_id JOIN students s ON s.id=cs.student_id
                   WHERE c.teacher_id=%s AND s.teacher_id=%s""",(teacher_id,teacher_id))
    rows=cur.fetchall();cur.close();conn.close();return rows


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

@app.get("/analytics")
@teacher_required
def analytics():
    class_id = request.args.get("class_id", type=int)
    class_record = None

    # --------------------------------------------------------
    # FILTER VALUES
    # --------------------------------------------------------

    selected_month = (
        request.args.get(
            "month"
        )
        or
        date.today().strftime(
            "%Y-%m"
        )
    )

    search = (
        request.args.get(
            "q",
            ""
        )
        .strip()
    )

    min_pct, min_error = parse_percentage_filter(
        request.args.get("min_pct"),
        "Minimum attendance"
    )

    max_pct, max_error = parse_percentage_filter(
        request.args.get("max_pct"),
        "Maximum attendance"
    )

    if min_error or max_error:
        flash(min_error or max_error, "error")
        return redirect(url_for("analytics", class_id=class_id))

    if (
        min_pct is not None
        and max_pct is not None
        and min_pct > max_pct
    ):
        flash(
            "Minimum attendance cannot be greater than maximum attendance.",
            "error"
        )
        return redirect(url_for("analytics", class_id=class_id))

    activity = (
        request.args.get(
            "activity",
            "all"
        )
        .strip()
        .lower()
    )

    if activity not in {
        "all",
        "active",
        "inactive",
        "below75"
    }:
        activity = "all"

    # --------------------------------------------------------
    # PARSE MONTH
    # --------------------------------------------------------

    try:

        month_start = datetime.strptime(
            selected_month + "-01",
            "%Y-%m-%d"
        ).date()

    except ValueError:

        month_start = (
            date.today().replace(
                day=1
            )
        )

        selected_month = (
            month_start.strftime(
                "%Y-%m"
            )
        )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    if class_id is not None:
        check_conn=get_db();check_cur=check_conn.cursor(dictionary=True)
        class_record=owned_class(class_id,session["teacher_id"],check_cur)
        check_cur.close();check_conn.close()
        if not class_record:
            return "Class not found",404

    conn = get_db()
    cur = conn.cursor()

    # Total number of distinct attendance dates
    if class_id is None:
        cur.execute(
        """
        SELECT COUNT(
            DISTINCT attendance_date
        )

        FROM attendance a
        JOIN students s ON s.id=a.student_id
        WHERE s.teacher_id=%s
        """
        , (session["teacher_id"],)
        )
    else:
        cur.execute("""SELECT COUNT(DISTINCT a.attendance_date) FROM attendance a
                      JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                      JOIN students s ON s.id=a.student_id
                      WHERE a.class_id=%s AND s.teacher_id=%s""",
                    (class_id,session["teacher_id"]))

    total_recorded_days = int(
        cur.fetchone()[0]
        or 0
    )

    # Total recorded attendance days
    # for selected month
    if class_id is None:
        cur.execute(
        """
        SELECT COUNT(
            DISTINCT attendance_date
        )

        FROM attendance a
        JOIN students s ON s.id=a.student_id

        WHERE s.teacher_id=%s
        AND attendance_date >= %s

        AND attendance_date <
            DATE_ADD(
                %s,
                INTERVAL 1 MONTH
            )
        """,
        (
            session["teacher_id"],
            month_start,
            month_start
        )
        )
    else:
        cur.execute("""SELECT COUNT(DISTINCT a.attendance_date) FROM attendance a
                      JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                      JOIN students s ON s.id=a.student_id
                      WHERE a.class_id=%s AND s.teacher_id=%s AND a.attendance_date >= %s
                      AND a.attendance_date < DATE_ADD(%s, INTERVAL 1 MONTH)""",
                    (class_id,session["teacher_id"],month_start,month_start))

    month_recorded_days = int(
        cur.fetchone()[0]
        or 0
    )

    cur.close()
    conn.close()

    # --------------------------------------------------------
    # ALL STUDENTS
    # --------------------------------------------------------

    students_data = (students_in_class(class_id, session["teacher_id"])
                     if class_id is not None else all_students())

    # --------------------------------------------------------
    # OVERALL STATISTICS
    # --------------------------------------------------------

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    if class_id is None:
        cur.execute(
        """
        SELECT

            s.id,

            COUNT(
                DISTINCT a.attendance_date
            ) AS present_days,

            MAX(
                a.attendance_date
            ) AS last_attendance,

            MIN(
                a.attendance_date
            ) AS first_attendance

        FROM students s

        LEFT JOIN attendance a

            ON a.student_id=s.id

        WHERE s.teacher_id=%s

        GROUP BY s.id
        """,
        (session["teacher_id"],)
        )
    else:
        cur.execute("""SELECT s.id,COUNT(DISTINCT CASE WHEN a.status='Present'
                          THEN a.attendance_date END) AS present_days,
                          MAX(a.attendance_date) AS last_attendance,
                          MIN(a.attendance_date) AS first_attendance
                       FROM class_students cs JOIN students s ON s.id=cs.student_id
                       LEFT JOIN attendance a ON a.student_id=s.id AND a.class_id=cs.class_id
                       WHERE cs.class_id=%s AND s.teacher_id=%s GROUP BY s.id""",
                    (class_id, session["teacher_id"]))

    overall_rows = {
        int(row["id"]):
            row
        for row in cur.fetchall()
    }

    # --------------------------------------------------------
    # MONTHLY STATISTICS
    # --------------------------------------------------------

    if class_id is None:
        cur.execute(
        """
        SELECT

            a.student_id,

            COUNT(
                DISTINCT attendance_date
            ) AS present_days

        FROM attendance a
        JOIN students s ON s.id=a.student_id

        WHERE s.teacher_id=%s
        AND attendance_date >= %s

        AND attendance_date <
            DATE_ADD(
                %s,
                INTERVAL 1 MONTH
            )

        GROUP BY a.student_id
        """,
        (
            session["teacher_id"],
            month_start,
            month_start
        )
        )
    else:
        cur.execute("""SELECT a.student_id,COUNT(DISTINCT a.attendance_date) AS present_days
                       FROM attendance a JOIN students s ON s.id=a.student_id
                       WHERE a.class_id=%s AND s.teacher_id=%s
                         AND a.attendance_date >= %s
                         AND a.attendance_date < DATE_ADD(%s, INTERVAL 1 MONTH)
                       GROUP BY a.student_id""",
                    (class_id, session["teacher_id"], month_start, month_start))

    month_rows = {
        int(row["student_id"]):
            int(
                row["present_days"]
                or 0
            )
        for row in cur.fetchall()
    }

    cur.close()
    conn.close()

    # --------------------------------------------------------
    # CALCULATE STUDENT METRICS
    # --------------------------------------------------------

    for student in students_data:

        sid = int(
            student["id"]
        )

        overall = (
            overall_rows.get(
                sid,
                {}
            )
        )

        present_days = int(
            overall.get(
                "present_days",
                0
            )
            or 0
        )

        month_present = (
            month_rows.get(
                sid,
                0
            )
        )

        last_attendance = (
            overall.get(
                "last_attendance"
            )
        )

        # Overall percentage
        student["overall_pct"] = (

            round(
                present_days *
                100 /
                total_recorded_days,
                1
            )

            if total_recorded_days
            else None
        )

        # Monthly percentage
        student["month_pct"] = (

            round(
                month_present *
                100 /
                month_recorded_days,
                1
            )

            if month_recorded_days
            else None
        )

        student["present_days"] = (
            present_days
        )

        student["month_present"] = (
            month_present
        )

        student["last_attendance"] = (
            last_attendance
        )

        # Active if attended
        # during last 7 days
        student["active"] = bool(

            last_attendance

            and

            (
                date.today()
                -
                last_attendance
            ).days <= 7
        )

        # Risk category
        if student["overall_pct"] is not None and student["overall_pct"] < 75:

            student["risk"] = (
                "warning"
            )

        else:
            student["risk"] = "no-data" if student["overall_pct"] is None else "good"

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------

    all_students_count = (
        len(students_data)
    )

    below_75 = sum(
        1
        for student
        in students_data

        if total_recorded_days and student["overall_pct"] is not None
        and student["overall_pct"] < 75
    )

    active_students = sum(
        1
        for student
        in students_data

        if student["active"]
    )

    inactive_students = (
        all_students_count
        -
        active_students
    )

    average_pct = (

        round(
            sum(
                student["overall_pct"]
                for student
                in students_data
            )
            /
            all_students_count,
            1
        )

        if all_students_count and total_recorded_days

        else None
    )

    # --------------------------------------------------------
    # FILTER STUDENTS
    # --------------------------------------------------------

    filtered = []

    for student in students_data:

        # Search
        search_match = (

            not search

            or

            search.lower()
            in student["name"].lower()

            or

            search.lower()
            in student["student_id"].lower()
        )

        # Minimum %
        min_match = (

            min_pct is None

            or
            (student["overall_pct"] is not None and student["overall_pct"] >= min_pct)
        )

        # Maximum %
        max_match = (

            max_pct is None

            or
            (student["overall_pct"] is not None and student["overall_pct"] <= max_pct)
        )

        # Activity filter
        activity_match = (

            activity == "all"

            or

            (
                activity == "active"
                and
                student["active"]
            )

            or

            (
                activity == "inactive"
                and
                not student["active"]
            )

            or

            (
                activity == "below75"
                and
                total_recorded_days
                and
                student["overall_pct"] is not None
                and
                student["overall_pct"]
                < 75
            )
        )

        if (
            search_match
            and
            min_match
            and
            max_match
            and
            activity_match
        ):

            filtered.append(
                student
            )

    # Lowest attendance first
    filtered.sort(
        key=lambda x:
            x["overall_pct"] if x["overall_pct"] is not None else 101
    )

    # Chart data
    chart_students = (
        filtered[:15]
    )

    chart_labels = [
        student["name"]
        for student
        in chart_students
    ]

    chart_values = [
        student["overall_pct"]
        for student
        in chart_students
    ]

    # --------------------------------------------------------
    # RENDER ANALYTICS
    # --------------------------------------------------------

    return render_template(

        "analytics.html",

        students=filtered,

        summary={
            "all_students_count": all_students_count,
            "below_75": below_75,
            "active_students": active_students,
            "inactive_students": inactive_students,
            "average_pct": average_pct
        },

        selected_month=
            selected_month,

        search=
            search,

        min_pct=
            min_pct
            if min_pct is not None
            else "",

        max_pct=
            max_pct
            if max_pct is not None
            else "",

        activity=
            activity,

        recorded_days=
            total_recorded_days,

        month_recorded_days=
            month_recorded_days,

        chart_labels=
            chart_labels,

        chart_values=
            chart_values,
        class_id=class_id,
        class_record=class_record
    )


# ============================================================
# INDIVIDUAL STUDENT ANALYTICS
# ============================================================

@app.get("/students/<int:sid>/analytics")
@teacher_required
def student_analytics(sid):
    class_id=request.args.get("class_id",type=int)
    conn=get_db();cur=conn.cursor(dictionary=True)
    class_record=None
    if class_id is not None:
        class_record=owned_class(class_id,session["teacher_id"],cur)
        if not class_record:
            cur.close();conn.close();return "Class not found",404
        cur.execute("SELECT 1 FROM class_students WHERE class_id=%s AND student_id=%s",(class_id,sid))
        if not cur.fetchone():
            cur.close();conn.close();return "Student not found in this class",404
    cur.execute("SELECT id,student_id,name,created_at FROM students WHERE id=%s AND teacher_id=%s",
                (sid,session["teacher_id"]))
    student=cur.fetchone()
    if not student:
        cur.close();conn.close();return "Student not found",404
    if class_id is None:
        cur.execute("""SELECT attendance_date,attendance_time,status FROM attendance
                       WHERE student_id=%s ORDER BY attendance_date DESC,attendance_time DESC""",(sid,))
    else:
        cur.execute("""SELECT attendance_date,attendance_time,status FROM attendance
                       WHERE student_id=%s AND class_id=%s
                       ORDER BY attendance_date DESC,attendance_time DESC""",(sid,class_id))
    attendance_rows=cur.fetchall()
    if class_id is None:
        cur.execute("""SELECT COUNT(DISTINCT a.attendance_date) AS n FROM attendance a
                       JOIN students s ON s.id=a.student_id WHERE s.teacher_id=%s""",
                    (session["teacher_id"],))
    else:
        cur.execute("""SELECT COUNT(DISTINCT a.attendance_date) AS n FROM attendance a
                       JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
                       JOIN students s ON s.id=a.student_id
                       WHERE a.class_id=%s AND s.teacher_id=%s""",
                    (class_id,session["teacher_id"]))
    total_recorded_days=int(cur.fetchone()["n"] or 0)
    cur.close();conn.close()
    present=sum(1 for row in attendance_rows if row["status"]=="Present")
    percentage=round(present*100/total_recorded_days,1) if total_recorded_days else 0.0
    last_attendance=attendance_rows[0]["attendance_date"] if attendance_rows else None
    active=bool(last_attendance and (date.today()-last_attendance).days<=7)
    monthly=monthly_student_stats(sid,12,class_id)
    return render_template("student_analytics.html",student=student,
        attendance_history=attendance_rows,present_days=present,total_days=total_recorded_days,
        attendance_pct=percentage,active=active,last_attendance=last_attendance,
        monthly=monthly,monthly_labels=[r["month"] for r in monthly],
        monthly_values=[r["percentage"] for r in monthly],
        class_id=class_id,class_record=class_record)


# ============================================================
# STUDENTS PAGE
# ============================================================

@app.get("/students")
@teacher_required
def students():
    class_id = request.args.get("class_id", type=int)
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    class_record = None
    if class_id is not None:
        class_record = owned_class(class_id, session["teacher_id"], cur)
        if not class_record:
            cur.close(); conn.close()
            return "Class not found", 404
        cur.execute("""SELECT s.id, s.student_id, s.name, s.created_at,
                         (s.embedding IS NOT NULL AND LENGTH(s.embedding)>0) AS face_registered
                      FROM class_students cs JOIN students s ON s.id=cs.student_id
                      WHERE cs.class_id=%s AND s.teacher_id=%s
                      ORDER BY s.name, s.student_id""",
                    (class_id, session["teacher_id"]))
        roster = cur.fetchall()
    else:
        roster = all_students()
        for student in roster:
            student["face_registered"] = True
    cur.close(); conn.close()
    return render_template("students.html", students=roster,
                           class_record=class_record, class_id=class_id)


@app.route("/students/add", methods=["GET", "POST"])
@teacher_required
def add_student():
    class_id = request.values.get("class_id", type=int)
    class_record = None
    if class_id is not None:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        class_record = owned_class(class_id, session["teacher_id"], cur)
        cur.close(); conn.close()
        if not class_record:
            return "Class not found", 404

    if request.method == "POST":
        uid = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()
        if not uid or not name:
            flash("UID and name are required.", "error")
        elif len(uid) > 50:
            flash("Student UID must be 50 characters or fewer.", "error")
        elif len(name) > 100:
            flash("Student name must be 100 characters or fewer.", "error")
        else:
            if class_id is not None:
                conn = get_db(); cur = conn.cursor(dictionary=True)
                cur.execute("SELECT id, teacher_id FROM students WHERE student_id=%s", (uid,))
                existing = cur.fetchone()
                if existing:
                    if existing["teacher_id"] != session["teacher_id"]:
                        cur.close(); conn.close()
                        flash("That UID is already in use.", "error")
                        return redirect(url_for("add_student", class_id=class_id))
                    cur.execute("SELECT 1 FROM class_students WHERE class_id=%s AND student_id=%s",
                                (class_id, existing["id"]))
                    if cur.fetchone():
                        cur.close(); conn.close()
                        flash("This student is already in this class.", "error")
                        return redirect(url_for("students", class_id=class_id))
                    cur.execute("INSERT INTO class_students (class_id, student_id) VALUES (%s,%s)",
                                (class_id, existing["id"]))
                    conn.commit(); cur.close(); conn.close()
                    flash("Existing student added to this class; face data and attendance were preserved.", "success")
                    return redirect(url_for("students", class_id=class_id))
                cur.close(); conn.close()
            return render_template("capture.html", uid=uid, name=name,
                                   class_id=class_id, student_record_id=None,
                                   class_record=class_record)
    return render_template("add_student.html", class_id=class_id, class_record=class_record)


@app.post("/api/students/register")
@teacher_required
def register_student_api():
    uid = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    class_id = request.form.get("class_id", type=int)
    student_record_id = request.form.get("student_record_id", type=int)
    frame_file = request.files.get("frame")
    if not uid or not name or not frame_file:
        return jsonify(ok=False, message="UID, name and face image are required."), 400
    if len(uid) > 50 or len(name) > 100:
        return jsonify(ok=False, message="UID or name exceeds the allowed length."), 400
    if class_id is not None:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        class_record = owned_class(class_id, session["teacher_id"], cur)
        cur.close(); conn.close()
        if not class_record:
            return jsonify(ok=False, message="Class not found."), 404

    frame = cv2.imdecode(np.frombuffer(frame_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(ok=False, message="Invalid image."), 400
    faces = detect(detector(), frame)
    if len(faces) != 1:
        return jsonify(ok=False, message=f"Please show exactly one face. Detected: {len(faces)}"), 400
    emb = feature(recognizer(), frame, faces[0])
    conn = get_db(); cur = conn.cursor()
    try:
        if student_record_id is not None:
            if class_id is None:
                return jsonify(ok=False, message="Class is required to update face data."), 400
            cur.execute("""SELECT s.id FROM students s JOIN class_students cs ON cs.student_id=s.id
                          JOIN classes c ON c.id=cs.class_id
                          WHERE s.id=%s AND cs.class_id=%s AND s.teacher_id=%s AND c.teacher_id=%s""",
                        (student_record_id, class_id, session["teacher_id"], session["teacher_id"]))
            if not cur.fetchone():
                return jsonify(ok=False, message="Student not found in this class."), 404
            cur.execute("UPDATE students SET embedding=%s WHERE id=%s AND teacher_id=%s",
                        (emb.tobytes(), student_record_id, session["teacher_id"]))
            message = "Face data updated successfully."
        else:
            cur.execute("INSERT INTO students (student_id,name,embedding,teacher_id) VALUES (%s,%s,%s,%s)",
                        (uid, name, emb.tobytes(), session["teacher_id"]))
            new_student_id = cur.lastrowid
            if class_id is not None:
                cur.execute("INSERT INTO class_students (class_id,student_id) VALUES (%s,%s)",
                            (class_id, new_student_id))
            message = "Student registered successfully."
        conn.commit()
    except mysql.connector.IntegrityError:
        conn.rollback()
        return jsonify(ok=False, message="That UID already exists."), 409
    except Exception:
        conn.rollback()
        app.logger.exception("Student face registration failed")
        return jsonify(ok=False, message="Face registration failed. Please try again."), 500
    finally:
        cur.close(); conn.close()
    return jsonify(ok=True, message=message)


# ============================================================
# EDIT STUDENT
# ============================================================

@app.route(
    "/students/<int:sid>/edit",
    methods=["GET", "POST"]
)
@teacher_required
def edit_student(sid):

    conn = get_db()
    cur = conn.cursor(dictionary=True)
    class_id = request.values.get("class_id", type=int)
    if class_id is not None:
        if not owned_class(class_id, session["teacher_id"], cur):
            cur.close(); conn.close()
            return "Class not found", 404
        cur.execute("SELECT 1 FROM class_students WHERE class_id=%s AND student_id=%s", (class_id, sid))
        if not cur.fetchone():
            cur.close(); conn.close()
            return "Student not found in this class", 404

    if request.method == "POST":

        uid = request.form.get(
            "student_id",
            ""
        ).strip()

        name = request.form.get(
            "name",
            ""
        ).strip()

        if not uid or not name:
            flash(
                "UID and name are required.",
                "error"
            )

        elif len(uid) > 50:
            flash(
                "Student UID must be 50 characters or fewer.",
                "error"
            )

        elif len(name) > 100:
            flash(
                "Student name must be 100 characters or fewer.",
                "error"
            )

        else:
            try:
                cur.execute(
                    """
                    UPDATE students
                    SET student_id=%s,
                        name=%s
                    WHERE id=%s AND teacher_id=%s
                    """,
                    (uid, name, sid, session["teacher_id"])
                )

                if cur.rowcount == 0:
                    conn.rollback()
                    cur.close()
                    conn.close()
                    return "Student not found", 404

                conn.commit()

                cur.close()
                conn.close()

                flash(
                    "Student updated.",
                    "success"
                )

                return redirect(
                    url_for("students", class_id=class_id)
                )

            except mysql.connector.IntegrityError:
                conn.rollback()
                flash(
                    "UID already exists.",
                    "error"
                )

    cur.execute(
        """
        SELECT
            id,
            student_id,
            name
        FROM students
        WHERE id=%s AND teacher_id=%s
        """,
        (sid, session["teacher_id"])
    )

    student = cur.fetchone()

    cur.close()
    conn.close()

    if not student:
        return "Student not found", 404

    return render_template(
        "edit_student.html",
        student=student, class_id=class_id
    )


# ============================================================
# DELETE STUDENT
# ============================================================

@app.post(
    "/students/<int:sid>/delete"
)
@teacher_required
def delete_student(sid):
    class_id = request.args.get("class_id", type=int)
    conn = get_db(); cur = conn.cursor(dictionary=True)
    if class_id is not None:
        if not owned_class(class_id, session["teacher_id"], cur):
            cur.close(); conn.close()
            return "Class not found", 404
        cur.execute("""SELECT 1 FROM class_students cs JOIN students s ON s.id=cs.student_id
                      WHERE cs.class_id=%s AND s.id=%s AND s.teacher_id=%s""",
                    (class_id, sid, session["teacher_id"]))
        if not cur.fetchone():
            cur.close(); conn.close()
            return "Student not found in this class", 404
        cur.execute("DELETE FROM class_students WHERE class_id=%s AND student_id=%s",
                    (class_id, sid))
        removed = cur.rowcount
        conn.commit(); cur.close(); conn.close()
        if not removed:
            return "Student not found in this class", 404
        flash("Student removed from this class. Student record and attendance history were preserved.", "success")
        return redirect(url_for("students", class_id=class_id))
    # Protect attendance history on the legacy teacher-wide roster as well.
    cur.execute("SELECT id FROM students WHERE id=%s AND teacher_id=%s", (sid, session["teacher_id"]))
    if not cur.fetchone():
        cur.close(); conn.close()
        return "Student not found", 404
    cur.execute("SELECT COUNT(*) AS n FROM attendance WHERE student_id=%s", (sid,))
    if cur.fetchone()["n"]:
        cur.close(); conn.close()
        flash("This student has attendance history. Remove class memberships instead to preserve records.", "error")
        return redirect(url_for("students"))
    cur.execute("DELETE FROM students WHERE id=%s AND teacher_id=%s", (sid, session["teacher_id"]))
    conn.commit(); cur.close(); conn.close()
    flash("Student deleted.", "success")
    return redirect(url_for("students"))


@app.get("/classes/<int:class_id>/students/<int:sid>/face")
@teacher_required
def capture_student_face(class_id, sid):
    conn = get_db(); cur = conn.cursor(dictionary=True)
    class_record = owned_class(class_id, session["teacher_id"], cur)
    if not class_record:
        cur.close(); conn.close()
        return "Class not found", 404
    cur.execute("""SELECT s.id, s.student_id, s.name FROM students s
                  JOIN class_students cs ON cs.student_id=s.id
                  WHERE cs.class_id=%s AND s.id=%s AND s.teacher_id=%s""",
                (class_id, sid, session["teacher_id"]))
    student = cur.fetchone()
    cur.close(); conn.close()
    if not student:
        return "Student not found in this class", 404
    return render_template("capture.html", uid=student["student_id"],
                           name=student["name"], class_id=class_id,
                           student_record_id=sid, class_record=class_record)


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

@app.get("/records")
@teacher_required
def records():
    class_id=request.args.get("class_id",type=int)
    conn=get_db();cur=conn.cursor(dictionary=True)
    class_record=None
    if class_id is not None:
        class_record=owned_class(class_id,session["teacher_id"],cur)
        if not class_record:
            cur.close();conn.close();return "Class not found",404
    cur.close();conn.close()

    selected = (
        request.args.get(
            "date"
        )
        or
        date.today().isoformat()
    )

    try:

        day = date.fromisoformat(
            selected
        )

    except ValueError:

        day = date.today()

    rows = attendance_for_day(day,class_id)

    return render_template(
        "records.html",

        rows=rows,

        selected=day.isoformat(),
        class_id=class_id,
        class_record=class_record
    )


# ============================================================
# CSV EXPORT
# ============================================================

@app.get("/export/attendance")
@teacher_required
def export_attendance():
    class_id=request.args.get("class_id",type=int)
    class_id_raw=request.args.get("class_id")
    if class_id_raw and class_id is None:
        return "Invalid class",400
    selected=request.args.get("date","").strip()
    start_raw=request.args.get("start_date","").strip()
    end_raw=request.args.get("end_date","").strip()
    class_record=None
    conn=get_db();cur=conn.cursor(dictionary=True)
    if class_id is not None:
        class_record=owned_class(class_id,session["teacher_id"],cur)
        if not class_record:
            cur.close();conn.close();return "Class not found",404
    cur.close();conn.close()

    day=None;start_day=None;end_day=None
    try:
        if selected:
            day=date.fromisoformat(selected)
        elif start_raw or end_raw:
            if not start_raw or not end_raw:
                return "Both start_date and end_date are required",400
            start_day=date.fromisoformat(start_raw);end_day=date.fromisoformat(end_raw)
            if start_day>end_day:return "Start date must be on or before end date",400
    except ValueError:
        return "Invalid attendance date",400

    conn=get_db();cur=conn.cursor(dictionary=True)
    if class_id is not None and day is not None:
        cur.execute("""SELECT s.student_id,s.name,c.class_name,%s AS attendance_date,
                          a.attendance_time,COALESCE(a.status,'Absent') AS status
                       FROM class_students cs JOIN students s ON s.id=cs.student_id
                       JOIN classes c ON c.id=cs.class_id
                       LEFT JOIN attendance a ON a.student_id=s.id AND a.class_id=cs.class_id
                         AND a.attendance_date=%s
                       WHERE cs.class_id=%s AND c.teacher_id=%s AND s.teacher_id=%s
                       ORDER BY s.name""",
                    (day,day,class_id,session["teacher_id"],session["teacher_id"]))
        filename=f"attendance_{secure_filename(class_record['class_name'])}_{day.isoformat()}.csv"
    elif class_id is not None:
        sql="""SELECT s.student_id,s.name,c.class_name,a.attendance_date,a.attendance_time,a.status
               FROM attendance a JOIN class_students cs ON cs.class_id=a.class_id AND cs.student_id=a.student_id
               JOIN students s ON s.id=a.student_id JOIN classes c ON c.id=a.class_id
               WHERE a.class_id=%s AND c.teacher_id=%s AND s.teacher_id=%s"""
        params=[class_id,session["teacher_id"],session["teacher_id"]]
        if start_day is not None:
            sql+=" AND a.attendance_date BETWEEN %s AND %s";params.extend([start_day,end_day])
        sql+=" ORDER BY a.attendance_date DESC,a.attendance_time DESC,s.name"
        cur.execute(sql,tuple(params))
        slug=secure_filename(class_record["class_name"]) or "class"
        filename=(f"attendance_{slug}_{start_day}_to_{end_day}.csv" if start_day else f"attendance_{slug}_all.csv")
    elif day is not None:
        cur.execute("""SELECT s.student_id,s.name,COALESCE(c.class_name,'Unassigned') AS class_name,
                          %s AS attendance_date,a.attendance_time,COALESCE(a.status,'Absent') AS status
                       FROM students s LEFT JOIN class_students cs ON cs.student_id=s.id
                       LEFT JOIN classes c ON c.id=cs.class_id AND c.teacher_id=%s
                       LEFT JOIN attendance a ON a.student_id=s.id AND a.attendance_date=%s
                         AND a.class_id <=> cs.class_id
                       WHERE s.teacher_id=%s AND (cs.class_id IS NULL OR c.id IS NOT NULL)
                       UNION ALL
                       SELECT s.student_id,s.name,'Unassigned' AS class_name,%s AS attendance_date,
                              a.attendance_time,a.status
                       FROM students s JOIN attendance a ON a.student_id=s.id
                       WHERE s.teacher_id=%s AND a.attendance_date=%s AND a.class_id IS NULL
                         AND EXISTS (SELECT 1 FROM class_students cs2 JOIN classes c2 ON c2.id=cs2.class_id
                                     WHERE cs2.student_id=s.id AND c2.teacher_id=%s)
                       ORDER BY name,class_name""",
                    (day,session["teacher_id"],day,session["teacher_id"],
                     day,session["teacher_id"],day,session["teacher_id"]))
        filename=f"attendance_{day.isoformat()}.csv"
    else:
        sql="""SELECT s.student_id,s.name,COALESCE(c.class_name,'Unassigned') AS class_name,
                      a.attendance_date,a.attendance_time,a.status
               FROM attendance a JOIN students s ON s.id=a.student_id
               LEFT JOIN classes c ON c.id=a.class_id AND c.teacher_id=%s
               WHERE s.teacher_id=%s AND (a.class_id IS NULL OR (c.id IS NOT NULL AND EXISTS
                 (SELECT 1 FROM class_students cs WHERE cs.class_id=a.class_id AND cs.student_id=s.id)))"""
        params=[session["teacher_id"],session["teacher_id"]]
        if start_day is not None:
            sql+=" AND a.attendance_date BETWEEN %s AND %s";params.extend([start_day,end_day])
        sql+=" ORDER BY a.attendance_date DESC,a.attendance_time DESC,s.name"
        cur.execute(sql,tuple(params))
        filename=(f"attendance_{start_day}_to_{end_day}.csv" if start_day else "attendance_all.csv")

    rows=cur.fetchall();cur.close();conn.close()
    output=StringIO(newline="");writer=csv.writer(output)
    writer.writerow(["Student ID","Name","Class","Attendance Date","Attendance Time","Status"])
    for row in rows:
        writer.writerow([row["student_id"],row["name"],row["class_name"] or "Unassigned",
                         row["attendance_date"] or "",row["attendance_time"] or "",row["status"] or ""])
    response=Response("\ufeff"+output.getvalue(),mimetype="text/csv; charset=utf-8")
    response.headers["Content-Disposition"]=f'attachment; filename="{filename}"'
    return response


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        conn = get_db()

        cur = conn.cursor()

        cur.execute(
            "SELECT DATABASE()"
        )

        database = (
            cur.fetchone()[0]
        )

        cur.close()
        conn.close()

        return jsonify(
            ok=True,
            database=database
        )

    except Exception as e:

        return jsonify(
            ok=False,
            error=str(e)
        ), 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    try:

        download_models()

        ensure_teacher()
        with app.app_context():
            ensure_engagement_tables()

    except Exception as e:

        print(
            "\nSTARTUP ERROR:",
            e
        )

        print(
            "\nMake sure:"
        )

        print(
            "1. MySQL is running."
        )

        print(
            "2. database.sql has been executed."
        )

        print(
            "3. .env contains the correct "
            "MySQL password."
        )

        print()

        raise

    print(
        "Student attendance:"
    )

    print(
        "http://127.0.0.1:5000/attendance"
    )

    print(
        "Teacher login:"
    )

    print(
        "http://127.0.0.1:5000/teacher/login"
    )

    print(
        "Analytics:"
    )

    print(
        "http://127.0.0.1:5000/analytics"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
