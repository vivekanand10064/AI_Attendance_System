"""Additive classroom engagement portal. Does not read or write attendance tables."""
from __future__ import annotations

import base64
import json
import os
import threading
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import (Blueprint, current_app, flash, jsonify, redirect,
                   render_template, request, session, url_for, abort)

engagement_bp = Blueprint("engagement", __name__)
MODEL_DIR = Path(__file__).resolve().parent / "models"
FER_MODEL = MODEL_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx"
FER_URL = ("https://huggingface.co/opencv/opencv_zoo/resolve/main/models/"
           "facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx?download=true")
EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
POSITIVE = {"happy"}
NEGATIVE = {"angry", "disgust", "fearful", "sad"}
_model_lock = threading.Lock()
_model_cache = None


def db():
    import mysql.connector
    return mysql.connector.connect(**current_app.config["DB_CONFIG"])


def ensure_engagement_tables():
    """Create only new engagement tables. Existing attendance schema is untouched."""
    conn = db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS engagement_sessions (
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
    ) ENGINE=InnoDB""")
    cur.execute("""CREATE TABLE IF NOT EXISTS engagement_buckets (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id BIGINT NOT NULL,
        bucket_number INT NOT NULL,
        emotion_counts LONGTEXT NOT NULL,
        face_observations INT NOT NULL DEFAULT 0,
        UNIQUE KEY uq_engagement_bucket (session_id, bucket_number),
        CONSTRAINT fk_engagement_bucket_session FOREIGN KEY (session_id)
            REFERENCES engagement_sessions(id) ON DELETE CASCADE
    ) ENGINE=InnoDB""")
    conn.commit()
    cur.close()
    conn.close()


def _teacher_id():
    return session.get("teacher_id")


def _owned_session(cur, sid, teacher_id, dictionary=True):
    cur.execute("""SELECT es.*, c.class_name FROM engagement_sessions es
                   JOIN classes c ON c.id=es.class_id
                   WHERE es.id=%s AND es.teacher_id=%s AND c.teacher_id=%s""",
                (sid, teacher_id, teacher_id))
    return cur.fetchone()


def _counts(value=None):
    try:
        data = json.loads(value or "{}")
    except (TypeError, ValueError):
        data = {}
    return {emotion: int(data.get(emotion, 0)) for emotion in EMOTIONS}


def _finish_if_due(cur, row):
    if row and row["status"] == "active" and row["scheduled_end_at"] and datetime.now() >= row["scheduled_end_at"]:
        cur.execute("""UPDATE engagement_sessions
                       SET status='completed', ended_at=scheduled_end_at,
                           actual_duration_seconds=duration_seconds
                       WHERE id=%s AND status='active'""", (row["id"],))
        row["status"] = "completed"
        row["ended_at"] = row["scheduled_end_at"]
        row["actual_duration_seconds"] = row["duration_seconds"]
    return row


def _load_model():
    global _model_cache
    with _model_lock:
        if _model_cache is not None:
            return _model_cache
        yunet_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
        if not yunet_path.exists():
            raise RuntimeError("YuNet face detection model is missing from models/.")
        if not FER_MODEL.exists() or FER_MODEL.stat().st_size < 1_000_000:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(FER_URL, FER_MODEL)
            except Exception as exc:
                raise RuntimeError("Could not download the facial-expression model. Check internet access and restart the app.") from exc
        detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.75, 0.3, 5000)
        net = cv2.dnn.readNet(str(FER_MODEL))
        _model_cache = (detector, net)
        return _model_cache


def _detect_expressions(frame):
    detector, net = _load_model()
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    counts = {emotion: 0 for emotion in EMOTIONS}
    if faces is None:
        return counts, 0
    # MobileFaceNet FER model's five-point alignment template (112x112).
    target = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                       [56.0252, 71.7366], [41.5493, 92.3655],
                       [70.7299, 92.2041]], dtype=np.float32)
    for face in faces:
        points = face[4:14].reshape(5, 2).astype(np.float32)
        transform, _ = cv2.estimateAffinePartial2D(points, target)
        if transform is None:
            continue
        aligned = cv2.warpAffine(frame, transform, (112, 112))
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        normalized = aligned.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        blob = cv2.dnn.blobFromImage(normalized)
        net.setInput(blob, "data")
        output = net.forward("label")
        emotion_index = int(np.argmax(np.asarray(output).reshape(1, -1), axis=1)[0])
        if 0 <= emotion_index < len(EMOTIONS):
            counts[EMOTIONS[emotion_index]] += 1
    return counts, int(sum(counts.values()))


@engagement_bp.before_app_request
def _engagement_lazy_schema():
    # Schema creation is performed explicitly at application startup instead.
    return None


@engagement_bp.get("/engagement")
def index():
    if not _teacher_id():
        return redirect(url_for("teacher_login"))
    conn = db(); cur = conn.cursor(dictionary=True)
    cur.execute("""UPDATE engagement_sessions SET status='completed', ended_at=scheduled_end_at,
                  actual_duration_seconds=duration_seconds
                  WHERE teacher_id=%s AND status='active' AND scheduled_end_at<=NOW()""", (_teacher_id(),))
    cur.execute("SELECT id,class_name,department,academic_year,section FROM classes WHERE teacher_id=%s ORDER BY class_name", (_teacher_id(),))
    classes = cur.fetchall()
    cur.execute("""SELECT es.id, es.subject, es.status, es.duration_seconds, es.created_at,
                          es.started_at, es.ended_at, c.class_name
                   FROM engagement_sessions es JOIN classes c ON c.id=es.class_id
                   WHERE es.teacher_id=%s ORDER BY es.created_at DESC LIMIT 30""", (_teacher_id(),))
    history = cur.fetchall()
    cur.close(); conn.close()
    return render_template("engagement.html", classes=classes, history=history)


@engagement_bp.post("/engagement/create")
def create_session():
    if not _teacher_id():
        return redirect(url_for("teacher_login"))
    class_id = request.form.get("class_id", type=int)
    subject = (request.form.get("subject") or "").strip()[:150]
    duration = request.form.get("duration_seconds", type=int)
    if not class_id or not subject or not duration or duration < 10 or duration > 14400:
        flash("Choose a class, enter a subject, and select a duration from 10 seconds to 4 hours.", "error")
        return redirect(url_for("engagement.index"))
    conn = db(); cur = conn.cursor()
    cur.execute("SELECT id FROM classes WHERE id=%s AND teacher_id=%s", (class_id, _teacher_id()))
    if not cur.fetchone():
        cur.close(); conn.close(); abort(404)
    cur.execute("""INSERT INTO engagement_sessions
        (teacher_id,class_id,subject,duration_seconds,emotion_counts)
        VALUES (%s,%s,%s,%s,%s)""",
        (_teacher_id(), class_id, subject, duration, json.dumps({e: 0 for e in EMOTIONS})))
    sid = cur.lastrowid
    conn.commit(); cur.close(); conn.close()
    return redirect(url_for("engagement.live", sid=sid))


@engagement_bp.get("/engagement/session/<int:sid>")
def live(sid):
    if not _teacher_id():
        return redirect(url_for("teacher_login"))
    conn = db(); cur = conn.cursor(dictionary=True)
    row = _owned_session(cur, sid, _teacher_id())
    cur.close(); conn.close()
    if not row: abort(404)
    if row["status"] == "completed":
        return redirect(url_for("engagement.report", sid=sid))
    return render_template("engagement_live.html", item=row)


@engagement_bp.post("/engagement/api/<int:sid>/start")
def start(sid):
    if not _teacher_id(): return jsonify(ok=False, error="Login required"), 401
    conn = db(); cur = conn.cursor(dictionary=True)
    row = _owned_session(cur, sid, _teacher_id())
    if not row:
        cur.close(); conn.close(); return jsonify(ok=False, error="Session not found"), 404
    if row["status"] == "draft":
        cur.execute("""UPDATE engagement_sessions SET status='active', started_at=NOW(),
                    scheduled_end_at=DATE_ADD(NOW(), INTERVAL duration_seconds SECOND)
                    WHERE id=%s AND teacher_id=%s AND status='draft'""", (sid, _teacher_id()))
        conn.commit()
    cur.execute("SELECT status, duration_seconds, started_at, scheduled_end_at FROM engagement_sessions WHERE id=%s", (sid,))
    row = cur.fetchone()
    cur.close(); conn.close()
    return jsonify(ok=True, status=row["status"], duration_seconds=row["duration_seconds"],
                   end_at=row["scheduled_end_at"].isoformat() if row["scheduled_end_at"] else None)


@engagement_bp.get("/engagement/api/<int:sid>/status")
def status(sid):
    if not _teacher_id(): return jsonify(ok=False), 401
    conn = db(); cur = conn.cursor(dictionary=True)
    row = _owned_session(cur, sid, _teacher_id())
    if not row:
        cur.close(); conn.close(); return jsonify(ok=False), 404
    row = _finish_if_due(cur, row)
    conn.commit()
    remaining = 0
    if row["status"] == "active" and row["scheduled_end_at"]:
        remaining = max(0, int((row["scheduled_end_at"] - datetime.now()).total_seconds()))
    cur.close(); conn.close()
    return jsonify(ok=True, status=row["status"], remaining_seconds=remaining,
                   face_observations=row["total_face_observations"], frame_samples=row["frame_samples"])


@engagement_bp.post("/engagement/api/<int:sid>/frame")
def frame(sid):
    if not _teacher_id(): return jsonify(ok=False, error="Login required"), 401
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    if not isinstance(image_data, str) or "," not in image_data or len(image_data) > 1_500_000:
        return jsonify(ok=False, error="Invalid frame"), 400
    conn = db(); cur = conn.cursor(dictionary=True)
    row = _owned_session(cur, sid, _teacher_id())
    if not row:
        cur.close(); conn.close(); return jsonify(ok=False), 404
    row = _finish_if_due(cur, row)
    if row["status"] != "active":
        conn.commit(); cur.close(); conn.close()
        return jsonify(ok=True, status=row["status"], stopped=True)
    try:
        raw = base64.b64decode(image_data.split(",", 1)[1], validate=True)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None: raise ValueError("Could not decode frame")
        counts, face_count = _detect_expressions(image)
    except Exception as exc:
        conn.rollback(); cur.close(); conn.close()
        current_app.logger.exception("Engagement frame processing failed")
        return jsonify(ok=False, error=str(exc)), 503
    totals = _counts(row["emotion_counts"])
    for emotion, count in counts.items(): totals[emotion] += count
    cur.execute("""UPDATE engagement_sessions SET emotion_counts=%s,
                total_face_observations=total_face_observations+%s,
                frame_samples=frame_samples+1 WHERE id=%s AND status='active'""",
                (json.dumps(totals), face_count, sid))
    elapsed = max(0, int((datetime.now() - row["started_at"]).total_seconds()))
    bucket_no = elapsed // 300
    cur.execute("SELECT id,emotion_counts,face_observations FROM engagement_buckets WHERE session_id=%s AND bucket_number=%s", (sid, bucket_no))
    bucket = cur.fetchone()
    if bucket:
        bucket_counts = _counts(bucket["emotion_counts"])
        for emotion, count in counts.items(): bucket_counts[emotion] += count
        cur.execute("UPDATE engagement_buckets SET emotion_counts=%s,face_observations=face_observations+%s WHERE id=%s",
                    (json.dumps(bucket_counts), face_count, bucket["id"]))
    else:
        cur.execute("INSERT INTO engagement_buckets (session_id,bucket_number,emotion_counts,face_observations) VALUES (%s,%s,%s,%s)",
                    (sid, bucket_no, json.dumps(counts), face_count))
    conn.commit(); cur.close(); conn.close()
    return jsonify(ok=True, face_count=face_count, counts=counts, total_face_observations=row["total_face_observations"] + face_count)


@engagement_bp.post("/engagement/api/<int:sid>/finish")
def finish(sid):
    if not _teacher_id(): return jsonify(ok=False), 401
    conn = db(); cur = conn.cursor(dictionary=True)
    row = _owned_session(cur, sid, _teacher_id())
    if not row:
        cur.close(); conn.close(); return jsonify(ok=False), 404
    if row["status"] in ("active", "draft"):
        cur.execute("""UPDATE engagement_sessions SET status='completed', ended_at=NOW(),
                    actual_duration_seconds=CASE WHEN started_at IS NULL THEN 0
                    ELSE TIMESTAMPDIFF(SECOND,started_at,NOW()) END WHERE id=%s""", (sid,))
        conn.commit()
    cur.close(); conn.close()
    return jsonify(ok=True, redirect=url_for("engagement.report", sid=sid))


@engagement_bp.get("/engagement/report/<int:sid>")
def report(sid):
    if not _teacher_id(): return redirect(url_for("teacher_login"))
    conn = db(); cur = conn.cursor(dictionary=True)
    row = _owned_session(cur, sid, _teacher_id())
    if not row:
        cur.close(); conn.close(); abort(404)
    row = _finish_if_due(cur, row)
    cur.execute("SELECT bucket_number,emotion_counts,face_observations FROM engagement_buckets WHERE session_id=%s ORDER BY bucket_number", (sid,))
    buckets = cur.fetchall()
    conn.commit(); cur.close(); conn.close()
    counts = _counts(row["emotion_counts"])
    total = sum(counts.values())
    positive = sum(counts[e] for e in POSITIVE)
    negative = sum(counts[e] for e in NEGATIVE)
    for bucket in buckets: bucket["counts"] = _counts(bucket["emotion_counts"])
    return render_template("engagement_report.html", item=row, counts=counts,
                           total=total, positive=positive, negative=negative,
                           buckets=buckets, emotions=EMOTIONS)


@engagement_bp.get("/engagement/export/<int:sid>.csv")
def export_csv(sid):
    if not _teacher_id(): return redirect(url_for("teacher_login"))
    import csv
    from io import StringIO
    from flask import Response
    conn = db(); cur = conn.cursor(dictionary=True)
    row = _owned_session(cur, sid, _teacher_id())
    if not row:
        cur.close(); conn.close(); abort(404)
    counts = _counts(row["emotion_counts"])
    cur.close(); conn.close()
    out = StringIO(); writer = csv.writer(out)
    writer.writerow(["Metric", "Count"])
    writer.writerows(counts.items())
    writer.writerow(["Total face observations", row["total_face_observations"]])
    writer.writerow(["Frame samples", row["frame_samples"]])
    return Response("\ufeff" + out.getvalue(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename=engagement_session_{sid}.csv"})
