# Smart Attendance System (MySQL)

## 1. Database setup and upgrades
For a fresh installation, run `database.sql` in MySQL Workbench. It uses
`IF NOT EXISTS` and does not drop the database.

For an existing installation, do not recreate the database or rerun old
destructive setup scripts. On startup, `app.py` safely creates the `classes`
table if it is missing. Existing teachers, students, face embeddings, and
attendance rows are preserved. Existing students are not automatically
assigned to a class.

## 2. Configure MySQL
Copy `.env.example` to a new file named `.env`.

Edit `.env`:
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=YOUR_REAL_MYSQL_PASSWORD
MYSQL_DATABASE=smart_attendance
SECRET_KEY=anything-long-and-random

Do not put quotes around the password unless the quotes are actually part of the password.

## 3. Create a clean virtual environment (Windows)
Open CMD/PowerShell in this project folder:

python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

## 4. Run
python app.py

First run downloads YuNet and SFace model files, so internet is needed once.

Open:
Student attendance: http://127.0.0.1:5000/attendance
Teacher login: http://127.0.0.1:5000/teacher/login

Teacher:
username: teacher
password: teacher123

New teachers can use **Teacher Register** on the login/navigation page to
create an account. After login they can add multiple classes. Existing sample
records remain under the default `teacher` account; those student records
remain teacher-owned until a later phase adds explicit class assignment.

## 5. Check DB connection
Open:
http://127.0.0.1:5000/health

Expected:
{"database":"smart_attendance","ok":true}

## 6. Demo order
1. Teacher Login
2. Students -> Add Student
3. Enter UID + Name (there is NO roll number)
4. Capture face
5. Add at least two students
6. Logout
7. Attendance -> face camera
8. First recognition: Attendance marked successfully
9. Same student again: Attendance already marked
10. Teacher login -> Dashboard
11. Dashboard shows Present and Absent students
12. Records -> choose a date
13. Students -> Edit/Delete

## Database tables
teacher_users(id, username, password_hash, class_name, created_at)
classes(id, teacher_id, class_name, department, academic_year, section, created_at)
students(id, student_id, name, embedding, teacher_id, created_at)
attendance(id, student_id, attendance_date, attendance_time, status)

## Important
- This is an academic prototype.
- Face embeddings are stored in MySQL as LONGBLOB.
- One attendance row per student per date is enforced by MySQL UNIQUE(student_id, attendance_date).
- Student-facing pages cannot access teacher management without login.


## Multi-student classroom attendance
The attendance camera now processes every face detected in a single frame instead
of rejecting frames with more than one face. It compares each detected face with
the selected class roster, marks each recognized student once per class/day, and
shows counts for newly marked, already-present, and unrecognized faces.

For a classroom of 40+ students, use a high-resolution camera (preferably 1080p or
higher), mount it so all faces are visible, provide even lighting, and avoid placing
students too far from the camera. The actual number recognized depends on face size,
camera quality, lighting, and the quality of each student's registered face image.
Test with your classroom before relying on it for official attendance.


## Teacher-controlled classroom sessions
1. Sign in and open a class, then choose **Take Attendance**.
2. Allow camera access and click **Start Attendance**.
3. Keep the camera fixed with all students' faces visible. The browser sends frames repeatedly while the session is active.
4. Use **Pause** when scanning should stop temporarily; **Resume** continues the same session.
5. Click **Finish & Finalize** when the class scan is complete. Students not marked Present are recorded as Absent for the day.
6. Review the roster and use **Set Present / Set Absent** to correct mistakes. Export the CSV report from the page.

A class can have one finalized attendance session per date. The application creates the
`attendance_sessions` table and adds `attendance.session_id` automatically on startup
for existing databases. Existing attendance records are preserved. The finish action
creates explicit Absent rows, so use the manual correction controls if a student arrives
late or the camera missed them.

**Performance note:** The system processes every face detected in each uploaded frame;
40+ simultaneous recognition is dependent on camera resolution, face size, lighting,
CPU performance and enrollment image quality. Run a classroom pilot before using the
records as an official source of attendance. Provide an alternate manual process and
inform students that face recognition is being used.

## Add-on: Classroom Engagement Portal

The Engagement item in the teacher navigation is a separate add-on. It uses the existing teacher login and existing teacher-owned classes, but stores its results in two new tables only: `engagement_sessions` and `engagement_buckets`. It does not insert, update, or delete attendance, student, face-embedding, or attendance-session records.

### Run

1. Start MySQL and use the same `.env` configuration as the attendance application.
2. Run the existing app as usual: `python app.py`.
3. On startup, the app creates the two engagement-only tables. Existing tables and rows are not migrated by the engagement add-on.
4. Log in as a teacher and open **Engagement** in the navigation.
5. Select an existing class, enter a subject, and choose a duration. The default is one minute for a supervisor demo. Presets include 50 seconds, 10 minutes, 50 minutes and 60 minutes; custom duration is also supported.
6. On the session page, confirm that required permissions have been obtained, then select **Start monitoring** and allow webcam access.

### Video and data lifecycle

The browser captures webcam frames and sends sampled still frames to the Flask endpoint for immediate in-memory processing. The application does not create a video recording or save camera frames to disk. Only aggregate expression-observation counts and five-minute trend buckets are stored in MySQL. When the server-side timer expires, the session is marked completed; the browser stops its camera stream and the report is available in session history. If the browser is closed unexpectedly, the camera stream ends with the browser; the session is marked complete on the next status/portal request after its scheduled end time.

The facial-expression model is downloaded on first use into `models/` from OpenCV Zoo's Hugging Face mirror. Internet access is required for that first model download. The detector uses the existing YuNet face detector in `models/` and OpenCV Zoo's MobileFaceNet facial-expression model. The model labels seven expression categories: angry, disgust, fearful, happy, neutral, sad and surprised. Model source and model card: https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition (Apache-2.0).

### Interpretation and safeguards

Counts represent repeated model classifications of detected faces in sampled frames, not unique people, distinct emotional events, student identity, attention, comprehension, laziness or teaching quality. Facial-expression inference is probabilistic and can be inaccurate across lighting, pose, demographics and classroom conditions. Use the report only as a coarse classroom-level demonstration signal, not for grading, discipline or decisions about individual students. Obtain appropriate school, guardian and student permissions before enabling the camera. Audio is not captured.

### Engagement database

The add-on creates `engagement_sessions` (session metadata and aggregate counts) and `engagement_buckets` (five-minute aggregate trend buckets). No raw images or video are stored. A CSV export is available from each completed report.
