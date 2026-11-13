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
