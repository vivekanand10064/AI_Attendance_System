# Database schema snapshot before Phase 1 class changes

Captured from the live `smart_attendance` database on 2026-10-21 before any
Phase 1 schema changes. This is a schema and row-count record, not a data dump.

## Existing tables and rows

| Table | Rows | Existing purpose |
| --- | ---: | --- |
| `teacher_users` | 1 | Teacher accounts and password hashes |
| `students` | 8 | Student UID, name, face embedding, and teacher ownership |
| `attendance` | 39 | Per-student attendance by date and time |

## Existing relationships and constraints

- `teacher_users.username` is unique.
- `students.student_id` is unique.
- `students.teacher_id` references `teacher_users.id` (`fk_students_teacher`).
- `attendance.student_id` references `students.id` (`fk_attendance_student`).
- `attendance(student_id, attendance_date)` is unique (`uq_student_day`).
- There is no `classes` table and no class ID on students.
- `teacher_users.class_name` exists as a non-null `VARCHAR(100)` with default
  `My Class`; it is a legacy single-class field.
- Student face data is stored in `students.embedding` as `LONGBLOB`.

## Existing columns

- `teacher_users`: `id INT AUTO_INCREMENT` (primary key), `username
  VARCHAR(50)` (unique, not null), `password_hash VARCHAR(255)` (not null),
  `created_at DATETIME` (current timestamp, not null), and legacy `class_name
  VARCHAR(100)` (not null, default `My Class`).
- `students`: `id INT AUTO_INCREMENT` (primary key), `student_id VARCHAR(50)`
  (unique, not null), `name VARCHAR(100)` (not null), `embedding LONGBLOB`
  (not null), `created_at DATETIME` (current timestamp, not null), and
  `teacher_id INT` (not null).
- `attendance`: `id INT AUTO_INCREMENT` (primary key), `student_id INT` (not
  null), `attendance_date DATE` (not null), `attendance_time TIME` (not null),
  and `status VARCHAR(20)` (not null, default `Present`).

Existing student and attendance rows will remain unchanged. Phase 1 will add
class records without assigning the existing students to a class.
