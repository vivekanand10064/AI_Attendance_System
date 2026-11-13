# Phase 3 database snapshot (before attendance migration)

Captured from the configured MySQL database on 2026-10-21 immediately before Phase 3 schema changes.

| Table | Rows |
|---|---:|
| `teacher_users` | 2 |
| `classes` | 1 |
| `students` | 9 |
| `class_students` | 1 |
| `attendance` | 40 |

The attendance table has `id`, `student_id`, `attendance_date`, `attendance_time`, and `status`, with unique key `uq_student_day (student_id, attendance_date)`. It has no class reference. All 40 pre-migration attendance records therefore remain class-unassigned; none will be mapped by guessing.

Phase 3 adds a nullable class reference for future class attendance. Existing attendance IDs, dates, times, statuses, and student references are preserved. The daily uniqueness rule is migrated to `(student_id, class_id, attendance_date)` so different classes can record their own session while repeats within a class/day remain prevented by application locking and the unique key. Null class IDs preserve legacy rows.
