# Phase 2 database snapshot (before Phase 2 migration)

Captured from the configured MySQL database on 2026-10-21 before adding the class/student association table.

## Existing tables and row counts

| Table | Rows |
|---|---:|
| `teacher_users` | 2 |
| `classes` | 0 |
| `students` | 8 |
| `attendance` | 39 |

## Existing relationships

- `classes.teacher_id` references `teacher_users.id`.
- `students.teacher_id` references `teacher_users.id`.
- `attendance.student_id` references `students.id`; attendance is unique per student and date.
- `students.student_id` is globally unique and `students.embedding` is a non-null LONGBLOB.
- There is no class/student relationship in the pre-migration schema. Existing student/class memberships are therefore unknown and are not inferred or backfilled.

Phase 2 migration is additive: create a join table for future explicit memberships. It does not update or delete existing teacher, class, student, face, or attendance rows.
