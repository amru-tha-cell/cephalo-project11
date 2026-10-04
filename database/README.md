# Database

Default: **SQLite**, file created automatically at `backend/data/cephalo.db`
the first time the backend starts (`init_db()` in `app/db/database.py` calls
`Base.metadata.create_all`). No manual setup needed for local dev.

To switch to **PostgreSQL**: set `DATABASE_URL` in `backend/.env`, e.g.
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cephalo_db
```
then either let SQLAlchemy create the tables on startup, or run `schema.sql`
directly against an empty database:
```
psql -U postgres -d cephalo_db -f schema.sql
```

`schema.sql` here is a human-readable mirror of
`backend/app/models/db_models.py` -- the Python models are the source of
truth; keep this file in sync if you hand-edit the schema.
