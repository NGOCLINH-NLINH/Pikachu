import sqlite3
import json
from pathlib import Path

DB_PATH = "output/violations.db"


def get_all_violations():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM violations ORDER BY created_at DESC")
    violations = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return violations


def get_violation_by_id(violation_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM violations WHERE id = ?", (violation_id,))
    violation = cursor.fetchone()
    conn.close()

    if violation:
        return dict(violation)
    return None


def delete_violation_by_id(violation_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
        rows_deleted = cursor.rowcount
        conn.commit()
        return rows_deleted > 0
    except Exception as e:
        print(f"[DB_ERROR] Error deleting violation {violation_id}: {e}")
        return False
    finally:
        conn.close()
