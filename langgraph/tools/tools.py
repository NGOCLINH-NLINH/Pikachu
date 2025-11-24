from langchain_core.tools import tool
from typing import List, Dict, Any
import json
from pathlib import Path
import sqlite3

@tool
def save_violation(violation_data: str) -> str:
    """
    Save violation data to database
    """
    violation = json.loads(violation_data)
    
    db_path = "output/violations.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # create table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            speed REAL,
            speed_limit REAL,
            location TEXT,
            timestamp REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    
    cursor.execute(
        """
        INSERT INTO violations (plate_number, speed, speed_limit, location, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """, (
            violation.get("plate_number", "UNKNOWN"),
            violation["speed"],
            violation["speed_limit"],
            violation["location"],
            violation["timestamp"]
        )
    )
    
    violation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return json.dumps({"violation_id": violation_id, "status": "saved"})