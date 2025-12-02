from langchain_core.tools import tool
from typing import List, Dict, Any
import json
from pathlib import Path
import sqlite3

@tool
def lookup_db(plate_number: str) -> str:
    """
    Lookup vehcile info
    """
    vehicle_data = {
        "30A-123.45": {
            "id": "00130405771",
            "owner": " Ng Ngoc Linh",
            "phone": "0905123456",
            "address": "123 Duong Hoa",
            "vehicle_type": "car",
            "registered": True,
    },
        "29B-678.90": {
            "id": "001304021054",
            "owner": "Nhanntt",
            "phone": "0905987654",
            "address": "456 O Dien",
            "vehicle_type": "motorbike",
            "registered": True,
        }
    }
    
    result = vehicle_data.get(plate_number, {
        "registered": False,
        "error": "Vehicle not found in database"
    })
    
    return json.dumps(result)

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
            exceed_speed REAL,
            location TEXT,
            timestamp REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    
    cursor.execute(
        """
        INSERT INTO violations (plate_number, speed, speed_limit, exceed_speed, location, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            violation.get("plate_number", "UNKNOWN"),
            violation["speed"],
            violation["speed_limit"],
            violation["exceed_speed"],
            violation["location"],
            violation["timestamp"]
        )
    )
    
    violation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return json.dumps({"violation_id": violation_id, "status": "saved"})