from langchain_core.tools import tool
from typing import List, Dict, Any
import json
from pathlib import Path
import sqlite3

@tool
def lookup_db(plate_number: str) -> str:
    """
    Lookup vehicle and violation info from database
    """
    # Hardcoded vehicle data
    vehicle_data = {
        "29L-11156": {
            "id": "001304005771",
            "owner": "Nguyen Ngoc Linh",
            "phone": "0905123456",
            "address": "123 Duong Hoa",
            "vehicle_type": "motorbike",
            "registered": True,
        },
        "29E-30047": {
            "id": "001304021054",
            "owner": "Nguyen Thi Thanh Nhan",
            "phone": "0905987654",
            "address": "456 O Dien",
            "vehicle_type": "car",
            "registered": True,
        },
        "IM7STYSU": {
            "id": "001304033221",
            "owner": "Duyenntt",
            "phone": "0901233456",
            "address": "789 Cau Giay",
            "vehicle_type": "motorbike",
            "registered": True,
        }
    }
    
    violation_data = None
    db_path = "output/violations.db"
    
    try:
        if Path(db_path).exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT plate_number, speed, speed_limit, exceed_speed, location, timestamp, created_at
                FROM violations
                WHERE plate_number = ?
                ORDER BY created_at DESC
                LIMIT 1
                """, (plate_number,)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                violation_data = {
                    "plate_number": result[0],
                    "speed": result[1],
                    "speed_limit": result[2],
                    "exceed_speed": result[3],
                    "location": result[4],
                    "timestamp": result[5],
                    "created_at": result[6]
                }
    except Exception as e:
        print(f"[LOOKUP_DB] Error querying violations DB: {e}")
    
    # Combine vehicle data with violation data
    vehicle_info = vehicle_data.get(plate_number, {
        "registered": False,
        "error": "Vehicle not found in database"
    })
    
    if violation_data:
        vehicle_info.update(violation_data)
        vehicle_info["has_violation"] = True
    else:
        vehicle_info["has_violation"] = False
    
    return json.dumps(vehicle_info)

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