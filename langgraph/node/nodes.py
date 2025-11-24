from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from state import TrafficState
from inference_service.detector import process_detection
from inference_service.plate_reader import extract_and_read_plate
from tools.tools import save_violation
from agents.report_agent import report_agent
import json

def detect_vehicle(state: TrafficState) -> TrafficState:
    """
    Detect vehicles in the current video frame.
    """
    
    global cv_models
    
    try:
        detections = process_detection(
            cv_models["model"],
            cv_models["byte_track"],
            cv_models["polygon_zone"],
            state["frame"],
            cv_models["confidence"],
            cv_models["iou"]
        )
        
        print(f"[DETECT] Frame {state['frame_id']}: {len(detections) if detections else 0} vehicles")
        
        return {
            **state,
            "detections": detections,
            "next": "calculate_speed"
        }
    
    except Exception as e:
        return {
            **state,
            "detections": None,
            "error": [f"Vehicle detection failed: {str(e)}"],
            "next": "end"
        }

def calculate_speed(state: TrafficState) -> TrafficState:
    """
    Calculate the speed of detected vehicles.
    """
    
    global cv_models
    
    try:
        if state["detections"] is None:
            return {
                **state,
                "speed_values": {},
                "next": "end"
            }
            
        speed_estimator = cv_models["speed_estimator"]
        speed_labels = speed_estimator.update_and_estimate(state["detections"])
        
        speed_values = {}
        
        for labels in speed_labels:
            parts = labels.split()
            tracker_id = int(parts[0])
            speed = float(parts[1])
            speed_values[tracker_id] = speed
            
        print(f"[SPEED] calculated speeds: {speed_values}")
        
        return {
            **state,
            "speed_values": speed_values,
            "next": "check_violation"
        }
        
    except Exception as e:
        return {
            **state,
            "speed_values": {},
            "next": "end",
        }
        
def check_violation(state: TrafficState) -> TrafficState:
    """
    Check for speed violations.
    """
    try:
        violations = []
        
        for tracker_id, speed in state["speed_values"].items():
            if speed > state["speed_limit"]:
                violation = {
                    "tracker_id": tracker_id,
                    "speed": speed,
                    "speed_limit": state["speed_limit"],
                    "execess_speed": speed - state["speed_limit"],
                    "camera_id": state["camera_id"],
                    "location": state["location"],
                    "timestamp": state["timestamp"],
                }
                violations.append(violation)
        print(f"[CHECK] {len(violations)} violations detected")
        
        next_action = "ocr_plate" if violations else "end"
        
        return {
            **state,
            "violations": violations,
            "next": next_action
        }    
    
    except Exception as e:
        return {
            **state,
            "violations": [],
            "next": "end"
        }
        
def ocr_plate(state: TrafficState) -> TrafficState:
    """
    OCR license plates
    """
    try:
        if state["violations"] is None:
            return {
                **state,
                "violations_plates": {},
                "next": "end"
            }
            
        violation_tracker_ids = [v["tracker_id"] for v in state["violations"]]
        
        final_labels = extract_and_read_plate(
            state["frame"],
            state["detections"],
            state["speed_limit"]
        )
        
        violation_plates = {}
        for label in final_labels:
            if "km/h" in label:
                tracker_id = int(label.split()[0])
                
                if tracker_id in violation_tracker_ids:
                    plate_number = label.split("|")[-1].strip()
                    violation_plates[tracker_id] = plate_number
                    
        print(f"[OCR] Extracted {len(violation_plates)} plates")
        
        return {
            **state,
            "violation_plates": violation_plates,
            "next": "save_db"
        }
        
    except Exception as e:
        return {
            **state,
            "violation_plates": {},
            "next": "end"
        }
        
def save_db(state: TrafficState) -> TrafficState:
    """
    Save violations to db
    """
    
    try: 
        if state["violations"] is None:
            return {
                **state,
                "next": "end"
            }
        
        saved_count = 0
        
        for violation in state["violations"]:
            violation["plate_number"] = state["violation_plates"].get(violation["tracker_id"], "UNKNOWN")
            result = save_violation(json.dumps(violation))
            saved_count += 1
        print(f"[SAVE] Saved {saved_count} violations to DB")
        
        return {
            **state,
            "next": "generate_report"
        }
    
    except Exception as e:
        return {
            **state,
            "next": "end"
        }
        
def generate_report(state: TrafficState) -> TrafficState:
    """
    Generate report using LLM
    """
    try:
        if state["violations"] is None:
            return {
                **state,
                "llm_reports": [],
                "next": "end"
            }
        
        reports = report_agent(state)
        return {
            **state,
            "llm_reports": reports,
            "next": "end"
        }
    except Exception as e:
        return {
            **state,
            "llm_reports": [],
            "next": "end"
        }