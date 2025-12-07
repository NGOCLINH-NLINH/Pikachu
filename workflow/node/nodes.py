from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from workflow.state import TrafficState
from inference_service.detector import process_detection
from inference_service.plate_reader import extract_and_read_plate
from workflow.tools.tools import save_violation
from workflow.agents.report_agent import report_agent
import json
import os
from datetime import datetime
from pathlib import Path

# Global variable to store CV models
cv_models = {}

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
        
        print(f"[SPEED] Raw speed labels: {speed_labels}")
        
        speed_values = {}
        
        for labels in speed_labels:
            if "km/h" in labels:
                try:
                    # Parse format like "#1 127 km/h"
                    parts = labels.split()
                    tracker_id = int(parts[0].replace("#", ""))
                    speed_part = parts[1] if len(parts) > 1 else "0"
                    speed = float(speed_part.split()[0])  # Get first number before "km/h"
                    speed_values[tracker_id] = speed
                except:
                    print(f"[SPEED] Failed to parse speed label: {labels}")
            else:
                # Handle labels without km/h (like '#1', '#2', etc.)
                try:
                    tracker_id = int(labels.replace("#", ""))
                    speed_values[tracker_id] = 0.0  # Default speed when no km/h info
                except:
                    pass
            
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
                    "frame_id": state["frame_id"],
                    "tracker_id": tracker_id,
                    "speed": speed,
                    "speed_limit": state["speed_limit"],
                    "exceed_speed": speed - state["speed_limit"],  
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
                "violation_plates": [],
                "next": "end"
            }
            
        violation_tracker_ids = [v["tracker_id"] for v in state["violations"]]
        
        speed_labels = []
        for tracker_id, speed in state["speed_values"].items():
            if speed > state["speed_limit"]:
                speed_labels.append(f"#{tracker_id} {speed} km/h")
        
        final_labels = extract_and_read_plate(
            state["frame"],
            state["detections"],
            speed_labels,
            state["speed_limit"]
        )
        print(f"[OCR] extract_and_read_plate returned: {final_labels}")
        
        violation_plates = []
        for label in final_labels:
            if "km/h" in label:
                try:
                    tracker_id = int(label.split()[0].replace("#", ""))
                    
                    if tracker_id in violation_tracker_ids:
                        plate_number = label.split("|")[-1].strip()
                        
                        if plate_number == "":
                            continue
                        
                        violation_plates.append({
                            "frame_id": state["frame_id"],
                            "tracker_id": tracker_id,
                            "license_plate": plate_number
                        })
                        print(f"[OCR] Vehicle #{tracker_id}: Plate = {plate_number}")
                except Exception as e:
                    print(f"[OCR] Error parsing label '{label}': {e}")
                    
        print(f"[OCR] Extracted {len(violation_plates)} plates from violations")
        
        return {
            **state,
            "violation_plates": violation_plates,
            "next": "end"
        }
        
    except Exception as e:
        print(f"[OCR] Error in ocr_plate: {e}")
        return {
            **state,
            "violation_plates": [],
            "next": "end"
        }
        
def save_db(state: TrafficState) -> TrafficState:
    """
    Save violations to db
    """
    try:
    
        if state["violations"] is None or len(state["violations"]) == 0:
            return {
                **state,
                "next": "end"
            }
       
        saved_count = 0
        saved_plates = set()
        saved_violations = []  # Track violations that were actually saved
        
        for _, violation in enumerate(state["violations"]):
            plate_number = None
            for plate_info in state["violation_plates"]:
                if plate_info["frame_id"] == violation["frame_id"] and \
                plate_info["tracker_id"] == violation["tracker_id"]:
                    plate_number = plate_info["license_plate"]
                    break
            
            if plate_number is None:
                continue
            
            if plate_number in saved_plates:
                continue
            
            violation_copy = violation.copy()
            violation_copy["plate_number"] = plate_number
            
            _ = save_violation.invoke({"violation_data": json.dumps(violation_copy)})
            saved_count += 1
            saved_plates.add(plate_number)
            saved_violations.append(violation_copy)  
        
        print(f"[SAVE] Saved {saved_count} unique violations to DB")
        
        return {
            **state,
            "violations": saved_violations, 
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

        reports_placeholder = [f"Ticket template prepared for violation #{i + 1}" for i in
                               range(len(state["violations"]))]

        if reports_placeholder:
            output_dir = Path("output/reports")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Lưu file đánh dấu hoặc file template đơn giản
            for i, placeholder in enumerate(reports_placeholder):
                report_file = output_dir / f"processed_ticket_{timestamp}_{i + 1}.log"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(
                        f"[DASHBOARD_READY] Violation Ticket #{i + 1} saved to DB and ready for Dashboard display.\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(placeholder)
                print(f"[REPORT] Marked ticket as processed in {report_file}")

        return {
            **state,
            "llm_reports": reports_placeholder,  # Giờ là list các string placeholder
            "next": "end"
        }
    except Exception as e:
        print(f"[REPORT] Error in finalize node: {e}")
        return {
            **state,
            "llm_reports": [],
            "next": "end"
        }
