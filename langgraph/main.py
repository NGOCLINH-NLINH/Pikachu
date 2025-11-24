import supervision as sv
import cv2
from inference_service.speed_estimator import SpeedEstimator
from inference_service.detector import initialize_detector, process_detection
from langgraph.state import TrafficState
from langgraph.graph import StateGraph, END
from node.nodes import *

cv_models = {}

def initialize_models(
    model_path: str,
    video_info,
    confidence: float,
    iou: float
):
    """Initialize and cache CV models."""
    global cv_models
    
    (
        model,
        byte_track,
        polygon_zone,
        view_transformer,
        box_annotator,
        label_annotator,
        trace_annotator,
    ) = initialize_detector(
        model_path, video_info, confidence, iou
    )
    
    cv_models = {
        "model": model,
        "byte_track": byte_track,
        "polygon_zone": polygon_zone,
        "view_transformer": view_transformer,
        "box_annotator": box_annotator,
        "label_annotator": label_annotator,
        "trace_annotator": trace_annotator,
        "speed_estimator": SpeedEstimator(view_transformer, video_info.fps),
        "confidence": confidence,
        "iou": iou,
    }
    
    print("CV models initialized and cached.")
    
def create_traffic_graph():
    """
    Create traffic service workflow
    """
    workflow = StateGraph(TrafficState)
    
    # Add nodes
    workflow.add_node("detect_vehicle", detect_vehicle)
    workflow.add_node("calculate_speed", calculate_speed)
    workflow.add_node("check_violation", check_violation)
    workflow.add_node("ocr_plate", ocr_plate)
    workflow.add_node("save_db", save_db)
    workflow.add_node("generate_report", generate_report)
    
    # Set entry
    workflow.set_entry("detect_vehicle")
    
    workflow.add_conditional_edges(
        "detect_vehicles",
        lambda state: state["next"],
        {
            "calculate_speed": "calculate_speed",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "calculate_speed",
        lambda state: state["next"],
        {
            "check_violations": "check_violations",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "check_violations",
        lambda state: state["next"],
        {
            "ocr_plates": "ocr_plates",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "ocr_plates",
        lambda state: state["next"],
        {
            "save_to_database": "save_to_database",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "save_to_database",
        lambda state: state["next"],
        {
            "generate_llm_reports": "generate_llm_reports",
            "end": END
        }
    )
    
    workflow.add_edge("generate_llm_reports", END)
    
    return workflow.compile()
    
def process_video(
    source_video_path: str,
    speed_limit: float = 60.0,
    camera_id: str = "CAM_001",
    location: str = "HIGHWAY_1"
):
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    
    initialize_models(
        model_path="yolo11n.pt",
        video_info=video_info,
        confidence=0.3,
        iou=0.7
    )
    
    traffic_app = create_traffic_graph()
    
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    for frame_id, frame in enumerate(frame_generator):
        initial_state: TrafficState = {
            "frame": frame,
            "frame_id": frame_id,
            "timestamp": frame_id / video_info.fps,
            "camera_id": camera_id,
            "location": location,
            "speed_limit": speed_limit,
            "detections": None,
            "speed_values": {},
            "violations": [],
            "violation_plates": {},
            "llm_reports": [],
            "next": "",
        }
        
        result = traffic_app.invoke(initial_state)
        
        if result["violations"]:
            print(f"\n{'='*60}")
            print(f"Frame {frame_id}: {len(result['violations'])} violations")
            for i, report in enumerate(result["llm_reports"]):
                print(f"\nReport {i+1}:")
                print(report[:200] + "...")
            print('='*60 + "\n")
        
    
        if frame_id >= 100:  # Process first 100 frames
            break
    
    print("\n✅ Processing complete!")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standard LangGraph Traffic Monitoring")
    parser.add_argument("--source_video_path", default="data/vehicles.mp4", type=str)
    parser.add_argument("--speed_limit", default=60, type=float)
    parser.add_argument("--camera_id", default="CAM_001", type=str)
    parser.add_argument("--location", default="Highway A1 - KM 10", type=str)
    
    args = parser.parse_args()
    
    process_video(
        source_video_path=args.source_video_path,
        speed_limit=args.speed_limit,
        camera_id=args.camera_id,
        location=args.location
    )