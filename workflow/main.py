import time
from multiprocessing.managers import DictProxy

import cv2
import supervision as sv
import argparse
import sys
import os

from workflow.node import nodes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_service.speed_estimator import SpeedEstimator
from inference_service.detector import initialize_detector
from workflow.state import TrafficState
from langgraph.graph import StateGraph, END
from workflow.node.nodes import *

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


def create_processing_graph():
    """
    Create frame processing workflow (without DB save and report generation)
    """
    # Share cv_models with nodes
    import workflow.node.nodes as nodes
    nodes.cv_models = cv_models

    workflow = StateGraph(TrafficState)

    # Add nodes for frame processing only
    workflow.add_node("detect_vehicle", detect_vehicle)
    workflow.add_node("calculate_speed", calculate_speed)
    workflow.add_node("check_violation", check_violation)
    workflow.add_node("ocr_plate", ocr_plate)

    # Set entry point
    workflow.set_entry_point("detect_vehicle")

    workflow.add_conditional_edges(
        "detect_vehicle",
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
            "check_violation": "check_violation",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "check_violation",
        lambda state: state["next"],
        {
            "ocr_plate": "ocr_plate",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "ocr_plate",
        lambda state: state["next"],
        {
            "end": END
        }
    )

    return workflow.compile()


def create_finalization_graph():
    """
    Create finalization workflow for saving DB and generating reports
    """
    workflow = StateGraph(TrafficState)

    # Add nodes for finalization
    workflow.add_node("save_db", save_db)
    workflow.add_node("generate_report", generate_report)

    # Set entry point
    workflow.set_entry_point("save_db")

    workflow.add_conditional_edges(
        "save_db",
        lambda state: state["next"],
        {
            "generate_report": "generate_report",
            "end": END
        }
    )

    workflow.add_edge("generate_report", END)

    return workflow.compile()


def process_video(
        source_video_path: str,
        speed_limit: float = 60.0,
        camera_id: str = "CAM_001",
        location: str = "HIGHWAY_1"
):
    ocr_queue_ = nodes.ocr_queue
    ocr_results_ = nodes.ocr_results

    ocr_process = multiprocessing.Process(
        target=ocr_worker,
        args=(ocr_queue_, ocr_results_)
    )
    ocr_process.start()

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    print(f"[FPS CHECK]: FPS {video_info.fps}")
    initialize_models(
        model_path="inference_service\\yolo11n.pt",
        video_info=video_info,
        confidence=0.3,
        iou=0.7
    )

    # Create two  workflows
    processing_app = create_processing_graph()
    finalization_app = create_finalization_graph()

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    print(f"Kích thước khung hình: {video_info.width}x{video_info.height}, FPS: {video_info.fps}")

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(BASE_DIR, "inference_service", "output", "output_hehe.avi")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_size = (video_info.width, video_info.height)
    writer = cv2.VideoWriter(
        filename=output_path,
        fourcc=fourcc,
        fps=video_info.fps,
        frameSize=frame_size
    )

    if not writer.isOpened():
        print(f"!!! LỖI QUAN TRỌNG: cv2.VideoWriter KHÔNG THỂ KHỞI TẠO. Codec '{'X264'}' hoặc FFmpeg có vấn đề.")
        return

    print(f"[VIDEO] Saving annotated output to: {output_path} using cv2.VideoWriter")

    # Persistent state for accumulating violations across frames
    persistent_state = {
        "speed_values": {},
        "violations": [],
        "violation_plates": [],
        "plate_readings": {},
        "llm_reports": [],
    }

    total_violations = 0

    for frame_id, frame in enumerate(frame_generator):
        print(f"\n--- Processing Frame {frame_id} ---")

        initial_state: TrafficState = {
            "frame": frame,
            "frame_id": frame_id,
            "timestamp": frame_id / video_info.fps,
            "camera_id": camera_id,
            "location": location,
            "speed_limit": speed_limit,
            "detections": None,
            "speed_values": persistent_state["speed_values"],
            "violations": persistent_state["violations"],
            "violation_plates": persistent_state["violation_plates"],
            "plate_readings": persistent_state["plate_readings"],
            "llm_reports": persistent_state["llm_reports"],
            "next": "save_db",
        }

        # Process frame through detection, speed, violation check, and OCR only
        result = processing_app.invoke(initial_state)

        # Update persistent state with new results
        persistent_state["speed_values"] = result["speed_values"]

        if result["violations"]:
            persistent_state["violations"].extend(result["violations"])

        if result["violation_plates"]:
            persistent_state["violation_plates"].extend(result["violation_plates"])

        violations_count = len(result["violations"])
        total_violations += violations_count

        annotated_frame = frame.copy()
        detections = result["detections"]

        # Draw boxes, IDs, traces
        if detections is not None:
            annotated_frame = cv_models["box_annotator"].annotate(
                scene=annotated_frame,
                detections=detections
            )
            annotated_frame = cv_models["label_annotator"].annotate(
                scene=annotated_frame,
                detections=detections
            )
            annotated_frame = cv_models["trace_annotator"].annotate(
                scene=annotated_frame,
                detections=detections
            )

        # Draw speed per tracked vehicle
        speed_values = result.get("speed_values", {})

        if detections is not None and detections.tracker_id is not None:
            for det_idx in range(len(detections)):
                track_id = int(detections.tracker_id[det_idx])

                if track_id in speed_values:
                    speed = speed_values[track_id]
                    x1, y1, x2, y2 = detections.xyxy[det_idx]

                    anchor = sv.Point(int(x1), int(y1) - 12)
                    text = f"{speed:.1f} km/h"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    thickness = 2

                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    text_x = int(x1)
                    text_y = int(y1) - 10
                    if text_y < text_h:
                        text_y = text_h + 5

                    cv2.rectangle(
                        annotated_frame,
                        (text_x, text_y - text_h - 4),
                        (text_x + text_w + 4, text_y + 4),
                        (0, 0, 0),  # black background
                        -1  # filled
                    )

                    cv2.putText(
                        annotated_frame,
                        text,
                        (text_x + 2, text_y),
                        font,
                        font_scale,
                        (0, 255, 255),  # yellow text
                        thickness,
                        cv2.LINE_AA
                    )

        # Write final annotated frame
        writer.write(annotated_frame)

        if frame_id >= 100:
            break

    try:
        ocr_queue_.join()
    except Exception:
        pass

    ocr_process.terminate()
    ocr_process.join()

    final_violation_plates = []
    for task_id, plate_info in ocr_results_.items():
        if plate_info.get("license_plate"):
            final_violation_plates.append({
                "frame_id": plate_info["frame_id"],
                "tracker_id": plate_info["tracker_id"],
                "license_plate": plate_info["license_plate"]
            })

    writer.release()
    print("Video Writer released.")

    print(f"\n{'=' * 60}")
    print(f" Finalization phase - Processing {total_violations} violations")
    print(f" Total violations in persistent_state: {len(persistent_state['violations'])}")
    print(f" Total violation_plates in persistent_state: {len(persistent_state['violation_plates'])}")
    print('=' * 60)

    finalization_state: TrafficState = {
        "frame": None,
        "frame_id": frame_id,
        "timestamp": frame_id / video_info.fps,
        "camera_id": camera_id,
        "location": location,
        "speed_limit": speed_limit,
        "detections": None,
        "speed_values": persistent_state["speed_values"],
        "violations": persistent_state["violations"],
        "violation_plates": final_violation_plates,
        "plate_readings": persistent_state["plate_readings"],
        "llm_reports": [],
        "next": "",
    }

    # Run finalization workflow
    final_result = finalization_app.invoke(finalization_state)

    # Display final reports
    if final_result["llm_reports"]:
        print(f"\n Generated {len(final_result['llm_reports'])} reports")

    print("\n Processing complete!")


def ocr_worker(input_queue: multiprocessing.JoinableQueue, output_dict: DictProxy):
    try:
        from inference_service.plate_reader import PlateReader
        reader = PlateReader()

        print("\n=== OCR WORKER STARTED ===\n")

        while True:
            try:
                task = input_queue.get(timeout=1)
                task_id = task["task_id"]

                plate_number = reader.read_plate(task["plate_im"])

                output_dict[task_id] = {
                    "frame_id": task["frame_id"],
                    "tracker_id": task["tracker_id"],
                    "license_plate": plate_number,
                    "processed_at": time.time()
                }

                if plate_number:
                    print(f"[OCR WORKER] Task {task_id} SUCCESS: Plate = {plate_number}")
                else:
                    print(f"[OCR WORKER] Task {task_id} NO PLATE detected.")

                input_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[OCR WORKER] Unhandled error: {e}")
                if input_queue:
                    input_queue.task_done()
    except KeyboardInterrupt:
        print("\n=== OCR WORKER SHUTDOWN ===\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    nodes.init_multiprocessing_resources()

    parser = argparse.ArgumentParser(description="Standard LangGraph Traffic Monitoring")
    parser.add_argument("--source_video_path", default="inference_service/data/plate.mp4", type=str)
    parser.add_argument("--speed_limit", default=60, type=float)
    parser.add_argument("--camera_id", default="CAM_001", type=str)
    parser.add_argument("--location", default="Xuan Thuy - KM 10", type=str)

    args = parser.parse_args()

    process_video(
        source_video_path=args.source_video_path,
        speed_limit=args.speed_limit,
        camera_id=args.camera_id,
        location=args.location
    )
