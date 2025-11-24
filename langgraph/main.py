import supervision as sv
import cv2
from inference_service.speed_estimator import SpeedEstimator
from inference_service.detector import initialize_detector, process_detection

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