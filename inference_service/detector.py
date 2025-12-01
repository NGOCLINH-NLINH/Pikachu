import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

SOURCE = None
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def initialize_detector(
        model_path: str, video_info: sv.VideoInfo, conf_thres: float, iou_thres: float
):
    model = YOLO(model_path)
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=conf_thres
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    # Create a full-frame polygon zone for better detection with test videos
    width, height = video_info.resolution_wh
    source_polygon = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    
    polygon_zone = sv.PolygonZone(polygon=source_polygon)
    view_transformer = ViewTransformer(source=source_polygon, target=TARGET)

    return (
        model,
        byte_track,
        polygon_zone,
        view_transformer,
        box_annotator,
        label_annotator,
        trace_annotator,
    )


def process_detection(
        model: YOLO,
        byte_track: sv.ByteTrack,
        polygon_zone: sv.PolygonZone,
        frame: np.ndarray,
        conf_thres: float,
        iou_thres: float,
) -> sv.Detections:
    result = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    detections = detections[detections.confidence > conf_thres]
    detections = detections[polygon_zone.trigger(detections)]

    detections = detections.with_nms(threshold=iou_thres)
    detections = byte_track.update_with_detections(detections=detections)

    return detections