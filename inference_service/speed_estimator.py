from collections import defaultdict, deque
import numpy as np
import supervision as sv
from inference_service.detector import ViewTransformer


class SpeedEstimator:

    def __init__(self, view_transformer: ViewTransformer, fps: float):
        self.view_transformer = view_transformer
        self.fps = fps
        self.coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
        self.min_frames_for_speed = int(fps / 2)

    def update_and_estimate(self, detections: sv.Detections) -> list[str]:

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_points = self.view_transformer.transform_points(points=points).astype(int)

        labels = []

        for tracker_id, [_, y] in zip(detections.tracker_id, transformed_points):
            self.coordinates[tracker_id].append(y)

            # Tính toán tốc độ
            if len(self.coordinates[tracker_id]) < self.min_frames_for_speed:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = self.coordinates[tracker_id][0]
                coordinate_end = self.coordinates[tracker_id][-1]

                distance = abs(coordinate_start - coordinate_end)

                time = len(self.coordinates[tracker_id]) / self.fps

                speed = distance / time * 3.6

                labels.append(f"#{tracker_id} {int(speed)} km/h")

        return labels