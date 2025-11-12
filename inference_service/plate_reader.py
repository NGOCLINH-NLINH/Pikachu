import numpy as np
import supervision as sv


def extract_and_read_plate(
        frame: np.ndarray,
        detections: sv.Detections,
        labels: list[str],
        speed_threshold: int = 60
) -> list[str]:
    updated_labels = labels[:]

    for i, (tracker_id, label) in enumerate(zip(detections.tracker_id, labels)):
        if "km/h" in label:
            try:
                speed = int(label.split()[-2])

                if speed > speed_threshold:
                    plate_number = "666"

                    updated_labels[i] = f"#{tracker_id} {speed} km/h | {plate_number}"
            except ValueError:
                pass

    return updated_labels