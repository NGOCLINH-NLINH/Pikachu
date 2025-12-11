# main.py

import argparse
import cv2
import supervision as sv
from detector import initialize_detector, process_detection
from speed_estimator import SpeedEstimator
from plate_reader import extract_and_read_plate  


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation and Plate Extraction Service"
    )
    parser.add_argument(
        "--source_video_path", default="data/plate.mp4", help="Path to the source video file", type=str,
    )
    parser.add_argument(
        "--target_video_path", default="output/output.mp4", help="Path to the target video file (output)", type=str,
    )
    parser.add_argument(
        "--model_path", default="yolo11n.pt", help="Path to the YOLO model file", type=str,
    )
    parser.add_argument(
        "--confidence_threshold", default=0.3, help="Confidence threshold for the model", type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--speed_threshold_kmh", default=60, help="Speed threshold for plate extraction (km/h)", type=int
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    (
        model,
        byte_track,
        polygon_zone,
        view_transformer,
        box_annotator,
        label_annotator,
        trace_annotator,
    ) = initialize_detector(
        args.model_path, video_info, args.confidence_threshold, args.iou_threshold
    )

    speed_estimator = SpeedEstimator(view_transformer, video_info.fps)

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            detections = process_detection(
                model,
                byte_track,
                polygon_zone,
                frame,
                args.confidence_threshold,
                args.iou_threshold,
            )

            speed_labels = speed_estimator.update_and_estimate(detections)

            final_labels = extract_and_read_plate(
                frame,
                detections,
                speed_labels,
                args.speed_threshold_kmh
            )

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=final_labels
            )

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()