import numpy as np
import supervision as sv
import easyocr
import cv2
from typing import List
import re


class PlateReader:

    def __init__(self):
        # Initialize EasyOCR with GPU disabled and with verbose=False to speed up initialization
        self.reader = easyocr.Reader(['en', 'vi'], gpu=False, verbose=False)

    def preprocess_plate_image(self, plate_im: np.ndarray) -> np.ndarray:
        # convert to gray scale
        scale_factor = 2
        plate_im = cv2.resize(
            plate_im,
            (0,0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC
        )
        if len(plate_im.shape) ==3:
            gray = cv2.cvtColor(plate_im, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_im

        # increase contrast
        # gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

        # denoise
        # denoised = cv2.fastNlMeansDenoising(gray)

        # thresholding
        # thresh = cv2.adaptiveThreshold(
        #     denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY, 11, 2
        # )

        return gray

    def clean_plate_text(self, text: str) -> str:
        text = text.strip().upper()

        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        text = "".join(c for c in text if c in allowed)

        patterns = [
            r'(\d{2}[A-Z])-?(\d{4,5})',  # 30A-12345 or 30A12345
            r'(\d{2})-?([A-Z]\d{4,5})',  # 30-A12345
            r'(\d{2}[A-Z])(\d{4,5})'     # 30A12345 (no dash)
        ]

        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    return f"{match.group(1)}-{match.group(2)}"

        return text if len(text) >= 5 else ""

    def read_plate(self, plate_im: np.ndarray) -> str:
        processed = self.preprocess_plate_image(plate_im)

        results_list = self.reader.readtext(processed, detail=0)

        if results_list:
            text = "".join(results_list)
            print(f"[OCR RAW] Full recognized string (detail=0): {text}")
        else:
            print("[OCR RAW] EasyOCR returned no results.")
            return ""

        plate_number = self.clean_plate_text(text)

        if not plate_number:
            print(f"[OCR RAW] Cleaned text is EMPTY. Raw text was: {text}")

        return plate_number

def extract_plate_region(
    frame: np.ndarray,
    bbox: np.ndarray,
    expand_ratio: float = 0.2
) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int)

    height = y2 - y1
    width = x2 - x1

    plate_y1 = int(y2 - height * 0.4) # 30% from bottom
    plate_y2 = y2

    expand_x = int(width * expand_ratio)
    plate_x1 = max(0, x1 - expand_x)
    plate_x2 = min(frame.shape[1], x2 + expand_x)

    plate_im = frame[plate_y1:plate_y2, plate_x1:plate_x2]

    return plate_im

def extract_and_read_plate(
        frame: np.ndarray,
        detections: sv.Detections,
        labels: List[str],
        speed_threshold: int = 60
) -> List[str]:
    updated_labels = labels[:]

    reader = PlateReader()

    for i, (tracker_id, label) in enumerate(zip(detections.tracker_id, labels)):
        if "km/h" in label:
            try:
                speed = float(label.split()[-2])

                if speed > speed_threshold:
                    bbox = detections.xyxy[i]
                    plate_im = extract_plate_region(frame, bbox)

                    print(f"Extracting plate from vehicle #{tracker_id} with speed {speed} km/h")
                    plate_number = reader.read_plate(plate_im)

                    if plate_number:
                        print(f"Plate detected: {plate_number}")
                    else:
                        print("No plate detected")

                    updated_labels[i] = f"#{tracker_id} {speed} km/h | {plate_number}"
            except ValueError:
                pass

    return updated_labels