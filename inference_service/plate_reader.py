import numpy as np
import supervision as sv
import cv2
from typing import List
import re
import os
import base64

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL_NAME = "nvidia/Nemotron-Nano-V2-12b"
API_KEY = os.environ.get("API_KEY")


class PlateReader:

    def __init__(self):
        if not API_KEY:
            raise EnvironmentError("API_KEY is not set.")

        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
        print(f"VLM API Client initialized for {MODEL_NAME}.")

    def preprocess_plate_image(self, plate_im: np.ndarray) -> np.ndarray:
        return plate_im

    def _recognize_plate_via_vlm(self, plate_im: np.ndarray) -> str:

        _, buffer = cv2.imencode('.jpg', plate_im)
        base64_data = base64.b64encode(buffer).decode("utf-8")

        system_prompt = (
            "You are a STRICT, expert, and highly efficient ALPR processor. "
            "Your output MUST be ONLY one single string containing the license plate characters. "
            "You must NOT include ANY explanatory text, formatting, or analysis (like 'Okay, the plate is...'). "
            "If you successfully read the plate, output the raw characters (letters and numbers) without spaces or hyphens. "
            "If no plate is clearly visible, output ONLY the text 'NO_PLATE'."
        )

        user_prompt = "Identify the license plate number from the provided image. STRICTLY follow the output format guidelines."

        human_content = [
            {"type": "text",
             "text": "Identify the license plate number from the provided image. Output ONLY the raw characters (letters and numbers)."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}
        ]

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_content}
                ]
            )

            if response.choices:
                raw_response_content = response.choices[0].message.content
                pattern = r'(?:</think>\s*)([A-Z0-9\s.-]+)'
                match = re.search(pattern, raw_response_content, re.IGNORECASE | re.DOTALL)
                if match:
                    raw_text = match.group(1).split('\n')[0].strip()
                else:
                    raw_text = raw_response_content.strip()

                if raw_text.strip().upper() == 'NO_PLATE':
                    return ""

                return raw_text.strip()

            return ""

        except Exception as e:
            print(f"LỖI VLM API: {e}")
            return ""

    def clean_plate_text(self, text: str) -> str:
        text = text.replace(' ', '').replace('-', '')

        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        text = "".join(c for c in text if c in allowed)

        patterns = [
            r'(\d{2}[A-Z])-?(\d{4,5})',
            r'(\d{2})-?([A-Z]\d{4,5})',
            r'(\d{2}[A-Z])(\d{4,5})'
        ]

        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    return f"{match.group(1)}-{match.group(2)}"

        return text if len(text) >= 5 else ""

    def read_plate(self, plate_im: np.ndarray) -> str:

        text = self._recognize_plate_via_vlm(plate_im)

        if not text:
            print("[OCR RAW] VLM API returned no results or 'NO_PLATE'.")
            return ""

        print(f"[OCR RAW] Full recognized string (VLM API): {text}")

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

    plate_y1 = int(y2 - height * 0.4)  # 30% from bottom
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
