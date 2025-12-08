import os
import cv2
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

IMAGE_PATH = r"F:\PythonProj\Pikachu\inference_service\output\plates\00015_10_67.0kmh_NO_PLATE.png"
MODEL_NAME = "nvidia/Nemotron-Nano-V2-12b"
API_KEY = os.environ.get("API_KEY")

API_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def encode_image_to_base64(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return ""

    _, buffer = cv2.imencode('.jpg', img)

    return base64.b64encode(buffer).decode("utf-8")


def call_vlm_for_ocr(image_path: str, model_name: str) -> str:
    base64_data = encode_image_to_base64(image_path)

    if not base64_data:
        return "Encoding Failed"

    if not API_KEY:
        return "LỖI: API_KEY không được tìm thấy."

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

        system_prompt = (
            "You are an expert Automatic License Plate Recognition (ALPR) system. "
            "Your ONLY task is to identify the license plate number visible in the image. "
            "Respond ONLY with the sequence of characters (letters and numbers). "
            "If no plate is clearly visible, respond with 'NO_PLATE'."
        )

        user_message = "What is the license plate number?"

        print(f"Gửi ảnh tới API ({model_name})...")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_message
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_data}"
                            }
                        }
                    ]
                }
            ]
        )

        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "No response content received."

    except Exception as e:
        return f"LỖI API/MÔ HÌNH: {e}"


if __name__ == "__main__":
    print(f"Kiểm tra ảnh: {IMAGE_PATH}")

    plate_number = call_vlm_for_ocr(IMAGE_PATH, MODEL_NAME)

    print("\n=========================================")
    print(f"KẾT QUẢ VLM OCR: {plate_number}")
    print("=========================================")