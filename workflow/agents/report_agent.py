from dotenv import load_dotenv

from workflow.state import TrafficState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, Any
from langchain_openai import ChatOpenAI
import json
import os
from .prompts import SYSTEM_PROMPT

from ..tools.tools import lookup_db

def report_agent(plate_number: str, speed: float, speed_limit: float) -> str:
    """
    Agent to generate report for speeding violations.
    """
    load_dotenv()
    llm_agent = ChatOpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url="https://api.tokenfactory.nebius.com/v1/",
        model="moonshotai/Kimi-K2-Thinking",
        temperature=0.2,
    )

    exceed_speed = speed - speed_limit

    print(f"[REPORT_AGENT/API] Request received for plate: {plate_number}, exceed: {exceed_speed:.2f} km/h")

    try:
        vehicle_info = lookup_db.invoke({"plate_number": plate_number})

        # 2. Xây dựng Prompt
        prompt_content = f"""
            Vi phạm tốc độ đã được phát hiện:
            - Biển số: {plate_number}
            - Tốc độ thực tế: {speed:.2f} km/h
            - Tốc độ giới hạn: {speed_limit} km/h
            - Vượt quá: {exceed_speed:.2f} km/h

            Thông tin Chủ xe (CSDL): {vehicle_info}

            Hãy giải thích chi tiết hành vi vượt quá tốc độ {exceed_speed:.2f} km/h này theo luật giao thông.
            """

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]

        # 3. Gọi LLM
        response = llm_agent.invoke(messages)
        print(f"[REPORT_AGENT/API] Generated explanation for {plate_number}")

        return response.content

    except Exception as e:
        print(f"[REPORT_AGENT/API] Error generating explanation for {plate_number}: {e}")
        return f"Xin lỗi, Agent gặp lỗi khi tra cứu và giải thích vi phạm: {str(e)}"
