from state import TrafficState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, Any
from langchain_openai import ChatOpenAI
import json
import os
from prompts import SYSTEM_PROMPT

from tools.tools import lookup_db

def report_agent(state: TrafficState) -> TrafficState:
    """
    Agent to generate report for speeding violations.
    """
    llm_agent = ChatOpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url="https://api.z.ai/api/coding/paas/v4/",
        temperature=0.2,
    ).bind_tools([lookup_db])
    
    violations = state["violations"]
    reports = []
    
    for violation in violations:
        tid = violation["tracker_id"]
        plate_number = state["violation_plates"].get(tid, "UNKNOWN")
        vehicle_info = lookup_db(plate_number)
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
                Vehicle Info: {vehicle_info}
                """
            )
        ]
        
        response = llm_agent.invoke(messages)
        reports.append(response.content)
    
    return {
        **state,
        "llm_reports": reports,
    }
        