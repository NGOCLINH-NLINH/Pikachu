from workflow.state import TrafficState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, Any
from langchain_openai import ChatOpenAI
import json
import os
from .prompts import SYSTEM_PROMPT

from ..tools.tools import lookup_db

def report_agent(state: TrafficState) -> TrafficState:
    """
    Agent to generate report for speeding violations.
    """
    llm_agent = ChatOpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url="https://api.z.ai/api/coding/paas/v4/",
        model="glm-4.6",
        temperature=0.2,
    ).bind_tools([lookup_db])
    
    violations = state["violations"]
    reports = []
    
    print(f"[REPORT_AGENT] Processing {len(violations)} violations")
    print(f"[REPORT_AGENT] Available plates: {len(state.get('violation_plates', []))}")
    
    for i, violation in enumerate(violations):
        tid = violation["tracker_id"]
        fid = violation["frame_id"]
        
        plate_number = "UNKNOWN"
        for plate_info in state["violation_plates"]:
            if plate_info["frame_id"] == fid and \
                plate_info["tracker_id"] == tid:
                plate_number = plate_info["license_plate"]
                break
        
        print(f"[REPORT_AGENT] Violation {i+1}/{len(violations)} for tracker #{tid}, frame #{fid}, plate: {plate_number}")
        
        try:
            vehicle_info = lookup_db.invoke(plate_number)
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""
                    Vehicle Info: {vehicle_info}
                    """
                )
            ]
            
            response = llm_agent.invoke(messages)
            reports.append(response.content + "\n")
            print(f"[REPORT_AGENT] Generated report for tracker #{tid}")
        except Exception as e:
            print(f"[REPORT_AGENT] Error generating report for tracker #{tid}: {e}")
            reports.append(f"Error generating report: {str(e)}")
    
    return reports
        