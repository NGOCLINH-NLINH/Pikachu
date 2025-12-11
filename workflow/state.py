from typing import Annotated, Any, TypedDict, List, Optional, Dict
from langchain_core.messages import ToolMessage
import numpy as np

class TrafficState(TypedDict):
    
    # Frame info
    frame: Annotated[np.ndarray, "current video frame"]
    frame_id: Annotated[int, "current frame index"]
    timestamp: Annotated[float, "current timestamp in seconds"]
    
    # Camera metadata
    camera_id: Annotated[str, "camera identifier"]
    location: Annotated[Optional[str], "camera location"]
    speed_limit: Annotated[Optional[float], "speed limit in km/h"]
    
    # CV res
    detections: Annotated[Optional[Any], "vehicle detections in the frame"]
    speed_values: Annotated[Dict[int, float], "speed mapping"]
    
    # violation info
    violations: Annotated[List[Dict], "list of detected violations"]
    violation_plates: Annotated[List[Dict], "list of violation plates with frame_id, tracker_id and license plate"]
    
    llm_reports: Annotated[List[str], "Generated violation reports"]
    
    human_review_needed: Annotated[bool, "Whether human review is needed"]
    
    next: Annotated[str, "Next action"]