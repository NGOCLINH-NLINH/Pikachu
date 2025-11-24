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
    violation_plates: Annotated[Dict[int, str], "mapping of tracker_id to license plate"]
    
    human_review_needed: Annotated[bool, "Whether human review is needed"]
    
    next: Annotated[str, "Next action"]