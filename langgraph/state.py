from typing import Annotated, TypedDict, List, Optional
import operator
from langchain_core.messages import ToolMessage
import numpy as np

class TrafficState(TypedDict):
    
    frame: Annotated[np.ndarray, "current video frame"]
    frame_id: Annotated[int, "current frame index"]
    timestamp: Annotated[float, "current timestamp in seconds"]
    
    camera_id: Annotated[str, "camera identifier"]
    location: Annotated[Optional[str], "camera location"]
    speed: Annotated[float, "vehicle speed in km/h"]
    speed_limit: Annotated[Optional[float], "speed limit in km/h"]
    plate_number: Annotated[Optional[str], "vehicle plate number"]
    
    vehicle_detected: Annotated[bool, "whether a vehicle is detected in the frame"]
    vehicle_bbox: Annotated[Optional[List[int]], "bounding box of detected vehicle [x1, y1, x2, y2]"]
    
    tools: Annotated[List[dict], "Available tools"]
    tool_results: Annotated[List[ToolMessage], "Results from tools"]
    
    next: Annotated[str, "Next action"]