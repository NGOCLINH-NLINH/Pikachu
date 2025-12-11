import json

from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from . import database as db
from workflow.tools.tools import lookup_db
from workflow.agents.report_agent import report_agent as report_agent_llm

app = FastAPI(title="Traffic Violation Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/violations")
def list_violations():
    return db.get_all_violations()


@app.get("/violations/{violation_id}")
def get_violation_detail(violation_id: int):
    violation = db.get_violation_by_id(violation_id)
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")

    vehicle_info_json = lookup_db.invoke({"plate_number": violation["plate_number"]})

    vehicle_info = json.loads(vehicle_info_json)

    violation["vehicle_info"] = vehicle_info
    return violation


@app.post("/explain")
def get_explanation(plate_number: str, speed: float, speed_limit: float):
    explanation = report_agent_llm(plate_number, speed, speed_limit)

    return {"explanation": explanation}


@app.delete("/violations/{violation_id}")
def delete_violation(violation_id: int):

    success = db.delete_violation_by_id(violation_id)

    if success:
        return {"status": "success", "message": f"Violation ID {violation_id} deleted."}
    else:
        raise HTTPException(status_code=404, detail=f"Violation ID {violation_id} not found or could not be deleted.")

