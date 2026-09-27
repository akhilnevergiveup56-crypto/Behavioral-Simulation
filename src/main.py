from fastapi import FastAPI
from src.schemas.input import SimulationRequest
from src.services.simulation import run_simulation

app = FastAPI(title="Behavioral Simulation API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/simulate")
def simulate(request: SimulationRequest):
    return run_simulation(request)
