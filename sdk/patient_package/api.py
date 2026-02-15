from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List
import uuid
from datetime import datetime
from data_models import Patient
from file_watcher import start_file_watcher
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Patient API",
    description="API for managing patient data with automated FHIR file processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# In-memory storage for patients (will be replaced with Firestore)
patients_db: Dict[str, Patient] = {}

# File watcher observer
observer = None


@app.on_event("startup")
async def startup_event():
    """Runs when API starts - starts the file watcher automatically."""
    global observer
    observer = start_file_watcher(patients_db)
    logger.info("🚀 API started with automated file processing")


@app.on_event("shutdown")
async def shutdown_event():
    """Runs when API stops - cleanly stops the file watcher."""
    global observer
    if observer:
        observer.stop()
        observer.join()
    logger.info("🛑 API shutdown complete")


def generate_patient_id() -> str:
    return str(uuid.uuid4())


@app.post("/patients/create", response_model=Patient, status_code=status.HTTP_201_CREATED)
async def create_patient(patient: Patient):
    """Create a new patient record."""
    new_patient_id = generate_patient_id()
    patient.demographics.patient_id = new_patient_id
    patients_db[new_patient_id] = patient
    return patient


@app.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    """Get a specific patient by ID."""
    if patient_id not in patients_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    return patients_db[patient_id]


@app.get("/patients", response_model=List[Patient])
async def get_patients():
    """Get all patients."""
    all_patients = list(patients_db.values())
    return all_patients


@app.put("/patients/{patient_id}", response_model=Patient)
async def update_patient_info(patient_id: str, updated_patient: Patient):
    """Update an existing patient record."""
    if patient_id not in patients_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    updated_patient.demographics.patient_id = patient_id
    patients_db[patient_id] = updated_patient
    return updated_patient


@app.delete("/patients/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient_record(patient_id: str):
    """Delete a patient record."""
    if patient_id not in patients_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    del patients_db[patient_id]
    return None


@app.get("/health")
async def health_check():
    """Health check endpoint with file watcher status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_patients": len(patients_db),
        "file_watcher_active": observer is not None and observer.is_alive()
    }


@app.get("/stats")
async def get_patient_stats():
    """Get statistics about file processing."""
    script_dir = Path(__file__).parent
    incoming_dir = script_dir / "data" / "incoming"
    processed_dir = script_dir / "data" / "processed"
    failed_dir = script_dir / "data" / "failed"
    
    return {
        "total_patients": len(patients_db),
        "files_pending": len(list(incoming_dir.glob("*.json"))) if incoming_dir.exists() else 0,
        "files_processed": len(list(processed_dir.glob("*.json"))) if processed_dir.exists() else 0,
        "files_failed": len(list(failed_dir.glob("*.json"))) if failed_dir.exists() else 0
    }


@app.get("/")
async def get_root():
    """Root endpoint with API information."""
    return {
        "message": "Patients API endpoint for Trial Link Application",
        "version": "1.0.0",
        "automation": "File watcher enabled",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs",
            "patients": "/patients/"
        }
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
