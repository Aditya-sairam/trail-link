from fastapi import FastAPI, HTTPException, status 
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List,Optional 
import uuid 
from datetime import datetime
from data_models import Patient
from google.cloud import firestore
import os

PATIENTS_COLLECTION = "patients"

app = FastAPI(title="Patient API",
              description="API for managing patient data",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In prod, react API will be specified
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# In-memory storage for patients (Will be replaced with firestore)
firestore_db = firestore.Client(
    project = os.getenv("GCP_PROJECT_ID"),
    database = os.getenv("FIRESTORE_DATABASE","(default)")
)
def generate_patient_id() ->str: 
    return str(uuid.uuid4())

@app.post("/patients/create",response_model=Patient,status_code = status.HTTP_201_CREATED)
async def create_patient(patient:Patient):
    new_patient_id = generate_patient_id()
    patient.demographics.patient_id = new_patient_id
    doc_ref = firestore_db.collection(PATIENTS_COLLECTION).document(new_patient_id)
    doc_ref.set(patient.model_dump(mode='json'))
    return patient 



@app.get("/patients/{patient_id}",response_model=Patient)
async def get_patient(patient_id:str): 
    doc_ref = firestore_db.collection(PATIENTS_COLLECTION).document(patient_id)
    patient_doc = doc_ref.get()
    if not patient_doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Patient not found")
    return Patient(**patient_doc.to_dict())

## Might require pagination in the future.
@app.get("/patients",response_model=List[Patient])
async def get_patients():
    patients = []
    docs = firestore_db.collection(PATIENTS_COLLECTION).stream()
    for doc in docs:
        patients.append(Patient(**doc.to_dict()))
    return patients

@app.put("/patients/{patient_id}",response_model=Patient)
async def update_patient_info(patient_id:str,updated_patient:Patient): 
    doc_ref = firestore_db.collections(PATIENTS_COLLECTION).document(patient_id)
    patient_doc = doc_ref.get()
    if not patient_doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Patient not found")
    updated_data = updated_patient.model_dump()
    doc_ref.update(updated_data)
    return updated_patient


@app.delete("/patients/{patient_id}",status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient_record(patient_id:str):
    doc_ref = firestore_db.collections(PATIENTS_COLLECTION).document(patient_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Patient not found")
    doc_ref.delete()
    return

@app.get("/health")
async def health_check():
    doc_ref = firestore_db.collection(PATIENTS_COLLECTION).stream()
    return {
        "status":"healthy",
        "timestamp": datetime.now().isoformat(),
        "total_patients": len(list(doc_ref))
    }
    
@app.get("/")
async def get_root():
    return {
       "message" : "Patients API endpoint for Trial Link Application",
        "version" : "1.0.0",
         "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "patients": "/patients/"
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app",host="0.0.0.0",port=int(os.environ.get("PORT", 8080)),reload=True)