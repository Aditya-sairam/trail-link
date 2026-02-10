from fastapi import FastAPI, HTTPException, status 
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List,Optional 
import uuid 
from datetime import datetime
from data_models import Patient

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
patients_db: Dict[str, Patient] = {}
def generate_patient_id() ->str: 
    return str(uuid.uuid4())

@app.post("/patients/create",response_model=Patient,status_code = status.HTTP_201_CREATED)
async def create_patient(patient:Patient):
    new_patient_id = generate_patient_id()
    patient.demographics.patient_id = new_patient_id
    patients_db[new_patient_id] = patient 
    return patient 



@app.get("/patients/{patient_id}",response_model=Patient)
async def get_patient(patient_id:str): 
    if patient_id not in patients_db: 
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Patient not found")
    return patients_db[patient_id]

@app.get("/patients",response_model=List[Patient])
async def get_patients():
    all_patients = list(patients_db.values())
    return all_patients

@app.put("/patients/{patient_id}",response_model=Patient)
async def update_patient_info(patient_id:str,updated_patient:Patient): 
    if patient_id not in patients_db: 
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Patient not found")
    updated_patient.demographics.patient_id = patient_id
    patients_db[patient_id] = updated_patient 
    return updated_patient

@app.delete("/patients/{patient_id}",status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient_record(patient_id:str):
    if patient_id not in patients_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Patient not found")
    del patients_db[patient_id]
    return None 

@app.get("/health")
async def health_check():
    return {
        "status":"healthy",
        "timestamp": datetime.now().isoformat(),
        "total_patients": len(patients_db)
    }
    
@app.get("/")
async def get_toor():
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
    uvicorn.run("api:app",host="0.0.0.0",port=8000,reload=True)