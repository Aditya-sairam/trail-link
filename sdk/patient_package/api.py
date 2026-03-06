from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
import uuid
import json
from datetime import datetime
from data_models import Patient
from google.cloud import firestore
from auth import verify_token, require_admin, require_patient_or_admin
import os
import logging
from embeddinngs import get_patient_embedding

# ── Logging setup — makes logs visible in Cloud Run console ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

PATIENTS_COLLECTION = "patients"

app = FastAPI(title="Patient API",
              description="API for managing patient data",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

_firestore_db = None

def get_db():
    global _firestore_db
    if _firestore_db is None:
        _firestore_db = firestore.Client(
            project=os.getenv("GCP_PROJECT_ID"),
            database=os.getenv("FIRESTORE_DATABASE", "(default)")
        )
    return _firestore_db

def generate_patient_id() -> str:
    return str(uuid.uuid4())


# ── Create patient — any authenticated user (patient creating their own profile)
logger = logging.getLogger(__name__)

@app.post("/patients/create", response_model=Patient, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: Patient,                              # ← should be Patient, not Request
    token: dict = Depends(require_patient_or_admin)
):
    patient.demographics.patient_id = generate_patient_id()
    patient.demographics.firebase_uid = token["uid"]
    logger.info(f"Creating patient — id: {patient.demographics.patient_id}, uid: {token['uid']}")

    try:
        doc_ref = get_db().collection(PATIENTS_COLLECTION).document(patient.demographics.patient_id)
        doc_ref.set(patient.model_dump(mode='json'))
        logger.info("Patient saved successfully")
    except Exception as e:
        logger.error(f"Firestore save error: {repr(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return patient


# ── Get single patient — admin can get any, patient can only get their own
@app.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(
    patient_id: str,
    token: dict = Depends(require_patient_or_admin)
):
    doc_ref = get_db().collection(PATIENTS_COLLECTION).document(patient_id)
    patient_doc = doc_ref.get()
    if not patient_doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    patient = Patient(**patient_doc.to_dict())

    # Non-admin users can only view their own profile
    if token.get("role") != "admin" and patient.demographics.firebase_uid != token["uid"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    return patient


# ── Get all patients — admin only
@app.get("/patients", response_model=List[Patient])
async def get_patients(token: dict = Depends(require_admin)):
    patients = []
    docs = get_db().collection(PATIENTS_COLLECTION).stream()
    for doc in docs:
        patients.append(Patient(**doc.to_dict()))
    return patients


# ── Update patient — admin can update any, patient can only update their own
@app.put("/patients/{patient_id}", response_model=Patient)
async def update_patient_info(
    patient_id: str,
    updated_patient: Patient,
    token: dict = Depends(require_patient_or_admin)
):
    doc_ref = get_db().collection(PATIENTS_COLLECTION).document(patient_id)
    patient_doc = doc_ref.get()
    if not patient_doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    existing = Patient(**patient_doc.to_dict())

    # Non-admin users can only update their own profile
    if token.get("role") != "admin" and existing.demographics.firebase_uid != token["uid"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    # Preserve the original firebase_uid and patient_id — never let client overwrite these
    updated_patient.demographics.patient_id = patient_id
    updated_patient.demographics.firebase_uid = existing.demographics.firebase_uid

    doc_ref.update(updated_patient.model_dump(mode='json'))
    return updated_patient


# ── Delete patient — admin only
@app.delete("/patients/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient_record(
    patient_id: str,
    token: dict = Depends(require_admin)
):
    doc_ref = get_db().collection(PATIENTS_COLLECTION).document(patient_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    doc_ref.delete()


# ── Get current user's own patient profile (patient-facing)
@app.get("/me", response_model=Patient)
async def get_my_profile(token: dict = Depends(require_patient_or_admin)):
    """Patient calls this to find their own profile by firebase_uid"""
    docs = get_db().collection(PATIENTS_COLLECTION)\
        .where("demographics.firebase_uid", "==", token["uid"])\
        .limit(1).stream()

    for doc in docs:
        return Patient(**doc.to_dict())

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No profile found.")

@app.get("/me/trial-suggestions", response_model=dict)
async def get_trial_suggestions(token: dict = Depends(require_patient_or_admin)):
    """Patient calls this to find their own profile by firebase_uid"""
    docs = get_db().collection(PATIENTS_COLLECTION)\
        .where("demographics.firebase_uid", "==", token["uid"])\
        .limit(1).stream()

    for doc in docs:
        text_summary =  Patient(**doc.to_dict()).to_text_summary()
        embeddings = get_patient_embedding(text_summary)
        return {
            "summary":text_summary,
            "embeddings":embeddings,
            "embedding_dimensions": len(embeddings)
        }

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No profile found.")


@app.get("/health")
async def health_check():
    doc_ref = get_db().collection(PATIENTS_COLLECTION).stream()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_patients": len(list(doc_ref))
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), reload=True)