from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth, credentials
import firebase_admin
import os
import logging

logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK once
if not firebase_admin._apps:
    logger.info("Initializing Firebase Admin SDK...")
    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {
            # Use FIREBASE_PROJECT_ID if set, fallback to GCP_PROJECT_ID
            "projectId": os.getenv("FIREBASE_PROJECT_ID", "patients-authentication"),
        })
        logger.info(f"Firebase Admin initialized with project: {os.getenv('FIREBASE_PROJECT_ID', os.getenv('GCP_PROJECT_ID'))}")
    except Exception as e:
        logger.error(f"Firebase Admin initialization failed: {str(e)}")
        raise
else:
    logger.info("Firebase Admin already initialized.")

bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    logger.info(f"Verifying token — first 20 chars: {token[:20]}...")

    try:
        decoded = auth.verify_id_token(
            token,
            check_revoked=False
        )
        logger.info(f"Token verified successfully — uid: {decoded.get('uid')}, role: {decoded.get('role')}")
        return decoded

    except auth.ExpiredIdTokenError as e:
        logger.error(f"Token expired: {str(e)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired.")

    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid token: {str(e)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected auth error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Auth error: {str(e)}")

def require_admin(decoded_token: dict = Depends(verify_token)):
    logger.info(f"require_admin check — role: {decoded_token.get('role')}")
    if decoded_token.get("role") != "admin":
        logger.warning(f"Access denied — user {decoded_token.get('uid')} is not admin")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required."
        )
    return decoded_token

def require_patient_or_admin(decoded_token: dict = Depends(verify_token)):
    logger.info(f"require_patient_or_admin — uid: {decoded_token.get('uid')}, role: {decoded_token.get('role')}")
    return decoded_token