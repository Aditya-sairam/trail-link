FROM python:3.11-slim 

ENV GCP_PROJECT_ID=project-1e322b00-efe1-41f8-8d9
ENV FIRESTORE_DATABASE=patient-db-dev
ENV FIREBASE_PROJECT_ID=patients-authentication
ENV ENVIRONMENT=development

WORKDIR /app 

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080 

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]