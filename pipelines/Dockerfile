FROM apache/airflow:2.8.1-python3.11

ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db
ENV AIRFLOW__CORE__EXECUTOR=SequentialExecutor
ENV AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8081
# ENV WTF_CSRF_ENABLED=False
# ENV AIRFLOW__WEBSERVER__SECRET_KEY=UYOY6--AHT4WmH4qmnq-pc0mQ0crzo-HTNE9TXc-VZw

COPY dags/ /opt/airflow/dags/

COPY requirements.txt /opt/airflow/

RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

EXPOSE 8081
RUN airflow db init
RUN airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
CMD ["airflow", "standalone"]



