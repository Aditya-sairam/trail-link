from __future__ import annotations

import os
from datetime import datetime, timedelta

import pendulum
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.timetables.trigger import CronTriggerTimetable


# Default weekly schedule: Fridays at 22:00
WEEKLY_SCHEDULE_CRON = os.getenv("TRAILLINK_WEEKLY_SCHEDULE_CRON", "0 22 * * 5")


with DAG(
    dag_id="clinical_trials_weekly_scheduler",
    description="Triggers weekly incremental clinical trials pipeline runs for both diseases in parallel (Friday night, UTC by default).",
    schedule=CronTriggerTimetable(WEEKLY_SCHEDULE_CRON, timezone=pendulum.timezone("UTC")),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "clinical-trials", "scheduler"],
    default_args={
        "owner": "sanika",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": False,
    },
) as dag:
    trigger_diabetes = TriggerDagRunOperator(
        task_id="trigger_diabetes_pipeline",
        trigger_dag_id="clinical_trials_data_pipeline",
        conf={"condition": "diabetes"},
        wait_for_completion=False,
        reset_dag_run=False,
    )

    trigger_breast_cancer = TriggerDagRunOperator(
        task_id="trigger_breast_cancer_pipeline",
        trigger_dag_id="clinical_trials_data_pipeline",
        conf={"condition": "breast_cancer"},
        wait_for_completion=False,
        reset_dag_run=False,
    )

    [trigger_diabetes, trigger_breast_cancer]
