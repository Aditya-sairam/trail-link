import argparse

from src.pipelines.breast_cancer.ingest import run_full_ingestion
from src.pipelines.breast_cancer.quality import run_quality_checks
from src.pipelines.breast_cancer.bias import generate_bias_report, save_bias_report
import pandas as pd

DEFAULT_RAW = "data/breast_cancer/raw/breast_cancer_trials_raw.csv"
DEFAULT_ENRICHED = "data/breast_cancer/processed/breast_cancer_trials_enriched.csv"
DEFAULT_STATS = "data/breast_cancer/reports/stats.json"
DEFAULT_ANOMALIES = "data/breast_cancer/reports/anomalies.json"
DEFAULT_BIAS = "data/breast_cancer/reports/bias_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Breast cancer trials ingestion (raw -> enriched CSV)")
    parser.add_argument("--status", type=str, default="RECRUITING", help="Trial status filter")
    parser.add_argument("--page_size", type=int, default=1000, help="API page size")
    parser.add_argument("--raw_path", type=str, default=DEFAULT_RAW, help="Path to save raw CSV")
    parser.add_argument("--enriched_path", type=str, default=DEFAULT_ENRICHED, help="Path to save enriched CSV")
    parser.add_argument("--stats_path", type=str, default=DEFAULT_STATS, help="Path to save stats JSON")
    parser.add_argument("--anomalies_path", type=str, default=DEFAULT_ANOMALIES, help="Path to save anomalies JSON")
    parser.add_argument("--bias_path", type=str, default=DEFAULT_BIAS, help="Path to save bias report JSON")

    args = parser.parse_args()

    enriched = run_full_ingestion(
        raw_file_path=args.raw_path,
        enriched_file_path=args.enriched_path,
        status=args.status,
        page_size=args.page_size,
    )

    run_quality_checks(
        enriched_csv_path=enriched,
        stats_path=args.stats_path,
        anomalies_path=args.anomalies_path,
    )

    df = pd.read_csv(args.enriched_path)
    report = generate_bias_report(df)
    save_bias_report(report, args.bias_path)



if __name__ == "__main__":
    main()
