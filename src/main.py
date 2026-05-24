from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import execute_pipeline, pretty_metrics_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BurnoutSense AI end-to-end training pipeline")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only validate existing artifacts instead of retraining.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.report_only:
        print("Use the generated files in models/, reports/, metrics/, and visuals/.")
        return
    artifacts = execute_pipeline()
    print(pretty_metrics_summary(artifacts.metrics))
    print(f"Report saved to: {Path('reports') / 'project_report.md'}")


if __name__ == "__main__":
    main()
