import csv
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "telemetry"
LOG_DIR.mkdir(parents=True, exist_ok=True)

QUERY_LOG = LOG_DIR / "query_logs.csv"

def log_query(question: str, answer: str, latency_ms: float, chunks_retrieved: int):
    write_header = not QUERY_LOG.exists()
    with open(QUERY_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "question", "answer_length", "chunks_retrieved", "latency_ms"])
        writer.writerow([datetime.now().isoformat(), question, len(answer), chunks_retrieved, latency_ms])