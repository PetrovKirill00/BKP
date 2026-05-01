from pathlib import Path

MOEX_SECURITIES_FILE = Path("moex_securities.json")
MOEX_CANDLES_DIR = Path("moex_candles")
SUMMARY_PATH = Path("candles_summary.csv")
FILTERED_SUMMARY_PATH = Path("filtered_candles_summary.csv")

HORIZON = 1
WINDOW_SIZE = 30

TRAIN_UNTIL = "2025-01-01"
VALIDATE_UNTIL = "2025-07-01"