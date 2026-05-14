from pathlib import Path

MOEX_SECURITIES_FILE = Path("moex_securities.json")
MOEX_CANDLES_DIR = Path("moex_candles")
SUMMARY_PATH = Path("candles_summary.csv")
FILTERED_SUMMARY_PATH = Path("filtered_candles_summary.csv")

HORIZON = 15
WINDOW_SIZE = 30

TRAIN_UNTIL = "2025-01-01"
VALIDATE_UNTIL = "2025-07-01"

# ============================================================
# Имена этапов обучения
# ============================================================

# validation: модель обучается только на train и строит valid predictions.
VALIDATION_STAGE_NAME = "validation"

# final: модель переобучается на train + validation и строит test predictions.
FINAL_STAGE_NAME = "final"

# ============================================================
# Директории для результатов и промежуточных данных
# ============================================================

DATA_ROOT = Path("data")

# Универсальные shard-файлы для sequence-моделей:
# GRU, LSTM, Transformer.
SEQUENCE_SHARDS_DIR = DATA_ROOT / "sequence_shards"

# Максимальное число строк внутри одного shard-файла.
# Чем меньше значение, тем меньше RAM требуется при обучении,
# но тем больше файлов и накладных расходов на чтение.
SEQUENCE_SHARD_MAX_ROWS = 750_000

