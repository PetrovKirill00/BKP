from __future__ import annotations

import builtins
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from constants import HORIZON, TRAIN_UNTIL, VALIDATE_UNTIL, WINDOW_SIZE

try:
    from constants import DATA_ROOT
except ImportError:
    DATA_ROOT = Path("data")

try:
    from constants import SEQUENCE_SHARDS_DIR
except ImportError:
    SEQUENCE_SHARDS_DIR = DATA_ROOT / "sequence_shards"

try:
    from constants import SEQUENCE_SHARD_MAX_ROWS
except ImportError:
    SEQUENCE_SHARD_MAX_ROWS = 750_000

try:
    from constants import FILTERED_SUMMARY_PATH
except ImportError:
    FILTERED_SUMMARY_PATH = Path("filtered_candles_summary.csv")

DATASET_PATH = Path("minute_dataset.parquet")

VERBOSE = True
FORCE_REBUILD_SEQUENCE_SHARDS = False

EXECUTION_TARGET_COLS = [
    f"execution_return_{horizon}"
    for horizon in range(1, HORIZON + 1)
]

HOLD_TARGET_COLS = [
    f"hold_return_{horizon}"
    for horizon in range(1, HORIZON + 1)
]

MODEL_TARGET_COLS = EXECUTION_TARGET_COLS + HOLD_TARGET_COLS

FEATURE_COLS = [
    "is_imputed",
    "hour", "minute", "minute_from_midnight",
    "day_of_week", "day_of_month", "month", "days_in_month",
    "is_month_start", "is_month_end",

    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "minute_from_midnight_sin", "minute_from_midnight_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_month_sin", "day_of_month_cos",

    "log_volume", "log_value",
    "return_1", "return_2", "return_3", "return_5",
    "return_10", "return_15", "return_30",

    "log_volume_diff_1", "log_value_diff_1",
    "log_volume_ma_5", "log_value_ma_5",

    "close_to_ma_5", "close_to_ma_15", "close_to_ma_30",

    "volatility_5", "volatility_15", "hl_range",

    "body", "upper_wick", "lower_wick",
]

SHARD_COLUMNS = [
    "begin",
    "secid",
    "trade_date",
] + FEATURE_COLS + MODEL_TARGET_COLS


def log(*args: Any) -> None:
    """
    Печатает диагностическое сообщение с текущим временем.
    """
    if VERBOSE:
        now = datetime.now().strftime("%H:%M:%S")
        message = " ".join(str(arg) for arg in args)
        builtins.print(f"[{now}] {message}", flush=True)


def normalize_dataset_name(dataset_name: str) -> str:
    """
    Приводит имя набора shard-файлов к каноническому виду.
    """
    name = dataset_name.lower()

    aliases = {
        "train": "train",
        "validation": "train",
        "train_only": "train",
        "train_valid": "train_valid",
        "final": "train_valid",
        "final_train_valid": "train_valid",
    }

    if name in aliases:
        return aliases[name]

    # Для произвольных экспериментов разрешаем безопасное имя.
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)


def get_sequence_shards_dir(dataset_name: str = "train") -> Path:
    """
    Возвращает директорию с shard-файлами для заданного обучающего набора.
    """
    normalized_name = normalize_dataset_name(dataset_name)

    return (
        SEQUENCE_SHARDS_DIR
        / f"horizon_{HORIZON}_window_{WINDOW_SIZE}"
        / normalized_name
    )


def get_manifest_path(dataset_name: str = "train") -> Path:
    """
    Возвращает путь к manifest.json для shard-файлов.
    """
    return get_sequence_shards_dir(dataset_name) / "manifest.json"


def get_shard_paths(dataset_name: str = "train") -> list[Path]:
    """
    Возвращает список shard-файлов в правильном порядке.
    """
    return sorted(get_sequence_shards_dir(dataset_name).glob("shard_*.npz"))


def get_sequence_train_shards_dir() -> Path:
    """
    Возвращает директорию с train shard-файлами для sequence-моделей.
    """
    return get_sequence_shards_dir("train")


def read_secids_for_sharding(train_until: str) -> list[str]:
    """
    Возвращает список бумаг, для которых нужно построить shard-файлы.
    """
    if FILTERED_SUMMARY_PATH.exists():
        summary = pd.read_csv(FILTERED_SUMMARY_PATH)

        if "secid" not in summary.columns:
            raise ValueError(
                f"В {FILTERED_SUMMARY_PATH} нет колонки secid."
            )

        secids = sorted(summary["secid"].astype(str).unique().tolist())
        log(f"Список secid взят из {FILTERED_SUMMARY_PATH}: {len(secids)}")
        return secids

    log(
        f"{FILTERED_SUMMARY_PATH} не найден. "
        "Читаю список secid из minute_dataset.parquet."
    )

    secids_df = pd.read_parquet(
        DATASET_PATH,
        columns=["secid"],
        filters=[("begin", "<", pd.Timestamp(train_until))],
    )

    secids = sorted(secids_df["secid"].astype(str).unique().tolist())

    del secids_df

    log(f"Список secid прочитан из parquet: {len(secids)}")

    return secids


def read_train_part_for_secid(secid: str, train_until: str) -> pd.DataFrame:
    """
    Читает обучающую часть minute_dataset.parquet только для одной бумаги.
    """
    df = pd.read_parquet(
        DATASET_PATH,
        columns=SHARD_COLUMNS,
        filters=[
            ("begin", "<", pd.Timestamp(train_until)),
            ("secid", "=", secid),
        ],
    )

    if df.empty:
        return df

    df = df.sort_values(["trade_date", "begin"], kind="mergesort")
    df = df.reset_index(drop=True)

    return df


def validate_shard_source_df(df: pd.DataFrame, secid: str) -> None:
    """
    Проверяет данные перед записью в shard.
    """
    missing_cols = [
        col
        for col in SHARD_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"{secid}: в train-данных отсутствуют колонки: {missing_cols}"
        )

    feature_nan_rows = df[FEATURE_COLS].isna().any(axis=1).sum()
    if feature_nan_rows > 0:
        raise ValueError(
            f"{secid}: найдено {feature_nan_rows} строк с NaN в признаках."
        )

    target_nan_rows = df[MODEL_TARGET_COLS].isna().any(axis=1).sum()
    if target_nan_rows > 0:
        raise ValueError(
            f"{secid}: найдено {target_nan_rows} строк с NaN в target."
        )

    feature_values = df[FEATURE_COLS].to_numpy(copy=False)
    if not np.isfinite(feature_values).all():
        raise ValueError(f"{secid}: есть inf/-inf в признаках.")

    target_values = df[MODEL_TARGET_COLS].to_numpy(copy=False)
    if not np.isfinite(target_values).all():
        raise ValueError(f"{secid}: есть inf/-inf в target.")


def save_shard(
        shard_id: int,
        groups_x: list[np.ndarray],
        groups_y: list[np.ndarray],
        group_lengths: list[int],
        group_secids: list[str],
        group_trade_dates: list[str],
        output_dir: Path,
) -> dict[str, Any]:
    """
    Сохраняет один shard-файл.
    """
    if not groups_x:
        raise ValueError("Нельзя сохранить пустой shard")

    x = np.concatenate(groups_x, axis=0).astype("float32", copy=False)
    y = np.concatenate(groups_y, axis=0).astype("float32", copy=False)

    group_lengths_np = np.asarray(group_lengths, dtype=np.int64)
    group_secids_np = np.asarray(group_secids)
    group_trade_dates_np = np.asarray(group_trade_dates)

    path = output_dir / f"shard_{shard_id:05d}.npz"

    # np.savez выбран намеренно: он быстрее np.savez_compressed.
    # Для обучения важнее скорость чтения, а не минимальный размер на диске.
    np.savez(
        path,
        x=x,
        y=y,
        group_lengths=group_lengths_np,
        group_secids=group_secids_np,
        group_trade_dates=group_trade_dates_np,
    )

    row_count = int(len(x))
    group_count = int(len(group_lengths))

    log(f"Сохранён {path.name}: rows={row_count}, groups={group_count}")

    return {
        "path": str(path),
        "rows": row_count,
        "groups": group_count,
    }


def build_sequence_shards(
        dataset_name: str,
        train_until: str,
        force_rebuild: bool = FORCE_REBUILD_SEQUENCE_SHARDS,
) -> Path:
    """
    Создаёт shard-файлы для sequence-моделей из minute_dataset.parquet.

    dataset_name='train'       -> строки begin < TRAIN_UNTIL.
    dataset_name='train_valid' -> строки begin < VALIDATE_UNTIL.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Не найден датасет: {DATASET_PATH}")

    output_dir = get_sequence_shards_dir(dataset_name)
    manifest_path = get_manifest_path(dataset_name)

    if manifest_path.exists() and get_shard_paths(dataset_name) and not force_rebuild:
        log(f"Sequence shards уже существуют: {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    for old_shard in output_dir.glob("shard_*.npz"):
        old_shard.unlink()

    if manifest_path.exists():
        manifest_path.unlink()

    log("=" * 80)
    log("Начинаю подготовку sequence shards")
    log(f"dataset_name: {dataset_name}")
    log(f"DATASET_PATH: {DATASET_PATH}")
    log(f"train_until: {train_until}")
    log(f"HORIZON: {HORIZON}")
    log(f"WINDOW_SIZE: {WINDOW_SIZE}")
    log(f"SEQUENCE_SHARD_MAX_ROWS: {SEQUENCE_SHARD_MAX_ROWS}")
    log(f"output_dir: {output_dir}")

    secids = read_secids_for_sharding(train_until=train_until)

    shard_infos = []
    shard_id = 0

    pending_x: list[np.ndarray] = []
    pending_y: list[np.ndarray] = []
    pending_group_lengths: list[int] = []
    pending_group_secids: list[str] = []
    pending_group_trade_dates: list[str] = []
    pending_rows = 0

    total_rows = 0
    total_groups = 0

    def flush_pending() -> None:
        nonlocal shard_id
        nonlocal pending_x
        nonlocal pending_y
        nonlocal pending_group_lengths
        nonlocal pending_group_secids
        nonlocal pending_group_trade_dates
        nonlocal pending_rows
        nonlocal total_rows
        nonlocal total_groups

        if not pending_x:
            return

        info = save_shard(
            shard_id=shard_id,
            groups_x=pending_x,
            groups_y=pending_y,
            group_lengths=pending_group_lengths,
            group_secids=pending_group_secids,
            group_trade_dates=pending_group_trade_dates,
            output_dir=output_dir,
        )

        shard_infos.append(info)

        total_rows += info["rows"]
        total_groups += info["groups"]

        shard_id += 1
        pending_x = []
        pending_y = []
        pending_group_lengths = []
        pending_group_secids = []
        pending_group_trade_dates = []
        pending_rows = 0

    for secid_idx, secid in enumerate(secids, start=1):
        log(f"[{secid_idx}/{len(secids)}] Читаю обучающие данные для {secid}")

        df = read_train_part_for_secid(secid=secid, train_until=train_until)

        if df.empty:
            log(f"{secid}: нет строк, пропускаю")
            continue

        validate_shard_source_df(df, secid)

        for (group_secid, trade_date), group in df.groupby(
                ["secid", "trade_date"],
                sort=False,
        ):
            if group.empty:
                continue

            group_len = int(len(group))

            x = (
                group[FEATURE_COLS]
                .to_numpy(copy=True)
                .astype("float32", copy=False)
            )
            y = (
                group[MODEL_TARGET_COLS]
                .to_numpy(copy=True)
                .astype("float32", copy=False)
            )

            if pending_rows > 0 and pending_rows + group_len > SEQUENCE_SHARD_MAX_ROWS:
                flush_pending()

            pending_x.append(x)
            pending_y.append(y)
            pending_group_lengths.append(group_len)
            pending_group_secids.append(str(group_secid))
            pending_group_trade_dates.append(str(trade_date))
            pending_rows += group_len

        del df

    flush_pending()

    manifest = {
        "dataset_path": str(DATASET_PATH),
        "dataset_name": normalize_dataset_name(dataset_name),
        "train_until": train_until,
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "feature_cols": FEATURE_COLS,
        "model_target_cols": MODEL_TARGET_COLS,
        "execution_target_cols": EXECUTION_TARGET_COLS,
        "hold_target_cols": HOLD_TARGET_COLS,
        "shard_max_rows": SEQUENCE_SHARD_MAX_ROWS,
        "total_rows": total_rows,
        "total_groups": total_groups,
        "num_shards": len(shard_infos),
        "shards": shard_infos,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log("=" * 80)
    log("Sequence shards подготовлены")
    log(f"dataset_name: {dataset_name}")
    log(f"num_shards: {len(shard_infos)}")
    log(f"total_rows: {total_rows}")
    log(f"total_groups: {total_groups}")
    log(f"manifest: {manifest_path}")

    return output_dir


def build_sequence_train_shards(
        force_rebuild: bool = FORCE_REBUILD_SEQUENCE_SHARDS,
) -> Path:
    """
    Создаёт shards для обычного этапа: train only.
    """
    return build_sequence_shards(
        dataset_name="train",
        train_until=TRAIN_UNTIL,
        force_rebuild=force_rebuild,
    )


def build_sequence_train_valid_shards(
        force_rebuild: bool = FORCE_REBUILD_SEQUENCE_SHARDS,
) -> Path:
    """
    Создаёт shards для финального этапа: train + validation.
    """
    return build_sequence_shards(
        dataset_name="train_valid",
        train_until=VALIDATE_UNTIL,
        force_rebuild=force_rebuild,
    )


def load_manifest(dataset_name: str = "train") -> dict[str, Any]:
    """
    Загружает manifest.json для sequence shard-файлов.
    """
    manifest_path = get_manifest_path(dataset_name)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Не найден manifest: {manifest_path}. "
            "Сначала запусти prepare_sequence_shards.py."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    build_sequence_train_shards(force_rebuild=FORCE_REBUILD_SEQUENCE_SHARDS)
    build_sequence_train_valid_shards(force_rebuild=FORCE_REBUILD_SEQUENCE_SHARDS)


if __name__ == "__main__":
    main()
