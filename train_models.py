from __future__ import annotations

import builtins
import json
import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from constants import (
    HORIZON,
    TRAIN_UNTIL,
    VALIDATE_UNTIL,
    WINDOW_SIZE,
    FINAL_STAGE_NAME,
    VALIDATION_STAGE_NAME,
)

from prepare_sequence_shards import (
    FEATURE_COLS,
    EXECUTION_TARGET_COLS,
    HOLD_TARGET_COLS,
    MODEL_TARGET_COLS,
    build_sequence_shards,
    get_shard_paths,
    load_manifest,
    normalize_dataset_name,
)


DATASET_PATH = Path("minute_dataset.parquet")

MODELS_ROOT = Path("models")
DATA_ROOT = Path("data")

VERBOSE = True


def log(*args: Any) -> None:
    """
    Безопасно печатает диагностическое сообщение с текущим временем.

    Если PyCharm/Windows stdout временно падает с OSError,
    обучение не останавливается.
    """
    if not VERBOSE:
        return

    now = datetime.now().strftime("%H:%M:%S")
    message = " ".join(str(arg) for arg in args)

    try:
        builtins.print(f"[{now}] {message}")
    except OSError:
        pass


# Здесь выбираешь, какие модели обучать и для каких моделей строить valid/test predictions.
# Можно писать названия в любом регистре.
# ML-модели: "ridge", "catboost", "random_forest", "decision_tree".
# Нейросетевые модели: "GRU", "LSTM", "transformer", "TCN".
ML_MODELS_TO_TRAIN = [
    #"ridge",
    #"catboost",
    #"random_forest",
    #"decision_tree",
]

NEURAL_MODELS_TO_TRAIN = [
    #"TCN",
    #"GRU",
    #"LSTM",
    #"transformer",
]

# Единый список нужен только для main(). Переключай модели через два списка выше.
TO_TRAIN = ML_MODELS_TO_TRAIN + NEURAL_MODELS_TO_TRAIN

# ============================================================
# Параметры ML-моделей
# ============================================================

RIDGE_ALPHA = 1.0

# CatBoostRegressor используется как отдельная flat ML-модель.
# Модель умеет multi-target regression через loss_function="MultiRMSE"
# и может использовать GPU.
CATBOOST_ITERATIONS = 2000
CATBOOST_LEARNING_RATE = 0.03
CATBOOST_DEPTH = 7
CATBOOST_L2_LEAF_REG = 10.0

CATBOOST_LOSS_FUNCTION = "MultiRMSE"
CATBOOST_EVAL_METRIC = "MultiRMSE"

CATBOOST_BOOTSTRAP_TYPE = "Bernoulli"
CATBOOST_SUBSAMPLE = 0.8
CATBOOST_RANDOM_STRENGTH = 1.0

CATBOOST_OD_TYPE = "Iter"
CATBOOST_OD_WAIT = 100
CATBOOST_USE_BEST_MODEL = True

CATBOOST_RANDOM_SEED = 46
CATBOOST_VERBOSE = 50

# GPU-режим CatBoost. Для возврата на CPU поменяй на "CPU".
# "0" означает первую CUDA GPU. Для нескольких GPU можно указать "0:1"
# или диапазон вроде "0-1".
CATBOOST_TASK_TYPE = "GPU"
CATBOOST_DEVICES = "0"
CATBOOST_THREAD_COUNT = -1

# RandomForestRegressor умеет multi-output regression сам.
# Ограничиваем число признаков на split и число потоков, чтобы не взрывать RAM.
RANDOM_FOREST_N_ESTIMATORS = 50
RANDOM_FOREST_MAX_DEPTH = 10
RANDOM_FOREST_MIN_SAMPLES_LEAF = 100
RANDOM_FOREST_MAX_FEATURES = "sqrt"
RANDOM_FOREST_RANDOM_SEED = 47
RANDOM_FOREST_N_JOBS = 4

DECISION_TREE_MAX_DEPTH = 8
DECISION_TREE_MIN_SAMPLES_LEAF = 100
DECISION_TREE_RANDOM_SEED = 48

# Глобальный fallback. None означает использовать весь train.
ML_MAX_TRAIN_SAMPLES = None

# Индивидуальные ограничения для тяжёлых flat ML-моделей.
# Ridge оставляем на всём датасете. Для CatBoost на GPU ограничиваем train,
# чтобы не упираться в VRAM. Ограничение берёт самые новые строки по begin,
# а не случайную подвыборку.
ML_MAX_TRAIN_SAMPLES_BY_MODEL = {
    "ridge": None,
    "catboost": 22_000_000,
    "random_forest": None,
    "decision_tree": None,
}

TREE_BASED_MODELS = {
    "catboost",
    "random_forest",
    "decision_tree",
}

# ============================================================
# Общие параметры sequence-моделей
# ============================================================

USE_SEQUENCE_SHARDS = True
FORCE_REBUILD_SEQUENCE_SHARDS = False

# Если .pt-файл sequence-модели уже существует в папке stage,
# обучение продолжится с сохранённых весов, а не начнётся с нуля.
# Важно: число epochs ниже означает ДОПОЛНИТЕЛЬНЫЕ эпохи обучения
# после загрузки существующих весов.
RESUME_SEQUENCE_MODEL_IF_EXISTS = True

# Если в .pt-файле есть optimizer_state_dict, загружаем состояние optimizer.
# Это полноценнее, чем resume только по весам: AdamW продолжит с накопленными
# моментами первого/второго порядка. Старые .pt-файлы без optimizer_state_dict
# всё равно поддерживаются, но optimizer для них начнётся с нуля.
RESUME_SEQUENCE_OPTIMIZER_IF_EXISTS = True

# После загрузки optimizer state PyTorch также восстанавливает lr/weight_decay
# из checkpoint. Обычно удобнее, чтобы текущие значения из этого файла
# оставались главными, поэтому после resume они заново прописываются
# в param_groups optimizer.
KEEP_CURRENT_OPTIMIZER_HYPERPARAMS_ON_RESUME = True

# Если True, checkpoint сохраняется после каждой эпохи. Это полезно для долгого
# обучения: при остановке можно продолжить почти с последней завершённой эпохи.
SAVE_SEQUENCE_CHECKPOINT_EVERY_EPOCH = True

# Если scaler для sequence-модели уже сохранён, используем его повторно.
# Это ускоряет resume и гарантирует, что модель получает признаки в той же шкале.
REUSE_SEQUENCE_SCALER_IF_EXISTS = True

# None означает использовать все train-окна из shard-файлов.
SEQUENCE_MAX_TRAIN_SAMPLES = None
SEQUENCE_NUM_WORKERS = 4

# Масштабирование target для нейросетевых sequence-моделей.
# Модель обучается на доходностях в базисных пунктах,
# а при prediction прогноз делится обратно на этот scale.
SEQUENCE_TARGET_SCALE = 10_000.0

# Weighted loss используем только для Transformer.
# Идея: обычный MSE слишком охотно учит модель предсказывать
# почти среднюю доходность. Поэтому строки с сильным положительным
# execution-target получают больший вес.
TRANSFORMER_USE_WEIGHTED_LOSS = True
TRANSFORMER_POSITIVE_TARGET_WEIGHT = 2.0
TRANSFORMER_POSITIVE_TARGET_CLIP_BP = 12.0
TRANSFORMER_GRAD_CLIP_NORM = 1.0

# ============================================================
# Параметры GRU
# ============================================================

GRU_HIDDEN_SIZE = 320
GRU_NUM_LAYERS = 3
GRU_DROPOUT = 0.05
GRU_BATCH_SIZE = 2048
GRU_EPOCHS = 5
GRU_LEARNING_RATE = 2e-4
GRU_WEIGHT_DECAY = 1e-5
GRU_RANDOM_SEED = 42
GRU_PREDICT_BATCH_SIZE = 16384

# ============================================================
# Параметры LSTM
# ============================================================

LSTM_HIDDEN_SIZE = 320
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.10
LSTM_BATCH_SIZE = 2048
LSTM_EPOCHS = 10
LSTM_LEARNING_RATE = 2e-4
LSTM_WEIGHT_DECAY = 1e-5
LSTM_RANDOM_SEED = 43
LSTM_PREDICT_BATCH_SIZE = 16384

# ============================================================
# Параметры Transformer encoder
# ============================================================

TRANSFORMER_D_MODEL = 192
TRANSFORMER_NHEAD = 6
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_DIM_FEEDFORWARD = 768
TRANSFORMER_DROPOUT = 0.08
TRANSFORMER_BATCH_SIZE = 1536
TRANSFORMER_EPOCHS = 15
TRANSFORMER_LEARNING_RATE = 8e-5
TRANSFORMER_WEIGHT_DECAY = 1e-5
TRANSFORMER_RANDOM_SEED = 44
TRANSFORMER_PREDICT_BATCH_SIZE = 16384

# ============================================================
# Параметры TCN
# ============================================================

TCN_CHANNELS = [192, 192, 192]
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.08
TCN_BATCH_SIZE = 1536
TCN_EPOCHS = 10
TCN_LEARNING_RATE = 1e-4
TCN_WEIGHT_DECAY = 1e-5
TCN_RANDOM_SEED = 49
TCN_PREDICT_BATCH_SIZE = 16384

# ============================================================
# Базовые колонки predictions
# ============================================================

BASE_PREDICTION_COLS = [
    "begin",
    "secid",
    "close",
    "next_open",
]


@dataclass
class ModelArtifacts:
    model_name: str
    model: Any
    scaler: StandardScaler | None
    config: dict[str, Any]


class NeuralRegressorWrapper:
    def __init__(self, model: Any, device: Any):
        self.model = model
        self.device = device


def set_random_seed(seed: int) -> None:
    """
    Фиксирует seed для воспроизводимости.
    """
    random.seed(seed)
    np.random.seed(seed)


def normalize_model_name(model_name: str) -> str:
    """
    Приводит имя модели к каноническому виду.
    """
    normalized_name = model_name.lower()

    aliases = {
        "ridge": "ridge",
        "catboost": "catboost",
        "cb": "catboost",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "decision_tree": "decision_tree",
        "decisiontree": "decision_tree",
        "dt": "decision_tree",
        "gru": "gru",
        "lstm": "lstm",
        "transformer": "transformer",
        "tcn": "tcn",
    }

    if normalized_name in aliases:
        return aliases[normalized_name]

    raise ValueError(
        f"Неизвестная модель: {model_name}. "
        "Допустимые значения: 'ridge', 'catboost', "
        "'random_forest', 'decision_tree', 'GRU', 'LSTM', "
        "'transformer', 'TCN'."
    )

def normalize_stage_name(stage_name: str | None) -> str:
    """
    Приводит имя этапа обучения к безопасному виду для имени папки.
    """
    if stage_name is None or stage_name == "":
        return "default"

    return "".join(
        ch if ch.isalnum() or ch in "_-" else "_"
        for ch in stage_name
    )


def get_model_dir(
        model_name: str,
        stage_name: str | None = None,
) -> Path:
    normalized_model_name = normalize_model_name(model_name)

    if stage_name is None:
        return MODELS_ROOT / normalized_model_name

    return MODELS_ROOT / normalized_model_name / normalize_stage_name(stage_name)


def get_data_dir(model_name: str) -> Path:
    return DATA_ROOT / normalize_model_name(model_name)


def get_model_config_path(
        model_name: str,
        stage_name: str | None = None,
) -> Path:
    return (
        get_model_dir(model_name, stage_name)
        / f"model_config_horizon_{HORIZON}.json"
    )


def validate_dataset(
        df: pd.DataFrame,
        split_name: str,
        need_targets: bool,
) -> None:
    """
    Проверяет наличие колонок и отсутствие NaN/inf.
    """
    required_cols = list(FEATURE_COLS)

    if need_targets:
        required_cols += MODEL_TARGET_COLS

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"В {split_name}-датасете отсутствуют колонки: {missing_cols}"
        )

    feature_nan_rows = df[FEATURE_COLS].isna().any(axis=1).sum()
    if feature_nan_rows > 0:
        raise ValueError(
            f"В {split_name}-датасете найдено {feature_nan_rows} строк "
            f"с NaN в признаках. Проверь build_dataset.py."
        )

    feature_values = df[FEATURE_COLS].to_numpy(copy=False)
    if not np.isfinite(feature_values).all():
        raise ValueError(
            f"В {split_name}-датасете есть inf/-inf в признаках. "
            f"Проверь build_dataset.py."
        )

    if need_targets:
        target_nan_rows = df[MODEL_TARGET_COLS].isna().any(axis=1).sum()
        if target_nan_rows > 0:
            raise ValueError(
                f"В {split_name}-датасете найдено {target_nan_rows} строк "
                f"с NaN в целевых колонках. Проверь build_dataset.py."
            )

        target_values = df[MODEL_TARGET_COLS].to_numpy(copy=False)
        if not np.isfinite(target_values).all():
            raise ValueError(
                f"В {split_name}-датасете есть inf/-inf в целевых колонках. "
                f"Проверь build_dataset.py."
            )


def read_train_dataset(
        train_until: str = TRAIN_UNTIL,
        split_name: str = "train",
) -> pd.DataFrame:
    """
    Читает обучающую часть датасета целиком.

    Используется для flat ML-моделей.
    Sequence-модели обучаются по shard-файлам.
    """
    columns = [
        "begin",
        "secid",
        "trade_date",
        "close",
        "next_open",
    ] + FEATURE_COLS + MODEL_TARGET_COLS

    log(f"Читаю обучающую часть датасета: begin < {train_until}")

    df = pd.read_parquet(
        DATASET_PATH,
        columns=columns,
        filters=[("begin", "<", pd.Timestamp(train_until))],
    )

    validate_dataset(df, split_name=split_name, need_targets=True)

    return df.reset_index(drop=True)


def read_prediction_split(split_name: str) -> pd.DataFrame:
    """
    Читает valid/test часть датасета для построения прогнозов.
    """
    columns = BASE_PREDICTION_COLS + [
        "trade_date",
    ] + FEATURE_COLS

    if split_name == "valid":
        filters = [
            ("begin", ">=", pd.Timestamp(TRAIN_UNTIL)),
            ("begin", "<", pd.Timestamp(VALIDATE_UNTIL)),
        ]
    elif split_name == "test":
        filters = [
            ("begin", ">=", pd.Timestamp(VALIDATE_UNTIL)),
        ]
    else:
        raise ValueError(f"Неизвестная выборка: {split_name}")

    log(f"Читаю {split_name}-часть датасета для построения прогнозов")

    df = pd.read_parquet(
        DATASET_PATH,
        columns=columns,
        filters=filters,
    )

    validate_dataset(df, split_name=split_name, need_targets=False)

    return df.reset_index(drop=True)


def calculate_score_matrix(
        forecast_matrix: np.ndarray,
        score_mode: str,
        top_p: int = 1,
        top_m: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразует прогнозы по горизонтам в score и выбранный горизонт.
    """
    if forecast_matrix.ndim != 2:
        raise ValueError("forecast_matrix должен быть двумерным массивом")

    n_horizons = forecast_matrix.shape[1]
    order = np.argsort(forecast_matrix, axis=1)[:, ::-1]

    if score_mode == "top_p":
        p_idx = min(max(top_p, 1), n_horizons) - 1
        selected_idx = order[:, p_idx]

        score = forecast_matrix[
            np.arange(len(forecast_matrix)),
            selected_idx,
        ]

        decision_horizon = selected_idx + 1

        return score, decision_horizon

    if score_mode == "top_mean":
        m = min(max(top_m, 1), n_horizons)
        top_indices = order[:, :m]

        row_indices = np.arange(len(forecast_matrix))[:, None]
        top_values = forecast_matrix[row_indices, top_indices]

        score = top_values.mean(axis=1)

        best_idx = order[:, 0]
        decision_horizon = best_idx + 1

        return score, decision_horizon

    raise ValueError(f"Неизвестный score_mode: {score_mode}")


def add_predictions_to_dataframe(
        df: pd.DataFrame,
        pred: np.ndarray,
) -> pd.DataFrame:
    """
    Добавляет forecasts и score-колонки в DataFrame для backtest.
    """
    if pred.ndim != 2:
        raise ValueError("Ожидался двумерный массив прогнозов")

    if pred.shape[1] != len(MODEL_TARGET_COLS):
        raise ValueError(
            f"Число выходов модели {pred.shape[1]} не совпадает "
            f"с числом целевых колонок {len(MODEL_TARGET_COLS)}"
        )

    result = df[BASE_PREDICTION_COLS].copy()

    execution_count = len(EXECUTION_TARGET_COLS)
    hold_count = len(HOLD_TARGET_COLS)

    execution_pred = pred[:, :execution_count]
    hold_pred = pred[:, execution_count:execution_count + hold_count]

    for idx, _col in enumerate(EXECUTION_TARGET_COLS, start=1):
        result[f"execution_forecast_{idx}"] = (
            execution_pred[:, idx - 1].astype("float32")
        )

    for idx, _col in enumerate(HOLD_TARGET_COLS, start=1):
        result[f"hold_forecast_{idx}"] = (
            hold_pred[:, idx - 1].astype("float32")
        )

    open_score, open_horizon = calculate_score_matrix(
        forecast_matrix=execution_pred,
        score_mode="top_p",
        top_p=1,
        top_m=3,
    )

    hold_score, hold_horizon = calculate_score_matrix(
        forecast_matrix=hold_pred,
        score_mode="top_p",
        top_p=1,
        top_m=3,
    )

    result["open_score"] = open_score.astype("float32")
    result["open_horizon"] = open_horizon.astype("int8")
    result["hold_score"] = hold_score.astype("float32")
    result["hold_horizon"] = hold_horizon.astype("int8")

    return result


def save_prediction_dataset(
        model_name: str,
        split_name: str,
        prediction_df: pd.DataFrame,
) -> None:
    """
    Сохраняет файл прогнозов модели.
    """
    data_dir = get_data_dir(model_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    path = data_dir / f"{split_name}_predictions.parquet"
    prediction_df.to_parquet(path, index=False)

    log(f"{model_name}: {split_name} predictions сохранены: {path}")
    log(f"{model_name}: {split_name} predictions shape: {prediction_df.shape}")


def make_base_config(
        model_name: str,
        train_until: str,
        stage_name: str,
) -> dict[str, Any]:
    """
    Создаёт базовый config модели.
    """
    normalized_name = normalize_model_name(model_name)

    return {
        "model_name": normalized_name,
        "stage_name": normalize_stage_name(stage_name),
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "train_until": train_until,
        "validate_until": VALIDATE_UNTIL,
        "target_type": "multi_output_execution_and_hold_return",
        "execution_target_cols": EXECUTION_TARGET_COLS,
        "hold_target_cols": HOLD_TARGET_COLS,
        "model_target_cols": MODEL_TARGET_COLS,
        "feature_cols": FEATURE_COLS,
        "prediction_data_dir": str(get_data_dir(normalized_name)),
    }


def get_ml_max_train_samples(model_name: str) -> int | None:
    """
    Возвращает ограничение обучающей выборки для конкретной flat ML-модели.
    """
    normalized_name = normalize_model_name(model_name)
    return ML_MAX_TRAIN_SAMPLES_BY_MODEL.get(
        normalized_name,
        ML_MAX_TRAIN_SAMPLES,
    )


def sample_train_df_for_ml(train_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    При необходимости ограничивает размер train для тяжёлых ML-моделей.

    Важно: ограничение берёт не случайные строки, а последние max_rows строк
    по времени begin. Это лучше соответствует временной природе задачи:
    модель обучается на самой свежей доступной части train-периода.
    """
    max_samples = get_ml_max_train_samples(model_name)

    if max_samples is None:
        log(f"{model_name}: обучаюсь на всём train: {len(train_df)} строк")
        return train_df

    max_rows = int(max_samples)
    if max_rows <= 0:
        raise ValueError(
            f"{model_name}: ML_MAX_TRAIN_SAMPLES должен быть положительным "
            f"числом или None, получено: {max_samples}"
        )

    if len(train_df) <= max_rows:
        log(
            f"{model_name}: размер train {len(train_df)} <= лимита {max_rows}, "
            "ограничение train не требуется"
        )
        return train_df.sort_values(["begin", "secid"]).reset_index(drop=True)

    log(
        f"{model_name}: ML_MAX_TRAIN_SAMPLES={max_rows}, "
        f"беру последние {max_rows} строк из {len(train_df)} по begin"
    )

    return (
        train_df
        .sort_values(["begin", "secid"], kind="mergesort")
        .tail(max_rows)
        .reset_index(drop=True)
    )

def create_ml_model(model_name: str) -> Any:
    """
    Создаёт одну из flat ML-моделей.
    """
    normalized_name = normalize_model_name(model_name)

    if normalized_name == "ridge":
        return Ridge(alpha=RIDGE_ALPHA)

    if normalized_name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError(
                "Для модели catboost нужен пакет CatBoost. "
                "Установи его командой: pip install catboost"
            )

        task_type = CATBOOST_TASK_TYPE.upper()

        catboost_params = {
            "loss_function": CATBOOST_LOSS_FUNCTION,
            "eval_metric": CATBOOST_EVAL_METRIC,
            "iterations": CATBOOST_ITERATIONS,
            "learning_rate": CATBOOST_LEARNING_RATE,
            "depth": CATBOOST_DEPTH,
            "l2_leaf_reg": CATBOOST_L2_LEAF_REG,
            "bootstrap_type": CATBOOST_BOOTSTRAP_TYPE,
            "subsample": CATBOOST_SUBSAMPLE,
            "task_type": task_type,
            "random_seed": CATBOOST_RANDOM_SEED,
            "verbose": CATBOOST_VERBOSE,
            "allow_writing_files": False,
        }

        if task_type == "GPU":
            catboost_params["devices"] = CATBOOST_DEVICES
        elif task_type == "CPU":
            catboost_params["thread_count"] = CATBOOST_THREAD_COUNT

            # random_strength в документации CatBoost указан как CPU-only
            # параметр, поэтому не передаём его при GPU-обучении.
            catboost_params["random_strength"] = CATBOOST_RANDOM_STRENGTH
        else:
            raise ValueError(
                "CATBOOST_TASK_TYPE должен быть 'CPU' или 'GPU', "
                f"получено: {CATBOOST_TASK_TYPE!r}"
            )

        return CatBoostRegressor(**catboost_params)

    if normalized_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            max_features=RANDOM_FOREST_MAX_FEATURES,
            random_state=RANDOM_FOREST_RANDOM_SEED,
            n_jobs=RANDOM_FOREST_N_JOBS,
        )

    if normalized_name == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=DECISION_TREE_MAX_DEPTH,
            min_samples_leaf=DECISION_TREE_MIN_SAMPLES_LEAF,
            random_state=DECISION_TREE_RANDOM_SEED,
        )

    raise ValueError(f"Не ML-модель: {model_name}")

def get_ml_model_config(model_name: str) -> dict[str, Any]:
    """
    Возвращает config с гиперпараметрами ML-модели.
    """
    normalized_name = normalize_model_name(model_name)

    if normalized_name == "ridge":
        return {
            "model_type": "Ridge",
            "alpha": RIDGE_ALPHA,
        }

    if normalized_name == "catboost":
        return {
            "model_type": "CatBoostRegressor_MultiRMSE",
            "iterations": CATBOOST_ITERATIONS,
            "learning_rate": CATBOOST_LEARNING_RATE,
            "depth": CATBOOST_DEPTH,
            "l2_leaf_reg": CATBOOST_L2_LEAF_REG,
            "loss_function": CATBOOST_LOSS_FUNCTION,
            "eval_metric": CATBOOST_EVAL_METRIC,
            "bootstrap_type": CATBOOST_BOOTSTRAP_TYPE,
            "subsample": CATBOOST_SUBSAMPLE,
            "task_type": CATBOOST_TASK_TYPE,
            "devices": CATBOOST_DEVICES if CATBOOST_TASK_TYPE.upper() == "GPU" else None,
            "thread_count": CATBOOST_THREAD_COUNT if CATBOOST_TASK_TYPE.upper() == "CPU" else None,
            "random_seed": CATBOOST_RANDOM_SEED,
            "verbose": CATBOOST_VERBOSE,
        }

    if normalized_name == "random_forest":
        return {
            "model_type": "RandomForestRegressor",
            "n_estimators": RANDOM_FOREST_N_ESTIMATORS,
            "max_depth": RANDOM_FOREST_MAX_DEPTH,
            "min_samples_leaf": RANDOM_FOREST_MIN_SAMPLES_LEAF,
            "max_features": RANDOM_FOREST_MAX_FEATURES,
            "random_seed": RANDOM_FOREST_RANDOM_SEED,
            "n_jobs": RANDOM_FOREST_N_JOBS,
        }

    if normalized_name == "decision_tree":
        return {
            "model_type": "DecisionTreeRegressor",
            "max_depth": DECISION_TREE_MAX_DEPTH,
            "min_samples_leaf": DECISION_TREE_MIN_SAMPLES_LEAF,
            "random_seed": DECISION_TREE_RANDOM_SEED,
        }

    raise ValueError(f"Не ML-модель: {model_name}")

def train_ml_model(
        model_name: str,
        train_df: pd.DataFrame,
        train_until: str,
        stage_name: str,
) -> ModelArtifacts:
    """
    Обучает flat ML-модель на текущих строковых признаках FEATURE_COLS.
    """
    normalized_name = normalize_model_name(model_name)
    model_dir = get_model_dir(normalized_name, stage_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{normalized_name}_horizon_{HORIZON}.joblib"
    scaler_path = model_dir / f"scaler_horizon_{HORIZON}.joblib"
    config_path = get_model_config_path(normalized_name, stage_name)

    train_df = sample_train_df_for_ml(train_df, normalized_name)

    log(f"{normalized_name}: готовлю матрицу признаков и целевых значений")
    x_train = (
        train_df[FEATURE_COLS]
        .to_numpy(copy=False)
        .astype("float32", copy=False)
    )
    y_train = (
        train_df[MODEL_TARGET_COLS]
        .to_numpy(copy=False)
        .astype("float32", copy=False)
    )

    log(f"{normalized_name}: X_train shape:", x_train.shape)
    log(f"{normalized_name}: y_train shape:", y_train.shape)

    scaler: StandardScaler | None = None

    if normalized_name in TREE_BASED_MODELS:
        # Деревьям масштабирование не нужно. Так мы не держим лишнюю копию X_train_scaled
        # и не тратим память на StandardScaler.
        log(f"{normalized_name}: scaler не используется для tree-based модели")
        x_train_model = x_train
    else:
        scaler = StandardScaler(copy=False)
        log(f"{normalized_name}: нормализую признаки")
        x_train_model = scaler.fit_transform(x_train).astype("float32", copy=False)

    model = create_ml_model(normalized_name)

    if normalized_name == "catboost":
        log(
            f"{normalized_name}: CatBoost task_type={CATBOOST_TASK_TYPE}, "
            f"devices={CATBOOST_DEVICES if CATBOOST_TASK_TYPE.upper() == 'GPU' else 'CPU'}"
        )

    log(f"{normalized_name}: обучаю модель")
    model.fit(x_train_model, y_train)

    joblib.dump(model, model_path)

    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        saved_scaler_path: str | None = str(scaler_path)
        log(f"{normalized_name}: scaler сохранён:", scaler_path)
    else:
        saved_scaler_path = None

    config = make_base_config(
        model_name=normalized_name,
        train_until=train_until,
        stage_name=stage_name,
    )
    config.update(get_ml_model_config(normalized_name))
    config.update({
        "model_path": str(model_path),
        "scaler_path": saved_scaler_path,
        "ml_max_train_samples": get_ml_max_train_samples(normalized_name),
        "ml_train_sample_strategy": "latest_rows_by_begin",
    })

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    log(f"{normalized_name}: модель сохранена:", model_path)
    log(f"{normalized_name}: config сохранён:", config_path)

    return ModelArtifacts(
        model_name=normalized_name,
        model=model,
        scaler=scaler,
        config=config,
    )

def predict_with_ml_model(
        artifacts: ModelArtifacts,
        split_df: pd.DataFrame,
) -> np.ndarray:
    """
    Строит прогнозы flat ML-модели.
    """
    x = (
        split_df[FEATURE_COLS]
        .to_numpy(copy=False)
        .astype("float32", copy=False)
    )

    if artifacts.scaler is None:
        x_model = x
    else:
        x_model = artifacts.scaler.transform(x).astype("float32", copy=False)

    pred = artifacts.model.predict(x_model)
    return np.asarray(pred, dtype="float32")


class SequenceIndexDataset:
    def __init__(
            self,
            groups_x: list[np.ndarray],
            groups_y: list[np.ndarray],
            window_size: int,
            sample_indices: np.ndarray | None = None,
    ) -> None:
        self.groups_x = groups_x
        self.groups_y = groups_y
        self.window_size = window_size
        self.group_lengths = np.array([len(x) for x in groups_x], dtype=np.int64)
        self.cumulative_lengths = np.cumsum(self.group_lengths)

        if sample_indices is None:
            self.sample_indices = None
            self.length = int(self.cumulative_lengths[-1])
        else:
            self.sample_indices = sample_indices.astype(np.int64, copy=False)
            self.length = int(len(self.sample_indices))

    def __len__(self) -> int:
        return self.length

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if self.sample_indices is not None:
            global_idx = int(self.sample_indices[idx])
        else:
            global_idx = int(idx)

        group_idx = int(
            np.searchsorted(self.cumulative_lengths, global_idx, side="right")
        )
        group_start = (
            0
            if group_idx == 0
            else int(self.cumulative_lengths[group_idx - 1])
        )
        local_idx = global_idx - group_start

        return group_idx, local_idx

    def __getitem__(self, idx: int):
        import torch

        group_idx, local_idx = self._resolve_index(idx)
        x_group = self.groups_x[group_idx]
        y_group = self.groups_y[group_idx]

        start = local_idx - self.window_size + 1

        if start >= 0:
            window = x_group[start:local_idx + 1]
        else:
            pad_count = -start
            pad = np.repeat(x_group[0:1], pad_count, axis=0)
            window = np.concatenate([pad, x_group[0:local_idx + 1]], axis=0)

        y = y_group[local_idx]

        return (
            torch.from_numpy(window.astype("float32", copy=False)),
            torch.from_numpy(y.astype("float32", copy=False)),
        )


def prepare_sequence_groups(
        df: pd.DataFrame,
        scaler: StandardScaler,
        need_targets: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Готовит последовательные группы для prediction.
    В обучении sequence-моделей используется sharded loader.
    """
    sort_cols = ["secid", "trade_date", "begin"]
    df = df.sort_values(sort_cols, kind="mergesort")

    groups_x = []
    groups_y = []

    for _key, group in df.groupby(["secid", "trade_date"], sort=False):
        x = group[FEATURE_COLS].to_numpy(copy=False).astype("float32", copy=False)
        x = scaler.transform(x).astype("float32", copy=False)
        groups_x.append(x)

        if need_targets:
            y = (
                group[MODEL_TARGET_COLS]
                .to_numpy(copy=False)
                .astype("float32", copy=False)
            )
        else:
            y = np.zeros((len(group), len(MODEL_TARGET_COLS)), dtype="float32")

        groups_y.append(y)

    return groups_x, groups_y


def load_sequence_shard(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает один sequence shard-файл.
    """
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype("float32", copy=False)
        y = data["y"].astype("float32", copy=False)
        group_lengths = data["group_lengths"].astype(np.int64, copy=False)

    return x, y, group_lengths


def fit_scaler_from_shards(shard_paths: list[Path]) -> StandardScaler:
    """
    Обучает StandardScaler по shard-файлам без загрузки всего train в RAM.
    """
    scaler = StandardScaler(copy=False)

    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        log(
            "sequence: scaler partial_fit "
            f"shard {shard_idx}/{len(shard_paths)}: {shard_path.name}"
        )

        x, _y, _group_lengths = load_sequence_shard(shard_path)
        scaler.partial_fit(x)

        del x
        del _y
        del _group_lengths

    return scaler


def load_or_fit_sequence_scaler(
        shard_paths: list[Path],
        scaler_path: Path,
) -> StandardScaler:
    """
    Загружает существующий scaler для sequence-модели или обучает новый.

    При продолжении обучения важно использовать тот же scaler, с которым
    модель обучалась раньше. Если scaler отсутствует, он строится по shard-файлам.
    """
    if REUSE_SEQUENCE_SCALER_IF_EXISTS and scaler_path.exists():
        log("sequence: загружаю существующий scaler:", scaler_path)
        scaler = joblib.load(scaler_path)

        if not isinstance(scaler, StandardScaler):
            raise TypeError(
                f"В файле {scaler_path} ожидался StandardScaler, "
                f"получен объект типа {type(scaler).__name__}."
            )

        return scaler

    log("sequence: существующий scaler не найден, обучаю новый")
    return fit_scaler_from_shards(shard_paths)


def set_optimizer_hyperparams_from_config(
        optimizer: Any,
        hp: dict[str, Any],
) -> None:
    """
    Принудительно выставляет lr/weight_decay из текущего train_models.py.

    Это нужно после optimizer.load_state_dict(...), потому что PyTorch
    восстанавливает param_groups из checkpoint и может перезаписать текущие
    значения learning_rate/weight_decay старыми значениями.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(hp["learning_rate"])
        param_group["weight_decay"] = float(hp["weight_decay"])


def load_existing_sequence_checkpoint_if_available(
        model: Any,
        optimizer: Any,
        model_path: Path,
        device: Any,
        model_name: str,
        hp: dict[str, Any],
) -> dict[str, Any]:
    """
    Если .pt-файл модели уже существует, загружает checkpoint.

    Поддерживаются два формата:
    1. старый формат: обычный model.state_dict();
    2. новый формат: checkpoint-словарь с ключами model_state_dict
       и, по возможности, optimizer_state_dict.

    Возвращает служебную информацию о resume.
    """
    resume_info = {
        "resumed_from_existing_model": False,
        "resumed_optimizer_state": False,
        "checkpoint_format": "none",
        "previous_completed_epochs": 0,
    }

    if not RESUME_SEQUENCE_MODEL_IF_EXISTS:
        return resume_info

    if not model_path.exists():
        log(f"{model_name}: существующая .pt-модель не найдена, обучаю с нуля")
        return resume_info

    import torch

    log(f"{model_name}: найдена существующая .pt-модель:", model_path)
    log(f"{model_name}: загружаю checkpoint и продолжаю обучение")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        resume_info["checkpoint_format"] = "checkpoint"
        resume_info["previous_completed_epochs"] = int(
            checkpoint.get("completed_epochs", 0)
        )
    else:
        # Backward compatibility со старыми файлами, где в .pt лежал только
        # model.state_dict(). В этом случае веса загрузятся, но optimizer
        # придётся начать заново.
        state_dict = checkpoint
        resume_info["checkpoint_format"] = "plain_state_dict"

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Не удалось загрузить веса из {model_path}. "
            "Скорее всего, изменилась архитектура модели "
            "(hidden_size, num_layers, d_model, nhead, WINDOW_SIZE и т.п.). "
            "Либо верни старые параметры архитектуры, либо удали/переименуй "
            "старый .pt-файл, чтобы начать обучение с нуля."
        ) from exc

    resume_info["resumed_from_existing_model"] = True

    if (
            RESUME_SEQUENCE_OPTIMIZER_IF_EXISTS
            and isinstance(checkpoint, dict)
            and "optimizer_state_dict" in checkpoint
    ):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            resume_info["resumed_optimizer_state"] = True
            log(f"{model_name}: optimizer state загружен из checkpoint")

            if KEEP_CURRENT_OPTIMIZER_HYPERPARAMS_ON_RESUME:
                set_optimizer_hyperparams_from_config(
                    optimizer=optimizer,
                    hp=hp,
                )
                log(
                    f"{model_name}: lr/weight_decay после resume взяты "
                    "из текущего train_models.py"
                )
        except ValueError as exc:
            raise RuntimeError(
                f"Не удалось загрузить optimizer state из {model_path}. "
                "Скорее всего, изменилась архитектура модели или состав "
                "параметров. Можно удалить/переименовать .pt-файл, чтобы "
                "начать обучение с нуля, или отключить "
                "RESUME_SEQUENCE_OPTIMIZER_IF_EXISTS."
            ) from exc
    else:
        if resume_info["checkpoint_format"] == "plain_state_dict":
            log(
                f"{model_name}: старый .pt содержит только веса; "
                "optimizer начнётся с нуля"
            )
        else:
            log(
                f"{model_name}: optimizer_state_dict в checkpoint не найден; "
                "optimizer начнётся с нуля"
            )

    return resume_info


def build_sequence_checkpoint(
        model: Any,
        optimizer: Any,
        normalized_name: str,
        stage_name: str,
        train_until: str,
        hp: dict[str, Any],
        completed_epochs: int,
        last_mean_loss: float | None,
) -> dict[str, Any]:
    """
    Собирает checkpoint для сохранения sequence-модели.
    """
    return {
        "checkpoint_format": "sequence_model_checkpoint_v1",
        "model_name": normalized_name,
        "stage_name": normalize_stage_name(stage_name),
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "train_until": train_until,
        "validate_until": VALIDATE_UNTIL,
        "completed_epochs": int(completed_epochs),
        "last_run_epochs": int(hp["epochs"]),
        "last_mean_loss": last_mean_loss,
        "sequence_target_scale": SEQUENCE_TARGET_SCALE,
        "sequence_max_train_samples": SEQUENCE_MAX_TRAIN_SAMPLES,
        "feature_cols": FEATURE_COLS,
        "model_target_cols": MODEL_TARGET_COLS,
        "sequence_hyperparams": hp,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def save_sequence_checkpoint(
        model: Any,
        optimizer: Any,
        model_path: Path,
        normalized_name: str,
        stage_name: str,
        train_until: str,
        hp: dict[str, Any],
        completed_epochs: int,
        last_mean_loss: float | None,
) -> None:
    """
    Сохраняет model state + optimizer state в .pt-файл.
    """
    import torch

    checkpoint = build_sequence_checkpoint(
        model=model,
        optimizer=optimizer,
        normalized_name=normalized_name,
        stage_name=stage_name,
        train_until=train_until,
        hp=hp,
        completed_epochs=completed_epochs,
        last_mean_loss=last_mean_loss,
    )

    torch.save(checkpoint, model_path)



def make_sequence_window_batch(
        x: np.ndarray,
        y: np.ndarray,
        group_lengths: np.ndarray,
        row_indices: np.ndarray,
        window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Собирает batch окон внутри одного shard.
    Окна не пересекают границы secid/trade_date.
    """
    cumulative_lengths = np.cumsum(group_lengths)
    group_starts = np.concatenate([
        np.array([0], dtype=np.int64),
        cumulative_lengths[:-1],
    ])

    batch_size = int(len(row_indices))
    n_features = int(x.shape[1])
    n_targets = int(y.shape[1])

    x_batch = np.empty(
        (batch_size, window_size, n_features),
        dtype="float32",
    )
    y_batch = np.empty(
        (batch_size, n_targets),
        dtype="float32",
    )

    for out_idx, row_idx_value in enumerate(row_indices):
        row_idx = int(row_idx_value)

        group_idx = int(
            np.searchsorted(cumulative_lengths, row_idx, side="right")
        )
        group_start = int(group_starts[group_idx])
        local_idx = row_idx - group_start

        start_local = local_idx - window_size + 1

        if start_local >= 0:
            start_abs = group_start + start_local
            window = x[start_abs:row_idx + 1]
        else:
            pad_count = -start_local
            pad = np.repeat(x[group_start:group_start + 1], pad_count, axis=0)
            window = np.concatenate([pad, x[group_start:row_idx + 1]], axis=0)

        x_batch[out_idx] = window
        y_batch[out_idx] = y[row_idx]

    return x_batch, y_batch


def create_sequence_model(
        model_name: str,
        input_size: int,
        output_size: int,
):
    """
    Создаёт GRU/LSTM/Transformer/TCN-регрессор.
    """
    import torch
    from torch import nn

    normalized_name = normalize_model_name(model_name)

    if normalized_name == "gru":
        class GruRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=GRU_HIDDEN_SIZE,
                    num_layers=GRU_NUM_LAYERS,
                    batch_first=True,
                    dropout=GRU_DROPOUT if GRU_NUM_LAYERS > 1 else 0.0,
                )
                self.head = nn.Linear(GRU_HIDDEN_SIZE, output_size)

            def forward(self, x):
                out, _hidden = self.gru(x)
                last = out[:, -1, :]
                return self.head(last)

        return GruRegressor()

    if normalized_name == "lstm":
        class LstmRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=LSTM_HIDDEN_SIZE,
                    num_layers=LSTM_NUM_LAYERS,
                    batch_first=True,
                    dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
                )
                self.head = nn.Linear(LSTM_HIDDEN_SIZE, output_size)

            def forward(self, x):
                out, _hidden = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last)

        return LstmRegressor()

    if normalized_name == "transformer":
        class TransformerRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.input_projection = nn.Linear(
                    input_size,
                    TRANSFORMER_D_MODEL,
                )

                self.position_embedding = nn.Parameter(
                    torch.empty(1, WINDOW_SIZE, TRANSFORMER_D_MODEL)
                )
                nn.init.normal_(
                    self.position_embedding,
                    mean=0.0,
                    std=0.02,
                )

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=TRANSFORMER_D_MODEL,
                    nhead=TRANSFORMER_NHEAD,
                    dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                    dropout=TRANSFORMER_DROPOUT,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )

                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=TRANSFORMER_NUM_LAYERS,
                    enable_nested_tensor=False,
                )

                self.head = nn.Sequential(
                    nn.LayerNorm(TRANSFORMER_D_MODEL),
                    nn.Linear(TRANSFORMER_D_MODEL, TRANSFORMER_D_MODEL),
                    nn.GELU(),
                    nn.Dropout(TRANSFORMER_DROPOUT),
                    nn.Linear(TRANSFORMER_D_MODEL, output_size),
                )

            def forward(self, x):
                x = self.input_projection(x)
                x = x + self.position_embedding[:, :x.shape[1], :]
                out = self.encoder(x)
                last = out[:, -1, :]
                return self.head(last)

        return TransformerRegressor()

    if normalized_name == "tcn":
        class TemporalBlock(nn.Module):
            def __init__(
                    self,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: int,
                    dilation: int,
                    dropout: float,
            ) -> None:
                super().__init__()
                padding = (kernel_size - 1) * dilation

                self.conv1 = nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
                self.conv2 = nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                self.downsample = (
                    nn.Conv1d(in_channels, out_channels, kernel_size=1)
                    if in_channels != out_channels
                    else None
                )

            def chomp(self, x, padding: int):
                if padding == 0:
                    return x
                return x[:, :, :-padding]

            def forward(self, x):
                padding = (TCN_KERNEL_SIZE - 1) * self.conv1.dilation[0]
                out = self.conv1(x)
                out = self.chomp(out, padding)
                out = self.activation(out)
                out = self.dropout(out)

                out = self.conv2(out)
                out = self.chomp(out, padding)
                out = self.activation(out)
                out = self.dropout(out)

                residual = x if self.downsample is None else self.downsample(x)
                return out + residual

        class TcnRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers = []
                in_channels = input_size

                for layer_idx, out_channels in enumerate(TCN_CHANNELS):
                    dilation = 2 ** layer_idx
                    layers.append(
                        TemporalBlock(
                            in_channels=in_channels,
                            out_channels=int(out_channels),
                            kernel_size=TCN_KERNEL_SIZE,
                            dilation=dilation,
                            dropout=TCN_DROPOUT,
                        )
                    )
                    in_channels = int(out_channels)

                self.network = nn.Sequential(*layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(int(TCN_CHANNELS[-1])),
                    nn.Linear(int(TCN_CHANNELS[-1]), int(TCN_CHANNELS[-1])),
                    nn.GELU(),
                    nn.Dropout(TCN_DROPOUT),
                    nn.Linear(int(TCN_CHANNELS[-1]), output_size),
                )

            def forward(self, x):
                # Conv1d ожидает [batch, features, time].
                x = x.transpose(1, 2)
                out = self.network(x)
                last = out[:, :, -1]
                return self.head(last)

        return TcnRegressor()

    raise ValueError(f"Не sequence-модель: {model_name}")


def get_sequence_hyperparams(model_name: str) -> dict[str, Any]:
    """
    Возвращает training hyperparameters для sequence-модели.
    """
    normalized_name = normalize_model_name(model_name)

    if normalized_name == "gru":
        return {
            "batch_size": GRU_BATCH_SIZE,
            "epochs": GRU_EPOCHS,
            "learning_rate": GRU_LEARNING_RATE,
            "weight_decay": GRU_WEIGHT_DECAY,
            "random_seed": GRU_RANDOM_SEED,
            "predict_batch_size": GRU_PREDICT_BATCH_SIZE,
            "model_type": "GRU",
            "hidden_size": GRU_HIDDEN_SIZE,
            "num_layers": GRU_NUM_LAYERS,
            "dropout": GRU_DROPOUT,
        }

    if normalized_name == "lstm":
        return {
            "batch_size": LSTM_BATCH_SIZE,
            "epochs": LSTM_EPOCHS,
            "learning_rate": LSTM_LEARNING_RATE,
            "weight_decay": LSTM_WEIGHT_DECAY,
            "random_seed": LSTM_RANDOM_SEED,
            "predict_batch_size": LSTM_PREDICT_BATCH_SIZE,
            "model_type": "LSTM",
            "hidden_size": LSTM_HIDDEN_SIZE,
            "num_layers": LSTM_NUM_LAYERS,
            "dropout": LSTM_DROPOUT,
        }

    if normalized_name == "transformer":
        return {
            "batch_size": TRANSFORMER_BATCH_SIZE,
            "epochs": TRANSFORMER_EPOCHS,
            "learning_rate": TRANSFORMER_LEARNING_RATE,
            "weight_decay": TRANSFORMER_WEIGHT_DECAY,
            "random_seed": TRANSFORMER_RANDOM_SEED,
            "predict_batch_size": TRANSFORMER_PREDICT_BATCH_SIZE,
            "model_type": "TransformerEncoder",
            "d_model": TRANSFORMER_D_MODEL,
            "nhead": TRANSFORMER_NHEAD,
            "num_layers": TRANSFORMER_NUM_LAYERS,
            "dim_feedforward": TRANSFORMER_DIM_FEEDFORWARD,
            "dropout": TRANSFORMER_DROPOUT,
            "use_weighted_loss": TRANSFORMER_USE_WEIGHTED_LOSS,
            "positive_target_weight": TRANSFORMER_POSITIVE_TARGET_WEIGHT,
            "positive_target_clip_bp": TRANSFORMER_POSITIVE_TARGET_CLIP_BP,
            "grad_clip_norm": TRANSFORMER_GRAD_CLIP_NORM,
        }

    if normalized_name == "tcn":
        return {
            "batch_size": TCN_BATCH_SIZE,
            "epochs": TCN_EPOCHS,
            "learning_rate": TCN_LEARNING_RATE,
            "weight_decay": TCN_WEIGHT_DECAY,
            "random_seed": TCN_RANDOM_SEED,
            "predict_batch_size": TCN_PREDICT_BATCH_SIZE,
            "model_type": "TCN",
            "channels": TCN_CHANNELS,
            "kernel_size": TCN_KERNEL_SIZE,
            "dropout": TCN_DROPOUT,
        }

    raise ValueError(f"Не sequence-модель: {model_name}")


def train_sequence_model(
        model_name: str,
        train_until: str,
        stage_name: str,
) -> ModelArtifacts:
    """
    Обучает GRU/LSTM/Transformer/TCN через shard-файлы, не загружая весь train в память.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Для обучения нейросетевых sequence-моделей нужен PyTorch."
        ) from exc

    normalized_name = normalize_model_name(model_name)
    hp = get_sequence_hyperparams(normalized_name)

    model_dir = get_model_dir(normalized_name, stage_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{normalized_name}_horizon_{HORIZON}.pt"
    scaler_path = model_dir / f"scaler_horizon_{HORIZON}.joblib"
    config_path = get_model_config_path(normalized_name, stage_name)

    set_random_seed(int(hp["random_seed"]))
    torch.manual_seed(int(hp["random_seed"]))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(hp["random_seed"]))

    shard_dataset_name = normalize_dataset_name(stage_name)

    if USE_SEQUENCE_SHARDS:
        build_sequence_shards(
            dataset_name=shard_dataset_name,
            train_until=train_until,
            force_rebuild=FORCE_REBUILD_SEQUENCE_SHARDS,
        )
    else:
        raise ValueError(
            "В этой версии нейросетевые sequence-модели обучаются через shard-файлы. "
            "Оставь USE_SEQUENCE_SHARDS = True."
        )

    manifest = load_manifest(dataset_name=shard_dataset_name)
    shard_paths = get_shard_paths(dataset_name=shard_dataset_name)

    if not shard_paths:
        raise FileNotFoundError(
            "Sequence shard-файлы не найдены. Запусти prepare_sequence_shards.py."
        )

    log(f"{normalized_name}: manifest total_rows:", manifest["total_rows"])
    log(f"{normalized_name}: manifest num_shards:", manifest["num_shards"])

    log(f"{normalized_name}: обучаю scaler по shard-файлам")
    log(f"{normalized_name}: sequence_target_scale:", SEQUENCE_TARGET_SCALE)

    if normalized_name == "transformer":
        log(
            f"{normalized_name}: weighted_loss={TRANSFORMER_USE_WEIGHTED_LOSS}, "
            f"positive_target_weight={TRANSFORMER_POSITIVE_TARGET_WEIGHT}, "
            f"positive_target_clip_bp={TRANSFORMER_POSITIVE_TARGET_CLIP_BP}, "
            f"grad_clip_norm={TRANSFORMER_GRAD_CLIP_NORM}"
        )

    scaler = load_or_fit_sequence_scaler(
        shard_paths=shard_paths,
        scaler_path=scaler_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{normalized_name}: device:", device)

    model = create_sequence_model(
        model_name=normalized_name,
        input_size=len(FEATURE_COLS),
        output_size=len(MODEL_TARGET_COLS),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]),
    )

    resume_info = load_existing_sequence_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        model_path=model_path,
        device=device,
        model_name=normalized_name,
        hp=hp,
    )

    resumed_from_existing_model = bool(
        resume_info["resumed_from_existing_model"]
    )
    resumed_optimizer_state = bool(
        resume_info["resumed_optimizer_state"]
    )
    previous_completed_epochs = int(
        resume_info["previous_completed_epochs"]
    )

    loss_fn = torch.nn.MSELoss()

    rng = np.random.default_rng(int(hp["random_seed"]))

    model.train()

    if resumed_from_existing_model:
        log(
            f"{normalized_name}: resume включён; "
            f"previous_completed_epochs={previous_completed_epochs}, "
            f"optimizer_state={resumed_optimizer_state}, "
            f"будет выполнено дополнительных эпох: {hp['epochs']}"
        )

    last_mean_loss: float | None = None

    for epoch in range(1, int(hp["epochs"]) + 1):
        shuffled_shards = list(shard_paths)
        rng.shuffle(shuffled_shards)

        total_loss = 0.0
        total_count = 0
        processed_in_epoch = 0

        log("=" * 80)
        global_epoch = previous_completed_epochs + epoch
        log(
            f"{normalized_name}: epoch {epoch}/{hp['epochs']} "
            f"(global {global_epoch}) старт"
        )

        for shard_idx, shard_path in enumerate(shuffled_shards, start=1):
            if (
                    SEQUENCE_MAX_TRAIN_SAMPLES is not None
                    and processed_in_epoch >= SEQUENCE_MAX_TRAIN_SAMPLES
            ):
                break

            x_raw, y, group_lengths = load_sequence_shard(shard_path)
            x = scaler.transform(x_raw).astype("float32", copy=False)

            row_indices = np.arange(len(x), dtype=np.int64)
            rng.shuffle(row_indices)

            if SEQUENCE_MAX_TRAIN_SAMPLES is not None:
                remaining = SEQUENCE_MAX_TRAIN_SAMPLES - processed_in_epoch
                row_indices = row_indices[:remaining]

            log(
                f"{normalized_name}: epoch {epoch}/{hp['epochs']}, "
                f"shard {shard_idx}/{len(shuffled_shards)} "
                f"{shard_path.name}, rows={len(row_indices)}"
            )

            batch_size_param = int(hp["batch_size"])

            for start in range(0, len(row_indices), batch_size_param):
                batch_indices = row_indices[start:start + batch_size_param]

                if len(batch_indices) == 0:
                    continue

                x_batch_np, y_batch_np = make_sequence_window_batch(
                    x=x,
                    y=y,
                    group_lengths=group_lengths,
                    row_indices=batch_indices,
                    window_size=WINDOW_SIZE,
                )

                x_batch = torch.from_numpy(x_batch_np).to(device, non_blocking=True)
                y_batch = torch.from_numpy(y_batch_np).to(device, non_blocking=True)
                y_batch = y_batch * SEQUENCE_TARGET_SCALE

                optimizer.zero_grad(set_to_none=True)
                pred = model(x_batch)

                if (
                        normalized_name == "transformer"
                        and TRANSFORMER_USE_WEIGHTED_LOSS
                ):
                    per_sample_loss = (pred - y_batch).pow(2).mean(dim=1)

                    execution_targets = y_batch[:, :len(EXECUTION_TARGET_COLS)]
                    best_execution_target_bp = torch.amax(
                        execution_targets,
                        dim=1,
                    )

                    sample_weights = 1.0 + TRANSFORMER_POSITIVE_TARGET_WEIGHT * torch.clamp(
                        best_execution_target_bp / TRANSFORMER_POSITIVE_TARGET_CLIP_BP,
                        min=0.0,
                        max=1.0,
                    )

                    loss = (per_sample_loss * sample_weights).mean()
                else:
                    loss = loss_fn(pred, y_batch)

                loss.backward()

                if normalized_name == "transformer":
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=TRANSFORMER_GRAD_CLIP_NORM,
                    )

                optimizer.step()

                current_batch_size = int(x_batch.shape[0])
                total_loss += float(loss.item()) * current_batch_size
                total_count += current_batch_size
                processed_in_epoch += current_batch_size

                if total_count % (batch_size_param * 500) < batch_size_param:
                    log(
                        f"{normalized_name}: epoch {epoch}/{hp['epochs']}, "
                        f"processed={total_count}, "
                        f"loss={loss.item():.8f}"
                    )

                if (
                        SEQUENCE_MAX_TRAIN_SAMPLES is not None
                        and processed_in_epoch >= SEQUENCE_MAX_TRAIN_SAMPLES
                ):
                    break

            del x_raw
            del x
            del y
            del group_lengths

        mean_loss = total_loss / max(total_count, 1)
        last_mean_loss = mean_loss
        completed_epochs = previous_completed_epochs + epoch

        log(
            f"{normalized_name}: epoch {epoch}/{hp['epochs']} "
            f"(global {completed_epochs}) завершена, "
            f"mean_loss={mean_loss:.8f}, samples={total_count}"
        )

        if SAVE_SEQUENCE_CHECKPOINT_EVERY_EPOCH:
            save_sequence_checkpoint(
                model=model,
                optimizer=optimizer,
                model_path=model_path,
                normalized_name=normalized_name,
                stage_name=stage_name,
                train_until=train_until,
                hp=hp,
                completed_epochs=completed_epochs,
                last_mean_loss=last_mean_loss,
            )
            log(
                f"{normalized_name}: checkpoint сохранён после epoch "
                f"{completed_epochs}: {model_path}"
            )

    final_completed_epochs = previous_completed_epochs + int(hp["epochs"])
    save_sequence_checkpoint(
        model=model,
        optimizer=optimizer,
        model_path=model_path,
        normalized_name=normalized_name,
        stage_name=stage_name,
        train_until=train_until,
        hp=hp,
        completed_epochs=final_completed_epochs,
        last_mean_loss=last_mean_loss,
    )
    joblib.dump(scaler, scaler_path)

    config = make_base_config(
        model_name=normalized_name,
        train_until=train_until,
        stage_name=stage_name,
    )
    config.update({
        "model_type": hp["model_type"],
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "sequence_max_train_samples": SEQUENCE_MAX_TRAIN_SAMPLES,
        "use_sequence_shards": USE_SEQUENCE_SHARDS,
        "sequence_target_scale": SEQUENCE_TARGET_SCALE,
        "resume_sequence_model_if_exists": RESUME_SEQUENCE_MODEL_IF_EXISTS,
        "resume_sequence_optimizer_if_exists": RESUME_SEQUENCE_OPTIMIZER_IF_EXISTS,
        "keep_current_optimizer_hyperparams_on_resume": KEEP_CURRENT_OPTIMIZER_HYPERPARAMS_ON_RESUME,
        "save_sequence_checkpoint_every_epoch": SAVE_SEQUENCE_CHECKPOINT_EVERY_EPOCH,
        "reuse_sequence_scaler_if_exists": REUSE_SEQUENCE_SCALER_IF_EXISTS,
        "resumed_from_existing_model": resumed_from_existing_model,
        "resumed_optimizer_state": resumed_optimizer_state,
        "previous_completed_epochs": previous_completed_epochs,
        "completed_epochs_after_run": previous_completed_epochs + int(hp["epochs"]),
        "last_mean_loss": last_mean_loss,
        "sequence_shards_manifest_path": str(
            Path("data")
            / "sequence_shards"
            / f"horizon_{HORIZON}_window_{WINDOW_SIZE}"
            / normalize_dataset_name(stage_name)
            / "manifest.json"
        ),
        "sequence_hyperparams": hp,
    })

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    log(f"{normalized_name}: checkpoint модели и optimizer сохранён:", model_path)
    log(f"{normalized_name}: scaler сохранён:", scaler_path)
    log(f"{normalized_name}: config сохранён:", config_path)

    return ModelArtifacts(
        model_name=normalized_name,
        model=NeuralRegressorWrapper(model=model, device=device),
        scaler=scaler,
        config=config,
    )


def predict_with_sequence_model(
        artifacts: ModelArtifacts,
        split_df: pd.DataFrame,
) -> np.ndarray:
    """
    Строит прогнозы GRU/LSTM/Transformer/TCN.
    """
    import torch

    if artifacts.scaler is None:
        raise ValueError("Для sequence-модели ожидается scaler")

    wrapper = artifacts.model
    model = wrapper.model
    device = wrapper.device

    hp = get_sequence_hyperparams(artifacts.model_name)
    predict_batch_size = int(hp["predict_batch_size"])

    sort_cols = ["secid", "trade_date", "begin"]
    sorted_df = split_df.sort_values(sort_cols, kind="mergesort")

    groups_x, groups_y = prepare_sequence_groups(
        df=sorted_df,
        scaler=artifacts.scaler,
        need_targets=False,
    )

    dataset = SequenceIndexDataset(
        groups_x=groups_x,
        groups_y=groups_y,
        window_size=WINDOW_SIZE,
        sample_indices=None,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=predict_batch_size,
        shuffle=False,
        num_workers=SEQUENCE_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    parts = []

    with torch.no_grad():
        for batch_idx, (x_batch, _y_batch) in enumerate(loader, start=1):
            x_batch = x_batch.to(device, non_blocking=True)
            pred = model(x_batch)
            parts.append(pred.cpu().numpy().astype("float32", copy=False))

            if batch_idx % 500 == 0:
                log(f"{artifacts.model_name}: predict batch {batch_idx}")

    sorted_pred = np.concatenate(parts, axis=0)
    sorted_pred = sorted_pred / SEQUENCE_TARGET_SCALE

    # Возвращаем прогнозы в исходный порядок split_df.
    order = sorted_df.index.to_numpy()
    inverse_order = (
        pd.Series(np.arange(len(order)), index=order)
        .loc[split_df.index]
        .to_numpy()
    )

    return sorted_pred[inverse_order]


def train_model_by_name(
        model_name: str,
        train_df: pd.DataFrame | None,
        train_until: str = TRAIN_UNTIL,
        stage_name: str = VALIDATION_STAGE_NAME,
) -> ModelArtifacts:
    """
    Обучает модель по имени.

    Это низкоуровневая функция. Для внешних модулей лучше использовать
    train_model_interface(...), потому что она сама читает нужные данные.
    """
    normalized_name = normalize_model_name(model_name)

    if normalized_name in {"ridge", "catboost", "random_forest", "decision_tree"}:
        if train_df is None:
            train_df = read_train_dataset(
                train_until=train_until,
                split_name=stage_name,
            )
        return train_ml_model(
            model_name=normalized_name,
            train_df=train_df,
            train_until=train_until,
            stage_name=stage_name,
        )

    if normalized_name in {"gru", "lstm", "transformer", "tcn"}:
        return train_sequence_model(
            model_name=normalized_name,
            train_until=train_until,
            stage_name=stage_name,
        )

    raise ValueError(f"Неизвестная модель: {model_name}")


def train_model_interface(
        model_name: str,
        train_until: str,
        stage_name: str,
) -> ModelArtifacts:
    """
    Единый интерфейс обучения.

    На вход подаётся имя модели и граница обучающего периода.
    На выход возвращаются обученная модель, scaler и config.
    """
    normalized_name = normalize_model_name(model_name)

    if normalized_name in {"ridge", "catboost", "random_forest", "decision_tree"}:
        full_train_df = read_train_dataset(
            train_until=train_until,
            split_name=stage_name,
        )

        log(
            f"{normalized_name}: размер обучающего датасета "
            f"для stage={stage_name}: {full_train_df.shape}"
        )

        train_df = sample_train_df_for_ml(full_train_df, normalized_name)

        if train_df is not full_train_df:
            log(
                f"{normalized_name}: после сэмплирования train shape: "
                f"{train_df.shape}"
            )
            del full_train_df

        return train_model_by_name(
            model_name=normalized_name,
            train_df=train_df,
            train_until=train_until,
            stage_name=stage_name,
        )

    return train_model_by_name(
        model_name=normalized_name,
        train_df=None,
        train_until=train_until,
        stage_name=stage_name,
    )

def predict_by_model_name(
        artifacts: ModelArtifacts,
        split_df: pd.DataFrame,
) -> np.ndarray:
    """
    Строит прогнозы модели по имени.
    """
    model_name = normalize_model_name(artifacts.model_name)

    if model_name in {"ridge", "catboost", "random_forest", "decision_tree"}:
        return predict_with_ml_model(artifacts, split_df)

    if model_name in {"gru", "lstm", "transformer", "tcn"}:
        return predict_with_sequence_model(artifacts, split_df)

    raise ValueError(f"Неизвестная модель: {artifacts.model_name}")


def build_prediction_dataset(
        artifacts: ModelArtifacts,
        split_name: str,
) -> pd.DataFrame:
    """
    Строит predictions для одного split и сохраняет их в data/<model>/.
    """
    split_df = read_prediction_split(split_name)

    log(f"{artifacts.model_name}: строю прогнозы для {split_name}")
    pred = predict_by_model_name(artifacts, split_df)

    log(f"{artifacts.model_name}: raw pred shape для {split_name}: {pred.shape}")

    prediction_df = add_predictions_to_dataframe(split_df, pred)
    save_prediction_dataset(artifacts.model_name, split_name, prediction_df)

    return prediction_df


def build_predictions_for_model(
        artifacts: ModelArtifacts,
        split_names: list[str] | None = None,
) -> None:
    """
    Строит predictions для нескольких split.
    """
    if split_names is None:
        split_names = ["valid", "test"]

    for split_name in split_names:
        prediction_df = build_prediction_dataset(
            artifacts=artifacts,
            split_name=split_name,
        )

        del prediction_df


def main() -> None:
    """
    Обучает выбранные модели и строит prediction-файлы для backtest_strategy.py.

    Для каждой модели создаются две версии:
    1. validation-версия: обучение на train, прогнозы для validation;
    2. final-версия: обучение на train + validation, прогнозы для test.

    Подбор threshold_bp/max_positions здесь не выполняется. Этим занимается
    только backtest_strategy.py на готовых prediction-файлах.
    """
    log("ML-модели для обучения:", ML_MODELS_TO_TRAIN)
    log("Нейросетевые модели для обучения:", NEURAL_MODELS_TO_TRAIN)
    log("Все модели для обучения:", TO_TRAIN)
    log("HORIZON:", HORIZON)
    log("WINDOW_SIZE:", WINDOW_SIZE)
    log("Количество признаков:", len(FEATURE_COLS))
    log("Количество выходов модели:", len(MODEL_TARGET_COLS))
    log("Execution-выходов:", len(EXECUTION_TARGET_COLS))
    log("Hold-выходов:", len(HOLD_TARGET_COLS))

    if not TO_TRAIN:
        raise ValueError(
            "TO_TRAIN пустой. Укажи хотя бы одну модель, например: "
            "ML_MODELS_TO_TRAIN = ['ridge'] или NEURAL_MODELS_TO_TRAIN = ['GRU']."
        )

    for model_name in TO_TRAIN:
        normalized_name = normalize_model_name(model_name)

        log("=" * 80)
        log(f"Начинаю validation-обучение модели: {model_name}")

        validation_artifacts = train_model_interface(
            model_name=normalized_name,
            train_until=TRAIN_UNTIL,
            stage_name=VALIDATION_STAGE_NAME,
        )

        build_predictions_for_model(
            artifacts=validation_artifacts,
            split_names=["valid"],
        )

        log(f"Начинаю final-обучение модели на train+validation: {model_name}")

        final_artifacts = train_model_interface(
            model_name=normalized_name,
            train_until=VALIDATE_UNTIL,
            stage_name=FINAL_STAGE_NAME,
        )

        build_predictions_for_model(
            artifacts=final_artifacts,
            split_names=["test"],
        )

        log(f"Модель {model_name} завершена")

    log("Обучение и построение прогнозов завершены")


if __name__ == "__main__":
    main()
