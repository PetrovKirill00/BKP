from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from constants import HORIZON, TRAIN_UNTIL, VALIDATE_UNTIL, WINDOW_SIZE


DATASET_PATH = Path("minute_dataset.parquet")

MODELS_ROOT = Path("models")
DATA_ROOT = Path("data")

# Здесь выбираешь, какие модели обучать и для каких моделей строить valid/test predictions.
# Для первого запуска лучше оставить ["ridge"], потом добавить "gru".
MODELS_TO_TRAIN = ["ridge", "gru"]

RIDGE_ALPHA = 1.0

GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 1
GRU_DROPOUT = 0.0
GRU_BATCH_SIZE = 1024
GRU_EPOCHS = 5
GRU_LEARNING_RATE = 1e-3
GRU_WEIGHT_DECAY = 1e-5
GRU_RANDOM_SEED = 42

# Полный train может быть очень большим. None означает использовать все доступные train-окна.
# Для первого эксперимента с GRU разумнее оставить ограничение.
GRU_MAX_TRAIN_SAMPLES = 1_000_000
GRU_NUM_WORKERS = 0
GRU_PREDICT_BATCH_SIZE = 4096

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
    scaler: StandardScaler
    config: dict[str, Any]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_model_dir(model_name: str) -> Path:
    return MODELS_ROOT / model_name


def get_data_dir(model_name: str) -> Path:
    return DATA_ROOT / model_name


def get_model_config_path(model_name: str) -> Path:
    return get_model_dir(model_name) / f"model_config_horizon_{HORIZON}.json"


def validate_dataset(
        df: pd.DataFrame,
        split_name: str,
        need_targets: bool,
) -> None:
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


def read_train_dataset() -> pd.DataFrame:
    columns = [
        "begin",
        "secid",
        "trade_date",
    ] + FEATURE_COLS + MODEL_TARGET_COLS

    print("Читаю train-часть датасета")

    df = pd.read_parquet(
        DATASET_PATH,
        columns=columns,
        filters=[("begin", "<", pd.Timestamp(TRAIN_UNTIL))],
    )

    validate_dataset(df, split_name="train", need_targets=True)

    return df


def read_prediction_split(split_name: str) -> pd.DataFrame:
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

    print(f"Читаю {split_name}-часть датасета для построения прогнозов")

    df = pd.read_parquet(
        DATASET_PATH,
        columns=columns,
        filters=filters,
    )

    validate_dataset(df, split_name=split_name, need_targets=False)

    return df


def calculate_score_matrix(
        forecast_matrix: np.ndarray,
        score_mode: str,
        top_p: int = 1,
        top_m: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
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

    for idx, col in enumerate(EXECUTION_TARGET_COLS, start=1):
        result[f"execution_forecast_{idx}"] = execution_pred[:, idx - 1].astype("float32")

    for idx, col in enumerate(HOLD_TARGET_COLS, start=1):
        result[f"hold_forecast_{idx}"] = hold_pred[:, idx - 1].astype("float32")

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
    data_dir = get_data_dir(model_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    path = data_dir / f"{split_name}_predictions.parquet"
    prediction_df.to_parquet(path, index=False)

    print(f"{model_name}: {split_name} predictions сохранены: {path}")
    print(f"{model_name}: {split_name} predictions shape: {prediction_df.shape}")


def make_base_config(model_name: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "train_until": TRAIN_UNTIL,
        "validate_until": VALIDATE_UNTIL,
        "target_type": "multi_output_execution_and_hold_return",
        "execution_target_cols": EXECUTION_TARGET_COLS,
        "hold_target_cols": HOLD_TARGET_COLS,
        "model_target_cols": MODEL_TARGET_COLS,
        "feature_cols": FEATURE_COLS,
        "prediction_data_dir": str(get_data_dir(model_name)),
    }


def train_ridge_model(train_df: pd.DataFrame) -> ModelArtifacts:
    model_name = "ridge"
    model_dir = get_model_dir(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"ridge_horizon_{HORIZON}.joblib"
    scaler_path = model_dir / f"scaler_horizon_{HORIZON}.joblib"
    config_path = get_model_config_path(model_name)

    print("ridge: готовлю матрицу признаков и целевых значений")

    x_train = train_df[FEATURE_COLS].to_numpy(copy=False).astype("float32", copy=False)
    y_train = train_df[MODEL_TARGET_COLS].to_numpy(copy=False).astype("float32", copy=False)

    print("ridge: X_train shape:", x_train.shape)
    print("ridge: y_train shape:", y_train.shape)

    scaler = StandardScaler(copy=False)

    print("ridge: нормализую признаки")
    x_train_scaled = scaler.fit_transform(x_train).astype("float32", copy=False)

    model = Ridge(alpha=RIDGE_ALPHA)

    print("ridge: обучаю multi-output Ridge-регрессию")
    model.fit(x_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    config = make_base_config(model_name)
    config.update({
        "model_type": "Ridge",
        "alpha": RIDGE_ALPHA,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    })

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("ridge: модель сохранена:", model_path)
    print("ridge: scaler сохранён:", scaler_path)
    print("ridge: config сохранён:", config_path)

    return ModelArtifacts(
        model_name=model_name,
        model=model,
        scaler=scaler,
        config=config,
    )


def predict_with_ridge(
        artifacts: ModelArtifacts,
        split_df: pd.DataFrame,
) -> np.ndarray:
    x = split_df[FEATURE_COLS].to_numpy(copy=False).astype("float32", copy=False)
    x_scaled = artifacts.scaler.transform(x).astype("float32", copy=False)
    pred = artifacts.model.predict(x_scaled)
    return pred.astype("float32", copy=False)


class GruRegressorWrapper:
    def __init__(self, model: Any, device: Any):
        self.model = model
        self.device = device


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

        group_idx = int(np.searchsorted(self.cumulative_lengths, global_idx, side="right"))
        group_start = 0 if group_idx == 0 else int(self.cumulative_lengths[group_idx - 1])
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
    sort_cols = ["secid", "trade_date", "begin"]
    df = df.sort_values(sort_cols, kind="mergesort")

    groups_x = []
    groups_y = []

    for _key, group in df.groupby(["secid", "trade_date"], sort=False):
        x = group[FEATURE_COLS].to_numpy(copy=False).astype("float32", copy=False)
        x = scaler.transform(x).astype("float32", copy=False)
        groups_x.append(x)

        if need_targets:
            y = group[MODEL_TARGET_COLS].to_numpy(copy=False).astype("float32", copy=False)
        else:
            y = np.zeros((len(group), len(MODEL_TARGET_COLS)), dtype="float32")

        groups_y.append(y)

    return groups_x, groups_y


def create_gru_model(input_size: int, output_size: int):
    import torch
    from torch import nn

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


def train_gru_model(train_df: pd.DataFrame) -> ModelArtifacts:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Для обучения GRU нужен PyTorch. Установи его или убери 'gru' из MODELS_TO_TRAIN."
        ) from exc

    model_name = "gru"
    model_dir = get_model_dir(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"gru_horizon_{HORIZON}.pt"
    scaler_path = model_dir / f"scaler_horizon_{HORIZON}.joblib"
    config_path = get_model_config_path(model_name)

    set_random_seed(GRU_RANDOM_SEED)
    torch.manual_seed(GRU_RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GRU_RANDOM_SEED)

    print("gru: готовлю scaler по train-признакам")

    x_train_flat = train_df[FEATURE_COLS].to_numpy(copy=False).astype("float32", copy=False)
    scaler = StandardScaler(copy=False)
    scaler.fit(x_train_flat)

    print("gru: строю последовательные группы secid/trade_date")

    groups_x, groups_y = prepare_sequence_groups(
        df=train_df,
        scaler=scaler,
        need_targets=True,
    )

    total_samples = int(sum(len(group) for group in groups_x))
    print("gru: всего train-строк/окон:", total_samples)
    print("gru: групп secid/trade_date:", len(groups_x))

    if GRU_MAX_TRAIN_SAMPLES is not None and total_samples > GRU_MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(GRU_RANDOM_SEED)
        sample_indices = rng.choice(
            total_samples,
            size=GRU_MAX_TRAIN_SAMPLES,
            replace=False,
        )
        sample_indices.sort()
        print("gru: используем подвыборку train-окон:", len(sample_indices))
    else:
        sample_indices = None
        print("gru: используем все train-окна")

    dataset = SequenceIndexDataset(
        groups_x=groups_x,
        groups_y=groups_y,
        window_size=WINDOW_SIZE,
        sample_indices=sample_indices,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=GRU_BATCH_SIZE,
        shuffle=True,
        num_workers=GRU_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gru: device:", device)

    model = create_gru_model(
        input_size=len(FEATURE_COLS),
        output_size=len(MODEL_TARGET_COLS),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GRU_LEARNING_RATE,
        weight_decay=GRU_WEIGHT_DECAY,
    )
    loss_fn = torch.nn.MSELoss()

    model.train()

    for epoch in range(1, GRU_EPOCHS + 1):
        total_loss = 0.0
        total_count = 0

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader, start=1):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = x_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            if batch_idx % 500 == 0:
                print(
                    f"gru: epoch {epoch}/{GRU_EPOCHS}, "
                    f"batch {batch_idx}, loss={loss.item():.8f}"
                )

        mean_loss = total_loss / max(total_count, 1)
        print(f"gru: epoch {epoch}/{GRU_EPOCHS}, mean_loss={mean_loss:.8f}")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    config = make_base_config(model_name)
    config.update({
        "model_type": "GRU",
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "gru_hidden_size": GRU_HIDDEN_SIZE,
        "gru_num_layers": GRU_NUM_LAYERS,
        "gru_dropout": GRU_DROPOUT,
        "gru_batch_size": GRU_BATCH_SIZE,
        "gru_epochs": GRU_EPOCHS,
        "gru_learning_rate": GRU_LEARNING_RATE,
        "gru_weight_decay": GRU_WEIGHT_DECAY,
        "gru_max_train_samples": GRU_MAX_TRAIN_SAMPLES,
    })

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("gru: модель сохранена:", model_path)
    print("gru: scaler сохранён:", scaler_path)
    print("gru: config сохранён:", config_path)

    return ModelArtifacts(
        model_name=model_name,
        model=GruRegressorWrapper(model=model, device=device),
        scaler=scaler,
        config=config,
    )


def predict_with_gru(
        artifacts: ModelArtifacts,
        split_df: pd.DataFrame,
) -> np.ndarray:
    import torch

    wrapper = artifacts.model
    model = wrapper.model
    device = wrapper.device

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
        batch_size=GRU_PREDICT_BATCH_SIZE,
        shuffle=False,
        num_workers=GRU_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    parts = []

    with torch.no_grad():
        for x_batch, _y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            pred = model(x_batch)
            parts.append(pred.cpu().numpy().astype("float32", copy=False))

    sorted_pred = np.concatenate(parts, axis=0)

    # Возвращаем прогнозы в исходный порядок split_df.
    order = sorted_df.index.to_numpy()
    inverse_order = pd.Series(np.arange(len(order)), index=order).loc[split_df.index].to_numpy()

    return sorted_pred[inverse_order]


def train_model_by_name(
        model_name: str,
        train_df: pd.DataFrame,
) -> ModelArtifacts:
    normalized_name = model_name.lower()

    if normalized_name == "ridge":
        return train_ridge_model(train_df)

    if normalized_name == "gru":
        return train_gru_model(train_df)

    raise ValueError(f"Неизвестная модель: {model_name}")


def predict_by_model_name(
        artifacts: ModelArtifacts,
        split_df: pd.DataFrame,
) -> np.ndarray:
    model_name = artifacts.model_name.lower()

    if model_name == "ridge":
        return predict_with_ridge(artifacts, split_df)

    if model_name == "gru":
        return predict_with_gru(artifacts, split_df)

    raise ValueError(f"Неизвестная модель: {artifacts.model_name}")


def build_predictions_for_model(artifacts: ModelArtifacts) -> None:
    for split_name in ["valid", "test"]:
        split_df = read_prediction_split(split_name)

        print(f"{artifacts.model_name}: строю прогнозы для {split_name}")
        pred = predict_by_model_name(artifacts, split_df)

        prediction_df = add_predictions_to_dataframe(split_df, pred)
        save_prediction_dataset(artifacts.model_name, split_name, prediction_df)

        del split_df
        del pred
        del prediction_df


def main() -> None:
    print("Модели для обучения:", MODELS_TO_TRAIN)
    print("HORIZON:", HORIZON)
    print("WINDOW_SIZE:", WINDOW_SIZE)
    print("Количество признаков:", len(FEATURE_COLS))
    print("Количество выходов модели:", len(MODEL_TARGET_COLS))
    print("Execution-выходов:", len(EXECUTION_TARGET_COLS))
    print("Hold-выходов:", len(HOLD_TARGET_COLS))

    train_df = read_train_dataset()

    print("Размер train-датасета:", train_df.shape)

    for model_name in MODELS_TO_TRAIN:
        print("=" * 80)
        print(f"Начинаю обучение модели: {model_name}")

        artifacts = train_model_by_name(model_name, train_df)
        build_predictions_for_model(artifacts)

        print(f"Модель {model_name} завершена")

    print("Обучение и построение прогнозов завершены")


if __name__ == "__main__":
    main()
