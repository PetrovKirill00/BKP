import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from constants import HORIZON, TRAIN_UNTIL, VALIDATE_UNTIL


DATASET_PATH = Path("minute_dataset.parquet")
DATA_ROOT = Path("data")

# Здесь выбираешь, какие модели прогонять через backtest.
# Для каждой модели ожидаются файлы:
# data/<model_name>/valid_predictions.parquet
# data/<model_name>/test_predictions.parquet
MODELS_TO_BACKTEST = ["ridge", "gru"]

MODEL_DIR = Path("models")
CONFIG_PATH = MODEL_DIR / f"model_config_horizon_{HORIZON}.json"

RESULTS_DIR = Path("backtest_results")

INITIAL_CASH = 1_000_000.0


# ============================================================
# Главный переключатель режима
# ============================================================

# True  -> быстрый частичный прогон для отладки.
# False -> полный прогон для итоговых результатов.
PARTIAL_BACKTEST = True


FULL_BACKTEST_SETTINGS = {
    "name": "full",

    "thresholds_bp": [1.0, 2.0, 3.0, 4.0, 5.0],
    "max_positions_list": [1, 2, 3, 4, 5],

    "cost_bp_pairs": [
        (1.0, 1.0),
        (3.0, 3.0),
        (5.0, 5.0),
    ],

    "debug_max_minutes": None,

    "progress_every_minutes": 10_000,

    "save_validation_details": False,
    "save_test_details": True,
}

PARTIAL_BACKTEST_SETTINGS = {
    "name": "partial",

    "thresholds_bp": [2.5],
    "max_positions_list": [1, 3],

    "cost_bp_pairs": [
        (2.5, 2.5),
    ],

    "debug_max_minutes": 20_000,

    "progress_every_minutes": 2_000,

    "save_validation_details": False,
    "save_test_details": True,
}

# ============================================================
# Параметры risk management
# ============================================================

MAX_INSTRUMENT_SHARE = 0.05
MAX_INSTRUMENT_CAP_RUB = 50_000.0

SPEND_ON_MINUTE_RUB = 100_000.0
SPEND_ON_MINUTE_SHARE = 0.10

MIN_TRADE_RUB = 1_000.0

MAX_HOLDING_MINUTES = 30


# ============================================================
# Параметры score по прогнозам
# ============================================================

OPEN_SCORE_MODE = "top_p"      # "top_p" или "top_mean"
OPEN_TOP_P = 1
OPEN_TOP_M = 3

HOLD_SCORE_MODE = "top_p"      # "top_p" или "top_mean"
HOLD_TOP_P = 1
HOLD_TOP_M = 3
HOLD_EXTRA_BUFFER_BP = 0.0


# ============================================================
# Логи и сохранение подробных результатов
# ============================================================

VERBOSE = True


def get_active_backtest_settings() -> dict[str, Any]:
    """
    Возвращает активный набор параметров для полного или частичного тестирования.
    """
    if PARTIAL_BACKTEST:
        return PARTIAL_BACKTEST_SETTINGS

    return FULL_BACKTEST_SETTINGS


def log(message: str) -> None:
    """
    Печатает диагностическое сообщение с текущим временем.
    """
    if VERBOSE:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {message}", flush=True)


def bp_to_return(value_bp: float) -> float:
    """
    Переводит базисные пункты в обычную дробную доходность.
    """
    return value_bp / 10_000


def build_parameter_grid(
        thresholds_bp: list[float],
        max_positions_list: list[int],
        cost_bp_pairs: list[tuple[float, float]],
) -> list[dict[str, float | int]]:
    """
    Создаёт список комбинаций параметров для validation.
    """
    parameter_grid = []

    for threshold_bp in thresholds_bp:
        for max_positions in max_positions_list:
            for buy_cost_bp, sell_cost_bp in cost_bp_pairs:
                parameter_grid.append({
                    "threshold_bp": threshold_bp,
                    "max_positions": max_positions,
                    "buy_cost_bp": buy_cost_bp,
                    "sell_cost_bp": sell_cost_bp,
                })

    return parameter_grid


def load_model_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    """
    Загружает config, обученную модель и scaler.
    """
    log(f"Загружаю config: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = [
        "model_path",
        "scaler_path",
        "feature_cols",
        "model_target_cols",
        "execution_target_cols",
        "hold_target_cols",
        "horizon",
    ]

    log("Проверяю ключи config")

    for key in required_keys:
        if key not in config:
            raise KeyError(
                f"В config нет ключа {key}. "
                f"Проверь, что train_models.py сохраняет новый config."
            )

    if int(config["horizon"]) != int(HORIZON):
        raise ValueError(
            f"В config horizon={config['horizon']}, "
            f"а в constants.py HORIZON={HORIZON}."
        )

    log(f"Загружаю модель: {config['model_path']}")
    model = joblib.load(config["model_path"])

    log(f"Загружаю scaler: {config['scaler_path']}")
    scaler = joblib.load(config["scaler_path"])

    log(
        "Артефакты загружены: "
        f"features={len(config['feature_cols'])}, "
        f"outputs={len(config['model_target_cols'])}, "
        f"execution_outputs={len(config['execution_target_cols'])}, "
        f"hold_outputs={len(config['hold_target_cols'])}"
    )

    return model, scaler, config


def read_dataset_part(
        split_name: str,
        feature_cols: list[str],
        debug_max_minutes: int | None,
) -> pd.DataFrame:
    """
    Читает validation или test часть датасета для исторического тестирования.
    """
    columns = [
        "begin",
        "secid",
        "close",
        "next_open",
    ] + feature_cols

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

    log(f"Читаю {split_name}-часть датасета")
    log(f"Колонок для чтения: {len(columns)}")
    log(f"Фильтры parquet: {filters}")

    df = pd.read_parquet(
        DATASET_PATH,
        columns=columns,
        filters=filters,
    )

    log(f"{split_name}: датасет прочитан, shape={df.shape}")
    log(f"{split_name}: проверяю входные данные")

    validate_input_data(df, split_name, feature_cols)

    if debug_max_minutes is not None:
        log(f"{split_name}: debug_max_minutes={debug_max_minutes}, обрезаю выборку")

        unique_minutes = df["begin"].drop_duplicates()

        if len(unique_minutes) > debug_max_minutes:
            last_allowed_minute = unique_minutes.iloc[debug_max_minutes - 1]
            df = df[df["begin"] <= last_allowed_minute].copy()

            log(
                f"{split_name}: после обрезки shape={df.shape}, "
                f"last_allowed_minute={last_allowed_minute}"
            )

    log(f"{split_name}: проверка данных завершена")
    log(
        f"{split_name}: диапазон begin: "
        f"{df['begin'].min()} -> {df['begin'].max()}"
    )
    log(f"{split_name}: уникальных минут: {df['begin'].nunique()}")
    log(f"{split_name}: уникальных бумаг: {df['secid'].nunique()}")

    return df


def validate_input_data(
        df: pd.DataFrame,
        split_name: str,
        feature_cols: list[str],
) -> None:
    """
    Проверяет, что входные данные для backtest не содержат пропусков и бесконечностей.
    """
    required_cols = [
        "begin",
        "secid",
        "close",
        "next_open",
    ] + feature_cols

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"В {split_name}-части отсутствуют колонки: {missing_cols}"
        )

    log(f"{split_name}: проверяю NaN")

    nan_rows = df[required_cols].isna().any(axis=1).sum()
    if nan_rows > 0:
        raise ValueError(
            f"В {split_name}-части найдено {nan_rows} строк с NaN. "
            f"Проверь build_dataset.py."
        )

    numeric_cols = [
        col
        for col in required_cols
        if col not in ["begin", "secid"]
    ]

    log(f"{split_name}: проверяю inf/-inf")

    finite_mask = np.isfinite(df[numeric_cols].to_numpy()).all(axis=1)
    bad_rows = len(df) - finite_mask.sum()

    if bad_rows > 0:
        raise ValueError(
            f"В {split_name}-части найдено {bad_rows} строк с inf/-inf. "
            f"Проверь build_dataset.py."
        )


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


def add_model_predictions(
        df: pd.DataFrame,
        model: Any,
        scaler: Any,
        config: dict[str, Any],
) -> pd.DataFrame:
    """
    Добавляет к данным заранее рассчитанные score для открытия и удержания позиции.
    """
    feature_cols = config["feature_cols"]
    execution_target_cols = config["execution_target_cols"]
    hold_target_cols = config["hold_target_cols"]
    model_target_cols = config["model_target_cols"]

    log("Готовлю матрицу признаков для прогноза")
    log(f"Размер df перед прогнозом: {df.shape}")
    log(f"Количество признаков: {len(feature_cols)}")

    x = df[feature_cols].to_numpy(copy=False)

    log(f"X shape: {x.shape}")
    log("Нормализую признаки scaler.transform")

    x_scaled = scaler.transform(x)

    log("Строю прогнозы model.predict")

    pred = model.predict(x_scaled)

    log(f"Pred shape: {pred.shape}")

    if pred.ndim != 2:
        raise ValueError(
            "Ожидался multi-output прогноз формы "
            "(n_samples, n_outputs), но получен одномерный прогноз."
        )

    if pred.shape[1] != len(model_target_cols):
        raise ValueError(
            f"Число выходов модели {pred.shape[1]} не совпадает "
            f"с числом целевых колонок {len(model_target_cols)}."
        )

    execution_count = len(execution_target_cols)
    hold_count = len(hold_target_cols)

    execution_pred = pred[:, :execution_count]
    hold_pred = pred[:, execution_count:execution_count + hold_count]

    log("Заранее считаю open_score/open_horizon")

    open_score, open_horizon = calculate_score_matrix(
        forecast_matrix=execution_pred,
        score_mode=OPEN_SCORE_MODE,
        top_p=OPEN_TOP_P,
        top_m=OPEN_TOP_M,
    )

    df["open_score"] = open_score.astype("float32")
    df["open_horizon"] = open_horizon.astype("int8")

    log("Заранее считаю hold_score/hold_horizon")

    hold_score, hold_horizon = calculate_score_matrix(
        forecast_matrix=hold_pred,
        score_mode=HOLD_SCORE_MODE,
        top_p=HOLD_TOP_P,
        top_m=HOLD_TOP_M,
    )

    df["hold_score"] = hold_score.astype("float32")
    df["hold_horizon"] = hold_horizon.astype("int8")

    log(f"Score-колонки добавлены, df shape={df.shape}")

    return df


def is_position_active(
        position: dict[str, Any],
        begin: pd.Timestamp,
) -> bool:
    """
    Проверяет, наступило ли время фактического входа в позицию.
    """
    return begin >= position["entry_time"]


def get_holding_minutes(
        position: dict[str, Any],
        begin: pd.Timestamp,
) -> float:
    """
    Считает, сколько минут позиция фактически удерживается.
    """
    if not is_position_active(position, begin):
        return 0.0

    holding_time = begin - position["entry_time"]

    return holding_time.total_seconds() / 60.0


def get_position_value(
        position: dict[str, Any],
        begin: pd.Timestamp,
        minute_by_secid: pd.DataFrame,
        sell_cost: float,
) -> float:
    """
    Считает ликвидационную стоимость одной позиции на текущей минуте.
    """
    if not is_position_active(position, begin):
        return position["cash_spent"]

    secid = position["secid"]

    if secid in minute_by_secid.index:
        row = minute_by_secid.loc[secid]
        current_close = float(row["close"])

        if np.isfinite(current_close) and current_close > 0:
            position["last_price"] = current_close

    price = position["last_price"]
    current_value = position["quantity"] * price

    return current_value * (1.0 - sell_cost)


def mark_to_market(
        cash: float,
        open_positions: list[dict[str, Any]],
        begin: pd.Timestamp,
        minute_by_secid: pd.DataFrame,
        sell_cost: float,
) -> tuple[float, float]:
    """
    Считает текущую стоимость портфеля по close текущей минуты.
    """
    position_value = 0.0

    for position in open_positions:
        position_value += get_position_value(
            position=position,
            begin=begin,
            minute_by_secid=minute_by_secid,
            sell_cost=sell_cost,
        )

    equity = cash + position_value

    return equity, position_value


def calculate_exposure_by_secid(
        open_positions: list[dict[str, Any]],
        begin: pd.Timestamp,
        minute_by_secid: pd.DataFrame,
) -> dict[str, float]:
    """
    Считает, сколько капитала сейчас связано с каждой бумагой.
    """
    exposure_by_secid = {}

    for position in open_positions:
        secid = position["secid"]

        if not is_position_active(position, begin):
            current_value = position["cash_spent"]
        else:
            if secid in minute_by_secid.index:
                row = minute_by_secid.loc[secid]
                current_close = float(row["close"])

                if np.isfinite(current_close) and current_close > 0:
                    position["last_price"] = current_close

            current_value = position["quantity"] * position["last_price"]

        exposure_by_secid[secid] = (
            exposure_by_secid.get(secid, 0.0) + current_value
        )

    return exposure_by_secid


def close_position(
        begin: pd.Timestamp,
        cash: float,
        position: dict[str, Any],
        current_close: float,
        sell_cost: float,
        close_reason: str,
        close_forecast: float,
        close_horizon: float,
        trade_events: list[dict[str, Any]],
        closed_trades: list[dict[str, Any]],
) -> float:
    """
    Закрывает одну позицию по close текущей минуты.
    """
    position["last_price"] = current_close

    current_value = position["quantity"] * current_close
    sell_cost_value = current_value * sell_cost
    sell_cash = current_value - sell_cost_value

    cash += sell_cash

    pnl = sell_cash - position["cash_spent"]

    trade_events.append({
        "time": begin,
        "event": "SELL",
        "secid": position["secid"],
        "price": current_close,
        "quantity": position["quantity"],
        "cash_change": sell_cash,
        "cash_after": cash,
        "reason": close_reason,
        "forecast": close_forecast,
        "horizon": close_horizon,
    })

    closed_trades.append({
        "secid": position["secid"],
        "signal_time": position["signal_time"],
        "entry_time": position["entry_time"],
        "exit_time": begin,
        "entry_price": position["entry_price"],
        "exit_price": current_close,
        "quantity": position["quantity"],
        "cash_spent": position["cash_spent"],
        "trade_value": position["trade_value"],
        "exit_cash": sell_cash,
        "buy_cost": position["buy_cost"],
        "sell_cost": sell_cost_value,
        "pnl": pnl,
        "return": pnl / position["cash_spent"],
        "open_forecast": position["open_forecast"],
        "open_horizon": position["open_horizon"],
        "close_forecast": close_forecast,
        "close_horizon": close_horizon,
        "close_reason": close_reason,
        "holding_minutes": get_holding_minutes(position, begin),
    })

    return cash


def try_close_positions(
        begin: pd.Timestamp,
        cash: float,
        open_positions: list[dict[str, Any]],
        minute_by_secid: pd.DataFrame,
        sell_cost: float,
        trade_events: list[dict[str, Any]],
        closed_trades: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]], int, int, int]:
    """
    Закрывает позиции, если прогноз удержания недостаточен или превышен срок удержания.
    """
    still_open_positions = []

    required_hold_return = sell_cost + bp_to_return(HOLD_EXTRA_BUFFER_BP)

    closed_count = 0
    max_hold_closed_count = 0
    forecast_closed_count = 0

    for position in open_positions:
        secid = position["secid"]

        if not is_position_active(position, begin):
            still_open_positions.append(position)
            continue

        if secid not in minute_by_secid.index:
            still_open_positions.append(position)
            continue

        row = minute_by_secid.loc[secid]

        current_close = float(row["close"])

        if not np.isfinite(current_close) or current_close <= 0:
            still_open_positions.append(position)
            continue

        holding_minutes = get_holding_minutes(position, begin)

        if holding_minutes >= MAX_HOLDING_MINUTES:
            cash = close_position(
                begin=begin,
                cash=cash,
                position=position,
                current_close=current_close,
                sell_cost=sell_cost,
                close_reason="max_holding_minutes",
                close_forecast=np.nan,
                close_horizon=np.nan,
                trade_events=trade_events,
                closed_trades=closed_trades,
            )

            closed_count += 1
            max_hold_closed_count += 1
            continue

        hold_score = float(row["hold_score"])
        hold_horizon = int(row["hold_horizon"])

        current_value = position["quantity"] * current_close
        sell_now_cash = current_value * (1.0 - sell_cost)

        expected_hold_cash = (
            current_value
            * (1.0 + hold_score)
            * (1.0 - sell_cost)
        )

        should_hold = (
            hold_score >= required_hold_return
            and expected_hold_cash > sell_now_cash
        )

        if should_hold:
            position["last_price"] = current_close
            position["last_hold_score"] = hold_score
            position["last_hold_horizon"] = hold_horizon
            still_open_positions.append(position)
            continue

        cash = close_position(
            begin=begin,
            cash=cash,
            position=position,
            current_close=current_close,
            sell_cost=sell_cost,
            close_reason="hold_forecast_not_enough",
            close_forecast=hold_score,
            close_horizon=hold_horizon,
            trade_events=trade_events,
            closed_trades=closed_trades,
        )

        closed_count += 1
        forecast_closed_count += 1

    return (
        cash,
        still_open_positions,
        closed_count,
        max_hold_closed_count,
        forecast_closed_count,
    )


def select_open_candidates(
        minute_data: pd.DataFrame,
        threshold: float,
        free_slots: int,
        buy_cost: float,
        sell_cost: float,
) -> pd.DataFrame:
    """
    Выбирает бумаги-кандидаты для открытия новых позиций.
    """
    if free_slots <= 0 or minute_data.empty:
        return minute_data.iloc[0:0].copy()

    candidates = minute_data.copy()

    candidates["expected_edge"] = (
        candidates["open_score"] - buy_cost - sell_cost
    ).clip(lower=0.0)

    candidates = candidates[
        (candidates["open_score"] >= threshold)
        & (candidates["expected_edge"] > 0.0)
    ]

    if candidates.empty:
        return candidates

    return candidates.nlargest(free_slots, "open_score")


def try_open_positions(
        begin: pd.Timestamp,
        final_time: pd.Timestamp,
        cash: float,
        open_positions: list[dict[str, Any]],
        minute_data: pd.DataFrame,
        minute_by_secid: pd.DataFrame,
        portfolio_value: float,
        threshold: float,
        max_positions: int,
        buy_cost: float,
        sell_cost: float,
        trade_events: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]], int]:
    """
    Открывает новые позиции по next_open, если прогноз открытия достаточно высокий.
    """
    free_slots = max_positions - len(open_positions)

    if free_slots <= 0 or cash <= 0:
        return cash, open_positions, 0

    candidates = select_open_candidates(
        minute_data=minute_data,
        threshold=threshold,
        free_slots=free_slots,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
    )

    if candidates.empty:
        return cash, open_positions, 0

    spend_on_minute = min(
        cash,
        SPEND_ON_MINUTE_RUB,
        portfolio_value * SPEND_ON_MINUTE_SHARE,
    )

    if spend_on_minute < MIN_TRADE_RUB:
        return cash, open_positions, 0

    total_edge = candidates["expected_edge"].sum()

    if total_edge <= 0:
        return cash, open_positions, 0

    exposure_by_secid = calculate_exposure_by_secid(
        open_positions=open_positions,
        begin=begin,
        minute_by_secid=minute_by_secid,
    )

    opened_count = 0

    for _, row in candidates.iterrows():
        secid = row["secid"]
        next_open = float(row["next_open"])

        if not np.isfinite(next_open) or next_open <= 0:
            continue

        entry_time = begin + pd.Timedelta(minutes=1)

        if entry_time > final_time:
            continue

        current_exposure = exposure_by_secid.get(secid, 0.0)

        max_instrument_exposure = min(
            MAX_INSTRUMENT_CAP_RUB,
            portfolio_value * MAX_INSTRUMENT_SHARE,
        )

        available_for_instrument = (
            max_instrument_exposure - current_exposure
        )

        if available_for_instrument < MIN_TRADE_RUB:
            continue

        raw_weight = row["expected_edge"] / total_edge
        desired_cash = spend_on_minute * raw_weight

        cash_to_spend = min(
            desired_cash,
            available_for_instrument,
            cash,
        )

        if cash_to_spend < MIN_TRADE_RUB:
            continue

        trade_value = cash_to_spend / (1.0 + buy_cost)
        buy_cost_value = cash_to_spend - trade_value

        quantity = trade_value / next_open

        if not np.isfinite(quantity) or quantity <= 0:
            continue

        decision_horizon = int(row["open_horizon"])
        decision_score = float(row["open_score"])

        position = {
            "secid": secid,
            "signal_time": begin,
            "entry_time": entry_time,
            "entry_price": next_open,
            "quantity": quantity,
            "cash_spent": cash_to_spend,
            "trade_value": trade_value,
            "buy_cost": buy_cost_value,
            "open_forecast": decision_score,
            "open_horizon": decision_horizon,
            "last_price": next_open,
            "last_hold_score": np.nan,
            "last_hold_horizon": np.nan,
        }

        open_positions.append(position)
        opened_count += 1

        cash -= cash_to_spend
        exposure_by_secid[secid] = current_exposure + cash_to_spend

        trade_events.append({
            "time": entry_time,
            "signal_time": begin,
            "event": "BUY",
            "secid": secid,
            "price": next_open,
            "quantity": quantity,
            "cash_change": -cash_to_spend,
            "cash_after": cash,
            "reason": "execution_forecast_above_threshold",
            "forecast": decision_score,
            "horizon": decision_horizon,
        })

        if len(open_positions) >= max_positions:
            break

        if cash < MIN_TRADE_RUB:
            break

    return cash, open_positions, opened_count


def close_all_positions_at_end(
        final_time: pd.Timestamp,
        cash: float,
        open_positions: list[dict[str, Any]],
        minute_by_secid: pd.DataFrame,
        sell_cost: float,
        trade_events: list[dict[str, Any]],
        closed_trades: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """
    Принудительно закрывает все оставшиеся позиции в конце периода.
    """
    log(f"Принудительно закрываю оставшиеся позиции: {len(open_positions)}")

    for position in open_positions:
        secid = position["secid"]

        if secid in minute_by_secid.index:
            row = minute_by_secid.loc[secid]
            close_price = float(row["close"])

            if np.isfinite(close_price) and close_price > 0:
                position["last_price"] = close_price

        exit_price = position["last_price"]

        cash = close_position(
            begin=final_time,
            cash=cash,
            position=position,
            current_close=exit_price,
            sell_cost=sell_cost,
            close_reason="end_of_period",
            close_forecast=np.nan,
            close_horizon=np.nan,
            trade_events=trade_events,
            closed_trades=closed_trades,
        )

    return cash, []


def save_backtest_details(
        split: str,
        threshold_bp: float,
        max_positions: int,
        buy_cost_bp: float,
        sell_cost_bp: float,
        trade_events_df: pd.DataFrame,
        closed_trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        save_details: bool,
        results_dir: Path,
) -> None:
    """
    Сохраняет подробные журналы сделок и кривую капитала, если это включено для split.
    """
    if not save_details:
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    prefix = (
        f"{split}_"
        f"thr_{threshold_bp}_"
        f"pos_{max_positions}_"
        f"buy_{buy_cost_bp}_"
        f"sell_{sell_cost_bp}"
    )

    log(f"Сохраняю подробные результаты: {prefix}")

    trade_events_df.to_csv(
        results_dir / f"{prefix}_events.csv",
        index=False,
        encoding="utf-8-sig",
    )

    closed_trades_df.to_csv(
        results_dir / f"{prefix}_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )

    equity_df.to_csv(
        results_dir / f"{prefix}_equity.csv",
        index=False,
        encoding="utf-8-sig",
    )

def safe_divide(
        numerator: float,
        denominator: float,
        default: float = np.nan,
) -> float:
    """
    Безопасно делит одно число на другое.
    """
    if denominator == 0:
        return default

    if not np.isfinite(denominator):
        return default

    return numerator / denominator


def calculate_backtest_metrics(
        equity_df: pd.DataFrame,
        trade_events_df: pd.DataFrame,
        closed_trades_df: pd.DataFrame,
        initial_cash: float,
) -> dict[str, Any]:
    """
    Считает расширенный набор метрик по результатам backtest.
    """
    metrics: dict[str, Any] = {}

    # ============================================================
    # Метрики по кривой капитала
    # ============================================================

    if equity_df.empty:
        metrics.update({
            "start_time": np.nan,
            "end_time": np.nan,
            "unique_minutes": 0,

            "final_equity": initial_cash,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "return_to_drawdown": np.nan,

            "avg_cash": 0.0,
            "min_cash": np.nan,
            "max_cash": np.nan,

            "avg_position_value": 0.0,
            "max_position_value": np.nan,

            "avg_equity": initial_cash,
            "min_equity": np.nan,
            "max_equity": np.nan,

            "avg_cash_share": np.nan,
            "avg_position_value_share": np.nan,

            "avg_open_positions": 0.0,
            "max_open_positions": 0,
            "time_in_market_share": 0.0,

            "minute_return_mean_bp": np.nan,
            "minute_return_std_bp": np.nan,
            "minute_return_min_bp": np.nan,
            "minute_return_max_bp": np.nan,
        })
    else:
        equity = equity_df["equity"].astype(float)
        drawdown = equity / equity.cummax() - 1.0

        minute_returns = equity.pct_change().dropna()

        final_equity = float(equity.iloc[-1])
        total_return = final_equity / initial_cash - 1.0
        max_drawdown = float(drawdown.min())

        avg_cash = float(equity_df["cash"].mean())
        avg_position_value = float(equity_df["position_value"].mean())
        avg_equity = float(equity_df["equity"].mean())

        metrics.update({
            "start_time": equity_df["begin"].iloc[0],
            "end_time": equity_df["begin"].iloc[-1],
            "unique_minutes": len(equity_df),

            "final_equity": final_equity,
            "total_return_pct": total_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "return_to_drawdown": safe_divide(total_return, abs(max_drawdown)),

            "avg_cash": avg_cash,
            "min_cash": float(equity_df["cash"].min()),
            "max_cash": float(equity_df["cash"].max()),

            "avg_position_value": avg_position_value,
            "max_position_value": float(equity_df["position_value"].max()),

            "avg_equity": avg_equity,
            "min_equity": float(equity_df["equity"].min()),
            "max_equity": float(equity_df["equity"].max()),

            "avg_cash_share": safe_divide(avg_cash, avg_equity),
            "avg_position_value_share": safe_divide(
                avg_position_value,
                avg_equity,
            ),

            "avg_open_positions": float(equity_df["open_positions"].mean()),
            "max_open_positions": int(equity_df["open_positions"].max()),
            "time_in_market_share": float(
                (equity_df["open_positions"] > 0).mean()
            ),

            "minute_return_mean_bp": (
                float(minute_returns.mean() * 10_000)
                if not minute_returns.empty else np.nan
            ),
            "minute_return_std_bp": (
                float(minute_returns.std(ddof=0) * 10_000)
                if not minute_returns.empty else np.nan
            ),
            "minute_return_min_bp": (
                float(minute_returns.min() * 10_000)
                if not minute_returns.empty else np.nan
            ),
            "minute_return_max_bp": (
                float(minute_returns.max() * 10_000)
                if not minute_returns.empty else np.nan
            ),
        })

    # ============================================================
    # Метрики по событиям BUY/SELL
    # ============================================================

    if trade_events_df.empty:
        buy_events = 0
        sell_events = 0
        first_event_time = np.nan
        last_event_time = np.nan
    else:
        buy_events = int((trade_events_df["event"] == "BUY").sum())
        sell_events = int((trade_events_df["event"] == "SELL").sum())
        first_event_time = trade_events_df["time"].iloc[0]
        last_event_time = trade_events_df["time"].iloc[-1]

    metrics.update({
        "buy_events": buy_events,
        "sell_events": sell_events,
        "event_imbalance": buy_events - sell_events,
        "first_event_time": first_event_time,
        "last_event_time": last_event_time,
    })

    # ============================================================
    # Метрики по закрытым сделкам
    # ============================================================

    if closed_trades_df.empty:
        metrics.update({
            "trades": 0,
            "unique_secids_traded": 0,

            "gross_pnl": 0.0,
            "gross_pnl_pct": 0.0,
            "total_buy_cost": 0.0,
            "total_sell_cost": 0.0,
            "total_cost": 0.0,
            "total_cost_pct": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,

            "cost_to_gross_pnl": np.nan,

            "mean_trade_return_bp": np.nan,
            "median_trade_return_bp": np.nan,
            "std_trade_return_bp": np.nan,
            "min_trade_return_bp": np.nan,
            "max_trade_return_bp": np.nan,

            "trade_return_q05_bp": np.nan,
            "trade_return_q25_bp": np.nan,
            "trade_return_q75_bp": np.nan,
            "trade_return_q95_bp": np.nan,

            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_win_bp": np.nan,
            "avg_loss_bp": np.nan,
            "payoff_ratio": np.nan,

            "mean_holding_minutes": np.nan,
            "median_holding_minutes": np.nan,
            "holding_minutes_q90": np.nan,
            "max_holding_minutes_actual": np.nan,

            "total_cash_spent": 0.0,
            "total_exit_cash": 0.0,
            "avg_cash_spent_per_trade": np.nan,
            "turnover": 0.0,

            "mean_open_forecast_bp": np.nan,
            "median_open_forecast_bp": np.nan,
            "mean_close_forecast_bp": np.nan,
            "median_close_forecast_bp": np.nan,

            "mean_open_horizon": np.nan,
            "median_open_horizon": np.nan,
            "mean_close_horizon": np.nan,
            "median_close_horizon": np.nan,

            "closed_by_max_hold": 0,
            "closed_by_forecast": 0,
            "closed_by_end": 0,
            "closed_by_max_hold_share": np.nan,
            "closed_by_forecast_share": np.nan,
            "closed_by_end_share": np.nan,
        })

        return metrics

    trades = len(closed_trades_df)

    # gross_pnl — прибыль/убыток без издержек.
    # exit_cash + sell_cost = стоимость продажи до вычета sell_cost.
    # trade_value = стоимость купленных бумаг без buy_cost.
    gross_pnl = (
        closed_trades_df["exit_cash"]
        + closed_trades_df["sell_cost"]
        - closed_trades_df["trade_value"]
    ).sum()

    total_buy_cost = closed_trades_df["buy_cost"].sum()
    total_sell_cost = closed_trades_df["sell_cost"].sum()
    total_cost = total_buy_cost + total_sell_cost

    total_pnl = closed_trades_df["pnl"].sum()

    trade_returns = closed_trades_df["return"].astype(float)
    trade_returns_bp = trade_returns * 10_000

    winning_returns_bp = trade_returns_bp[closed_trades_df["pnl"] > 0]
    losing_returns_bp = trade_returns_bp[closed_trades_df["pnl"] < 0]

    profitable_pnl = closed_trades_df.loc[
        closed_trades_df["pnl"] > 0,
        "pnl",
    ].sum()

    losing_pnl_abs = -closed_trades_df.loc[
        closed_trades_df["pnl"] < 0,
        "pnl",
    ].sum()

    profit_factor = safe_divide(profitable_pnl, losing_pnl_abs)

    avg_win_bp = (
        float(winning_returns_bp.mean())
        if len(winning_returns_bp) > 0 else np.nan
    )
    avg_loss_bp = (
        float(losing_returns_bp.mean())
        if len(losing_returns_bp) > 0 else np.nan
    )

    payoff_ratio = safe_divide(
        avg_win_bp,
        abs(avg_loss_bp) if np.isfinite(avg_loss_bp) else np.nan,
    )

    total_cash_spent = closed_trades_df["cash_spent"].sum()
    total_exit_cash = closed_trades_df["exit_cash"].sum()

    close_reason_counts = (
        closed_trades_df["close_reason"]
        .value_counts()
        .to_dict()
    )

    closed_by_max_hold = int(
        close_reason_counts.get("max_holding_minutes", 0)
    )
    closed_by_forecast = int(
        close_reason_counts.get("hold_forecast_not_enough", 0)
    )
    closed_by_end = int(
        close_reason_counts.get("end_of_period", 0)
    )

    close_forecast = pd.to_numeric(
        closed_trades_df["close_forecast"],
        errors="coerce",
    )

    close_horizon = pd.to_numeric(
        closed_trades_df["close_horizon"],
        errors="coerce",
    )

    metrics.update({
        "trades": trades,
        "unique_secids_traded": int(closed_trades_df["secid"].nunique()),

        "gross_pnl": float(gross_pnl),
        "gross_pnl_pct": float(gross_pnl / initial_cash * 100),
        "total_buy_cost": float(total_buy_cost),
        "total_sell_cost": float(total_sell_cost),
        "total_cost": float(total_cost),
        "total_cost_pct": float(total_cost / initial_cash * 100),
        "total_pnl": float(total_pnl),
        "total_pnl_pct": float(total_pnl / initial_cash * 100),

        "cost_to_gross_pnl": safe_divide(total_cost, gross_pnl),

        "mean_trade_return_bp": float(trade_returns_bp.mean()),
        "median_trade_return_bp": float(trade_returns_bp.median()),
        "std_trade_return_bp": float(trade_returns_bp.std(ddof=0)),
        "min_trade_return_bp": float(trade_returns_bp.min()),
        "max_trade_return_bp": float(trade_returns_bp.max()),

        "trade_return_q05_bp": float(trade_returns_bp.quantile(0.05)),
        "trade_return_q25_bp": float(trade_returns_bp.quantile(0.25)),
        "trade_return_q75_bp": float(trade_returns_bp.quantile(0.75)),
        "trade_return_q95_bp": float(trade_returns_bp.quantile(0.95)),

        "win_rate": float((closed_trades_df["pnl"] > 0).mean()),
        "profit_factor": profit_factor,
        "avg_win_bp": avg_win_bp,
        "avg_loss_bp": avg_loss_bp,
        "payoff_ratio": payoff_ratio,

        "mean_holding_minutes": float(
            closed_trades_df["holding_minutes"].mean()
        ),
        "median_holding_minutes": float(
            closed_trades_df["holding_minutes"].median()
        ),
        "holding_minutes_q90": float(
            closed_trades_df["holding_minutes"].quantile(0.90)
        ),
        "max_holding_minutes_actual": float(
            closed_trades_df["holding_minutes"].max()
        ),

        "total_cash_spent": float(total_cash_spent),
        "total_exit_cash": float(total_exit_cash),
        "avg_cash_spent_per_trade": safe_divide(total_cash_spent, trades),
        "turnover": float(
            (total_cash_spent + total_exit_cash) / initial_cash
        ),

        "mean_open_forecast_bp": float(
            closed_trades_df["open_forecast"].mean() * 10_000
        ),
        "median_open_forecast_bp": float(
            closed_trades_df["open_forecast"].median() * 10_000
        ),
        "mean_close_forecast_bp": float(close_forecast.mean() * 10_000),
        "median_close_forecast_bp": float(close_forecast.median() * 10_000),

        "mean_open_horizon": float(
            closed_trades_df["open_horizon"].mean()
        ),
        "median_open_horizon": float(
            closed_trades_df["open_horizon"].median()
        ),
        "mean_close_horizon": float(close_horizon.mean()),
        "median_close_horizon": float(close_horizon.median()),

        "closed_by_max_hold": closed_by_max_hold,
        "closed_by_forecast": closed_by_forecast,
        "closed_by_end": closed_by_end,

        "closed_by_max_hold_share": safe_divide(
            closed_by_max_hold,
            trades,
        ),
        "closed_by_forecast_share": safe_divide(
            closed_by_forecast,
            trades,
        ),
        "closed_by_end_share": safe_divide(
            closed_by_end,
            trades,
        ),
    })

    return metrics


def run_stateful_portfolio_backtest(
        name: str,
        data: pd.DataFrame,
        parameter_grid: list[dict[str, float | int]],
        progress_every_minutes: int,
        save_details: bool,
        model_name: str,
        results_dir: Path,
        initial_cash: float = INITIAL_CASH,
) -> pd.DataFrame:
    """
    Выполняет историческое тестирование stateful long-only стратегии.
    """
    log(f"Запускаю backtest для split={name}")

    work_cols = [
        "begin",
        "secid",
        "close",
        "next_open",
        "open_score",
        "open_horizon",
        "hold_score",
        "hold_horizon",
    ]

    work = data[work_cols].copy()

    grouped_by_minute = work.groupby("begin", sort=False)
    unique_minutes = work["begin"].nunique()
    final_time = work["begin"].iloc[-1]

    log(f"{name}: work shape={work.shape}")
    log(f"{name}: уникальных минут={unique_minutes}")
    log(f"{name}: final_time={final_time}")
    log(f"{name}: комбинаций параметров={len(parameter_grid)}")

    summary_rows = []

    for combo_idx, params in enumerate(parameter_grid, start=1):
        threshold_bp = float(params["threshold_bp"])
        max_positions = int(params["max_positions"])
        buy_cost_bp = float(params["buy_cost_bp"])
        sell_cost_bp = float(params["sell_cost_bp"])

        threshold = bp_to_return(threshold_bp)
        buy_cost = bp_to_return(buy_cost_bp)
        sell_cost = bp_to_return(sell_cost_bp)

        log(
            f"{name}: combo {combo_idx}/{len(parameter_grid)}: "
            f"threshold={threshold_bp}bp, "
            f"max_positions={max_positions}, "
            f"buy_cost={buy_cost_bp}bp, "
            f"sell_cost={sell_cost_bp}bp"
        )

        cash = initial_cash
        open_positions = []

        equity_history = []
        trade_events = []
        closed_trades = []

        last_minute_by_secid = None

        opened_total = 0
        closed_total = 0
        closed_by_max_hold = 0
        closed_by_forecast = 0

        minute_idx = 0

        for begin, minute_data in grouped_by_minute:
            minute_idx += 1

            if (
                    minute_idx == 1
                    or minute_idx % progress_every_minutes == 0
                    or minute_idx == unique_minutes
            ):
                log(
                    f"{name}: combo {combo_idx}/{len(parameter_grid)}, "
                    f"minute {minute_idx}/{unique_minutes}, "
                    f"begin={begin}, "
                    f"cash={cash:.2f}, "
                    f"open_positions={len(open_positions)}, "
                    f"opened_total={opened_total}, "
                    f"closed_total={closed_total}"
                )

            minute_by_secid = minute_data.set_index(
                "secid",
                drop=False,
            )

            last_minute_by_secid = minute_by_secid

            (
                cash,
                open_positions,
                closed_count,
                max_hold_closed_count,
                forecast_closed_count,
            ) = try_close_positions(
                begin=begin,
                cash=cash,
                open_positions=open_positions,
                minute_by_secid=minute_by_secid,
                sell_cost=sell_cost,
                trade_events=trade_events,
                closed_trades=closed_trades,
            )

            closed_total += closed_count
            closed_by_max_hold += max_hold_closed_count
            closed_by_forecast += forecast_closed_count

            equity_before_open, _position_value_before_open = (
                mark_to_market(
                    cash=cash,
                    open_positions=open_positions,
                    begin=begin,
                    minute_by_secid=minute_by_secid,
                    sell_cost=sell_cost,
                )
            )

            cash, open_positions, opened_count = try_open_positions(
                begin=begin,
                final_time=final_time,
                cash=cash,
                open_positions=open_positions,
                minute_data=minute_data,
                minute_by_secid=minute_by_secid,
                portfolio_value=equity_before_open,
                threshold=threshold,
                max_positions=max_positions,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                trade_events=trade_events,
            )

            opened_total += opened_count

            equity, position_value = mark_to_market(
                cash=cash,
                open_positions=open_positions,
                begin=begin,
                minute_by_secid=minute_by_secid,
                sell_cost=sell_cost,
            )

            equity_history.append({
                "begin": begin,
                "cash": cash,
                "position_value": position_value,
                "equity": equity,
                "open_positions": len(open_positions),
            })

        if open_positions:
            cash, open_positions = close_all_positions_at_end(
                final_time=final_time,
                cash=cash,
                open_positions=open_positions,
                minute_by_secid=last_minute_by_secid,
                sell_cost=sell_cost,
                trade_events=trade_events,
                closed_trades=closed_trades,
            )

            equity_history.append({
                "begin": final_time,
                "cash": cash,
                "position_value": 0.0,
                "equity": cash,
                "open_positions": 0,
            })

        equity_df = pd.DataFrame(equity_history)
        trade_events_df = pd.DataFrame(trade_events)
        closed_trades_df = pd.DataFrame(closed_trades)

        metrics = calculate_backtest_metrics(
            equity_df=equity_df,
            trade_events_df=trade_events_df,
            closed_trades_df=closed_trades_df,
            initial_cash=initial_cash,
        )

        log(
            f"{name}: combo {combo_idx}/{len(parameter_grid)} завершена: "
            f"trades={metrics['trades']}, "
            f"final_equity={metrics['final_equity']:.2f}, "
            f"return={metrics['total_return_pct']:.4f}%, "
            f"max_drawdown={metrics['max_drawdown_pct']:.4f}%, "
            f"profit_factor={metrics['profit_factor']:.4f}, "
            f"opened_total={opened_total}, "
            f"closed_total={closed_total}, "
            f"closed_by_max_hold={closed_by_max_hold}, "
            f"closed_by_forecast={closed_by_forecast}"
        )

        summary_row = {
            "model_name": model_name,
            "split": name,
            "threshold_bp": threshold_bp,
            "max_positions": max_positions,
            "buy_cost_bp": buy_cost_bp,
            "sell_cost_bp": sell_cost_bp,

            "opened_total": opened_total,
            "closed_total": closed_total,
            "closed_by_max_hold_loop": closed_by_max_hold,
            "closed_by_forecast_loop": closed_by_forecast,
        }

        summary_row.update(metrics)

        summary_rows.append(summary_row)

        save_backtest_details(
            split=name,
            threshold_bp=threshold_bp,
            max_positions=max_positions,
            buy_cost_bp=buy_cost_bp,
            sell_cost_bp=sell_cost_bp,
            trade_events_df=trade_events_df,
            closed_trades_df=closed_trades_df,
            equity_df=equity_df,
            save_details=save_details,
            results_dir=results_dir,
        )

    summary = pd.DataFrame(summary_rows)

    log(f"{name}: все комбинации параметров завершены")

    print(f"\nРезультаты backtest: {name}")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 300,
        "display.float_format", "{:.6f}".format,
    ):
        print(summary.to_string(index=False))

    return summary


def select_best_validation_config(
        valid_summary: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Выбирает лучшую конфигурацию по validation для последующей проверки на test.
    """
    if valid_summary.empty:
        raise ValueError("valid_summary пустой, выбрать параметры невозможно")

    candidates = valid_summary.copy()

    if "trades" in candidates.columns:
        candidates = candidates[candidates["trades"] > 0]

    if candidates.empty:
        raise ValueError("На validation нет ни одной конфигурации со сделками")

    candidates = candidates.sort_values(
        by=["total_return_pct", "max_drawdown_pct"],
        ascending=[False, False],
    )

    best = candidates.iloc[0]

    best_config = {
        "threshold_bp": float(best["threshold_bp"]),
        "max_positions": int(best["max_positions"]),
        "buy_cost_bp": float(best["buy_cost_bp"]),
        "sell_cost_bp": float(best["sell_cost_bp"]),
    }

    log(f"Лучшая validation-конфигурация: {best_config}")

    return best_config



def get_prediction_data_path(
        model_name: str,
        split_name: str,
) -> Path:
    """
    Возвращает путь к parquet-файлу с прогнозами модели для указанной выборки.
    """
    return DATA_ROOT / model_name / f"{split_name}_predictions.parquet"


def validate_prediction_data(
        df: pd.DataFrame,
        model_name: str,
        split_name: str,
) -> None:
    """
    Проверяет, что файл прогнозов содержит всё необходимое для backtest.
    """
    required_cols = [
        "begin",
        "secid",
        "close",
        "next_open",
        "open_score",
        "open_horizon",
        "hold_score",
        "hold_horizon",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"В прогнозах модели {model_name} для split={split_name} "
            f"нет колонок: {missing_cols}"
        )

    nan_rows = df[required_cols].isna().any(axis=1).sum()
    if nan_rows > 0:
        raise ValueError(
            f"В прогнозах модели {model_name} для split={split_name} "
            f"найдено {nan_rows} строк с NaN."
        )

    numeric_cols = [
        "close",
        "next_open",
        "open_score",
        "open_horizon",
        "hold_score",
        "hold_horizon",
    ]

    finite_mask = np.isfinite(df[numeric_cols].to_numpy()).all(axis=1)
    bad_rows = len(df) - finite_mask.sum()

    if bad_rows > 0:
        raise ValueError(
            f"В прогнозах модели {model_name} для split={split_name} "
            f"найдено {bad_rows} строк с inf/-inf."
        )


def read_prediction_dataset(
        model_name: str,
        split_name: str,
        debug_max_minutes: int | None,
) -> pd.DataFrame:
    """
    Читает valid/test файл с уже построенными прогнозами модели.
    """
    path = get_prediction_data_path(model_name, split_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл прогнозов не найден: {path}. "
            f"Сначала запусти train_models.py для модели {model_name}."
        )

    log(f"{model_name}: читаю прогнозы {split_name}: {path}")

    df = pd.read_parquet(path)

    log(f"{model_name}: {split_name} predictions shape={df.shape}")

    validate_prediction_data(
        df=df,
        model_name=model_name,
        split_name=split_name,
    )

    if debug_max_minutes is not None:
        log(
            f"{model_name}: {split_name}: "
            f"debug_max_minutes={debug_max_minutes}, обрезаю выборку"
        )

        unique_minutes = df["begin"].drop_duplicates()

        if len(unique_minutes) > debug_max_minutes:
            last_allowed_minute = unique_minutes.iloc[debug_max_minutes - 1]
            df = df[df["begin"] <= last_allowed_minute].copy()

            log(
                f"{model_name}: {split_name}: после обрезки shape={df.shape}, "
                f"last_allowed_minute={last_allowed_minute}"
            )

    log(
        f"{model_name}: {split_name}: диапазон begin: "
        f"{df['begin'].min()} -> {df['begin'].max()}"
    )
    log(f"{model_name}: {split_name}: уникальных минут: {df['begin'].nunique()}")
    log(f"{model_name}: {split_name}: уникальных бумаг: {df['secid'].nunique()}")

    return df


def process_split(
        split_name: str,
        model_name: str,
        parameter_grid: list[dict[str, float | int]],
        active_settings: dict[str, Any],
        results_dir: Path,
) -> pd.DataFrame:
    """
    Запускает backtest для одной выборки на уже готовых прогнозах модели.
    """
    log(f"{model_name}: начинаю обработку split={split_name}")

    df = read_prediction_dataset(
        model_name=model_name,
        split_name=split_name,
        debug_max_minutes=active_settings["debug_max_minutes"],
    )

    if split_name == "valid":
        save_details = active_settings["save_validation_details"]
    elif split_name == "test":
        save_details = active_settings["save_test_details"]
    else:
        raise ValueError(f"Неизвестная выборка: {split_name}")

    summary = run_stateful_portfolio_backtest(
        name=split_name,
        data=df,
        parameter_grid=parameter_grid,
        progress_every_minutes=active_settings["progress_every_minutes"],
        save_details=save_details,
        model_name=model_name,
        results_dir=results_dir,
        initial_cash=INITIAL_CASH,
    )

    log(f"{model_name}: обработка split={split_name} завершена")

    return summary


def save_run_settings(
        active_settings: dict[str, Any],
        best_config: dict[str, float | int],
) -> None:
    """
    Сохраняет настройки запуска и выбранную конфигурацию стратегии.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    run_settings = {
        "model_name": model_name,
        "partial_backtest": PARTIAL_BACKTEST,
        "active_settings": active_settings,
        "open_score_mode": OPEN_SCORE_MODE,
        "open_top_p": OPEN_TOP_P,
        "open_top_m": OPEN_TOP_M,
        "hold_score_mode": HOLD_SCORE_MODE,
        "hold_top_p": HOLD_TOP_P,
        "hold_top_m": HOLD_TOP_M,
        "hold_extra_buffer_bp": HOLD_EXTRA_BUFFER_BP,
        "max_holding_minutes": MAX_HOLDING_MINUTES,
        "max_instrument_share": MAX_INSTRUMENT_SHARE,
        "max_instrument_cap_rub": MAX_INSTRUMENT_CAP_RUB,
        "spend_on_minute_rub": SPEND_ON_MINUTE_RUB,
        "spend_on_minute_share": SPEND_ON_MINUTE_SHARE,
        "min_trade_rub": MIN_TRADE_RUB,
        "best_validation_config": best_config,
    }

    with open(results_dir / "run_settings.json", "w", encoding="utf-8") as f:
        json.dump(run_settings, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Для каждой выбранной модели запускает validation-grid, выбирает лучшие
    параметры отдельно для каждой пары издержек и проверяет их на test.
    """
    log("Старт backtest_strategy.py")

    active_settings = get_active_backtest_settings()

    log(
        f"Режим тестирования: {active_settings['name']} "
        f"(PARTIAL_BACKTEST={PARTIAL_BACKTEST})"
    )
    log(f"Модели для backtest: {MODELS_TO_BACKTEST}")
    log(f"Активные thresholds_bp: {active_settings['thresholds_bp']}")
    log(f"Активные max_positions_list: {active_settings['max_positions_list']}")
    log(f"Активные cost_bp_pairs: {active_settings['cost_bp_pairs']}")
    log("Для каждой cost_bp_pair параметры подбираются на validation отдельно")
    log(f"debug_max_minutes: {active_settings['debug_max_minutes']}")

    validation_grid = build_parameter_grid(
        thresholds_bp=active_settings["thresholds_bp"],
        max_positions_list=active_settings["max_positions_list"],
        cost_bp_pairs=active_settings["cost_bp_pairs"],
    )

    log(f"Validation-grid: {len(validation_grid)} конфигураций")

    all_valid_summaries = []
    all_test_summaries = []

    for model_name in MODELS_TO_BACKTEST:
        log("=" * 80)
        log(f"Запускаю backtest для модели: {model_name}")

        model_results_dir = RESULTS_DIR / model_name
        model_results_dir.mkdir(parents=True, exist_ok=True)

        log(f"{model_name}: запускаю validation backtest")
        valid_summary = process_split(
            split_name="valid",
            model_name=model_name,
            parameter_grid=validation_grid,
            active_settings=active_settings,
            results_dir=model_results_dir,
        )

        selected_configs_by_cost = select_best_validation_configs_by_cost(
            valid_summary
        )

        test_grid = selected_configs_by_cost

        log(
            f"{model_name}: запускаю test backtest: для каждой пары издержек "
            "используются threshold_bp и max_positions, подобранные на validation"
        )
        log(f"{model_name}: Test-grid: {len(test_grid)} конфигураций")
        log(f"{model_name}: Test-grid configs: {test_grid}")

        test_summary = process_split(
            split_name="test",
            model_name=model_name,
            parameter_grid=test_grid,
            active_settings=active_settings,
            results_dir=model_results_dir,
        )

        log(f"{model_name}: сохраняю valid_summary.csv")
        valid_summary.to_csv(
            model_results_dir / "valid_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        log(f"{model_name}: сохраняю test_summary.csv")
        test_summary.to_csv(
            model_results_dir / "test_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        log(f"{model_name}: сохраняю selected_configs_by_cost.json")
        with open(
                model_results_dir / "selected_configs_by_cost.json",
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(selected_configs_by_cost, f, ensure_ascii=False, indent=2)

        log(f"{model_name}: сохраняю selected_test_grid.json")
        with open(model_results_dir / "selected_test_grid.json", "w", encoding="utf-8") as f:
            json.dump(test_grid, f, ensure_ascii=False, indent=2)

        log(f"{model_name}: сохраняю run_settings.json")
        save_run_settings(
            active_settings=active_settings,
            selected_configs_by_cost=selected_configs_by_cost,
            model_name=model_name,
            results_dir=model_results_dir,
        )

        all_valid_summaries.append(valid_summary)
        all_test_summaries.append(test_summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if all_valid_summaries:
        combined_valid = pd.concat(all_valid_summaries, ignore_index=True)
        combined_valid.to_csv(
            RESULTS_DIR / "all_models_valid_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_test_summaries:
        combined_test = pd.concat(all_test_summaries, ignore_index=True)
        combined_test.to_csv(
            RESULTS_DIR / "all_models_test_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    log("Backtest завершён")
    print("\nBacktest завершён")
    print("Итоги сохранены в папку:", RESULTS_DIR)


if __name__ == "__main__":
    main()