import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from constants import HORIZON
from validation_hyperparameters import (
    RUN_VALIDATION_BACKTEST,
    RUN_TEST_BACKTEST,
    VALIDATION_PARAMETER_GRIDS,
)

try:
    from validation_hyperparameters import BACKTEST_STRATEGIES as CONFIG_BACKTEST_STRATEGIES
except ImportError:
    CONFIG_BACKTEST_STRATEGIES = None


DATA_ROOT = Path("data")

# Список моделей и grid торговых гиперпараметров берутся из
# validation_hyperparameters.py.
# backtest_strategy.py больше не обучает модели и не строит predictions.
# Перед запуском backtest сначала запусти train_models.py.

RESULTS_DIR = Path("backtest_results")

BEST_VALIDATION_HYPERPARAMETERS_PY_PATH = Path("best_validation_hyperparameters.py")
BEST_VALIDATION_HYPERPARAMETERS_TEXT_PATH = Path("best_validation_hyperparameters.txt")

INITIAL_CASH = 1_000_000.0

# ============================================================
# Главный массив: что именно прогонять в backtest
# ============================================================

# Здесь в одном месте перечислены все проверяемые подходы:
# 1) rule-based стратегии;
# 2) flat ML-модели;
# 3) нейросетевые sequence-модели.
BACKTEST_STRATEGIES = [
    "momentum",
    "mean_reversion",
    "ma_trend",
    "breakout",

    "ridge",
    "catboost",
    "random_forest",
    "decision_tree",

    "gru",
    "lstm",
    "transformer",
    "tcn",
]

if CONFIG_BACKTEST_STRATEGIES is not None:
    BACKTEST_STRATEGIES = list(CONFIG_BACKTEST_STRATEGIES)

RULE_BASED_STRATEGIES = {
    "momentum",
    "mean_reversion",
    "ma_trend",
    "breakout",
}

MODEL_BASED_STRATEGIES = {
    "ridge",
    "catboost",
    "random_forest",
    "decision_tree",
    "gru",
    "lstm",
    "transformer",
    "tcn",
}

STRATEGY_LABELS = {
    "momentum": "Momentum",
    "mean_reversion": "Mean Reversion",
    "ma_trend": "MA Trend",
    "breakout": "Breakout",
    "ridge": "Ridge",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "tcn": "TCN",
}



# ============================================================
# Pipeline
# ============================================================

# Обучение моделей и построение prediction-файлов выполняет только
# train_models.py. Этот файл читает уже готовые:
#   data/<model_name>/valid_predictions.parquet
#   data/<model_name>/test_predictions.parquet
# и выполняет только историческое тестирование торговой стратегии.


# ============================================================
# Главный переключатель режима
# ============================================================

# True  -> быстрый частичный прогон для отладки.
# False -> полный прогон для итоговых результатов.
PARTIAL_BACKTEST = False


FULL_BACKTEST_SETTINGS = {
    "name": "full",
    "debug_max_minutes": None,
    "progress_every_minutes": 10_000,
    "save_validation_details": True,
    "save_test_details": True,
}

PARTIAL_BACKTEST_SETTINGS = {
    "name": "partial",
    "debug_max_minutes": 20_000,
    "progress_every_minutes": 2_000,
    "save_validation_details": True,
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

LOG_FILE_PATH = Path("logs.txt")
RESET_LOG_ON_START = True

# Количество параллельных workers для перебора торговых параметров.
# 1  -> последовательный режим, как раньше.
# >1 -> параллельно тестируются несколько комбинаций параметров.
BACKTEST_N_JOBS = 16

# "process" обычно быстрее для CPU-bound backtest, но потребляет больше памяти.
# "thread" потребляет меньше памяти, но ускорение может быть слабее из-за GIL.
BACKTEST_PARALLEL_BACKEND = "process"


def normalize_model_name(model_name: str) -> str:
    """
    Приводит имя стратегии/модели к каноническому виду.
    """
    normalized_name = str(model_name).lower()

    aliases = {
        "momentum": "momentum",
        "mean_reversion": "mean_reversion",
        "meanreversion": "mean_reversion",
        "ma_trend": "ma_trend",
        "matrend": "ma_trend",
        "breakout": "breakout",
        "ridge": "ridge",
        "ridge_regression": "ridge",
        "ridgeregression": "ridge",
        "catboost": "catboost",
        "cat_boost": "catboost",
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
        f"Неизвестная стратегия/модель: {model_name}. "
        f"Допустимые значения: {sorted(aliases)}."
    )


def is_rule_based_strategy(name: str) -> bool:
    return normalize_model_name(name) in RULE_BASED_STRATEGIES


def is_model_based_strategy(name: str) -> bool:
    return normalize_model_name(name) in MODEL_BASED_STRATEGIES


def get_active_backtest_settings() -> dict[str, Any]:
    """
    Возвращает активный набор параметров для полного или частичного тестирования.
    """
    if PARTIAL_BACKTEST:
        return PARTIAL_BACKTEST_SETTINGS

    return FULL_BACKTEST_SETTINGS


def get_model_backtest_settings(
        active_settings: dict[str, Any],
        model_name: str,
) -> dict[str, Any]:
    """
    Возвращает технические настройки backtest для конкретной модели.

    Торговый grid threshold/max_positions/cost берётся отдельно из
    validation_hyperparameters.py.
    """
    _normalized_name = normalize_model_name(model_name)
    return dict(active_settings)

def safe_print(message: str = "") -> None:
    """
    Безопасно печатает сообщение в консоль.

    Если PyCharm/Windows stdout падает с OSError,
    backtest не останавливается.
    """
    try:
        print(message)
    except OSError:
        pass


def safe_write_log_file(message: str) -> None:
    """
    Безопасно дописывает строку лога в logs.txt.

    Файл открывается на каждую запись отдельно. Это проще и устойчивее при
    параллельном выполнении нескольких workers.
    """
    if LOG_FILE_PATH is None:
        return

    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except OSError:
        pass


def reset_log_file() -> None:
    """
    Очищает logs.txt в начале нового запуска, если включён RESET_LOG_ON_START.
    """
    if LOG_FILE_PATH is None or not RESET_LOG_ON_START:
        return

    try:
        LOG_FILE_PATH.write_text("", encoding="utf-8")
    except OSError:
        pass


def log(message: str) -> None:
    """
    Безопасно печатает диагностическое сообщение в консоль и logs.txt.

    Если PyCharm/Windows stdout падает с OSError,
    backtest не останавливается.
    """
    if not VERBOSE:
        return

    now = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{now}][pid={os.getpid()}] {message}"
    safe_print(formatted)
    safe_write_log_file(formatted)


def get_model_file_prefix(model_name: str) -> str:
    """
    Возвращает префикс имён файлов для конкретной стратегии/модели.
    """
    normalized_name = normalize_model_name(model_name)

    prefixes = {
        "momentum": "momentum",
        "mean_reversion": "mean_reversion",
        "ma_trend": "ma_trend",
        "breakout": "breakout",
        "ridge": "ridge",
        "catboost": "catboost",
        "random_forest": "random_forest",
        "decision_tree": "decision_tree",
        "gru": "GRU",
        "lstm": "LSTM",
        "transformer": "transformer",
        "tcn": "TCN",
    }

    return prefixes[normalized_name]


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




def normalize_parameter_config(
        config: dict[str, float | int],
        model_name: str,
) -> dict[str, float | int]:
    """
    Проверяет и нормализует одну комбинацию торговых параметров.
    """
    required_keys = [
        "threshold_bp",
        "max_positions",
        "buy_cost_bp",
        "sell_cost_bp",
    ]

    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(
            f"В validation_hyperparameters.py для модели {model_name} "
            f"нет ключей: {missing_keys}"
        )

    threshold_bp = float(config["threshold_bp"])
    max_positions = int(config["max_positions"])
    buy_cost_bp = float(config["buy_cost_bp"])
    sell_cost_bp = float(config["sell_cost_bp"])

    if max_positions <= 0:
        raise ValueError(
            f"Для модели {model_name} max_positions должен быть > 0, "
            f"получено {max_positions}."
        )

    return {
        "threshold_bp": threshold_bp,
        "max_positions": max_positions,
        "buy_cost_bp": buy_cost_bp,
        "sell_cost_bp": sell_cost_bp,
    }


def get_model_parameter_grid(model_name: str) -> list[dict[str, float | int]]:
    """
    Возвращает grid торговых гиперпараметров модели из
    validation_hyperparameters.py.
    """
    normalized_name = normalize_model_name(model_name)

    if normalized_name not in VALIDATION_PARAMETER_GRIDS:
        raise ValueError(
            f"В validation_hyperparameters.py нет grid для модели "
            f"{normalized_name}."
        )

    raw_grid = VALIDATION_PARAMETER_GRIDS[normalized_name]

    if not raw_grid:
        raise ValueError(
            f"В validation_hyperparameters.py grid для модели "
            f"{normalized_name} пустой."
        )

    return [
        normalize_parameter_config(config=config, model_name=normalized_name)
        for config in raw_grid
    ]

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
    Считает длительность удержания позиции в торговых минутах.

    Важно: используется не календарная разница begin - entry_time, а счётчик
    торговых шагов. Иначе позиции, перенесённые через ночь или иной разрыв
    в данных, могли получать holding_minutes_actual в сотни минут при лимите 30.
    """
    if not is_position_active(position, begin):
        return 0.0

    return float(position.get("holding_steps", 0))


def increment_position_holding_steps(
        open_positions: list[dict[str, Any]],
        begin: pd.Timestamp,
) -> None:
    """
    Увеличивает счётчик торговых минут удержания для активных позиций.
    """
    for position in open_positions:
        if is_position_active(position, begin):
            position["holding_steps"] = int(position.get("holding_steps", 0)) + 1


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

        holding_minutes = get_holding_minutes(position, begin)

        if secid in minute_by_secid.index:
            row = minute_by_secid.loc[secid]
            current_close = float(row["close"])

            if np.isfinite(current_close) and current_close > 0:
                position["last_price"] = current_close
            else:
                current_close = float(position["last_price"])
        else:
            current_close = float(position["last_price"])

        if not np.isfinite(current_close) or current_close <= 0:
            still_open_positions.append(position)
            continue

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

        if secid not in minute_by_secid.index:
            still_open_positions.append(position)
            continue

        row = minute_by_secid.loc[secid]

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
            "holding_steps": 0,
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
        model_name: str,
) -> None:
    """
    Сохраняет подробные журналы сделок и кривую капитала, если это включено для split.
    """
    if not save_details:
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = get_model_file_prefix(model_name)

    prefix = (
        f"{file_prefix}_"
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



_PARALLEL_BACKTEST_WORK: pd.DataFrame | None = None


def _init_parallel_backtest_worker(work: pd.DataFrame) -> None:
    """
    Инициализирует общий DataFrame в worker-процессе.

    Это позволяет не передавать большой набор данных отдельно в каждую задачу.
    """
    global _PARALLEL_BACKTEST_WORK
    _PARALLEL_BACKTEST_WORK = work


def _run_single_parameter_combo_from_global(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Запускает одну комбинацию параметров в worker-процессе.
    """
    if _PARALLEL_BACKTEST_WORK is None:
        raise RuntimeError("Worker не получил DataFrame для backtest.")

    return run_single_parameter_combo(
        work=_PARALLEL_BACKTEST_WORK,
        **kwargs,
    )


def run_single_parameter_combo(
        work: pd.DataFrame,
        name: str,
        params: dict[str, float | int],
        combo_idx: int,
        total_combos: int,
        progress_every_minutes: int,
        save_details: bool,
        model_name: str,
        results_dir: Path,
        initial_cash: float = INITIAL_CASH,
) -> dict[str, Any]:
    """
    Выполняет backtest для одной комбинации торговых параметров.

    Эта функция вынесена отдельно, чтобы её можно было запускать параллельно
    для разных threshold/max_positions/cost.
    """
    grouped_by_minute = work.groupby("begin", sort=False)
    unique_minutes = work["begin"].nunique()
    final_time = work["begin"].iloc[-1]

    threshold_bp = float(params["threshold_bp"])
    max_positions = int(params["max_positions"])
    buy_cost_bp = float(params["buy_cost_bp"])
    sell_cost_bp = float(params["sell_cost_bp"])

    threshold = bp_to_return(threshold_bp)
    buy_cost = bp_to_return(buy_cost_bp)
    sell_cost = bp_to_return(sell_cost_bp)

    log(
        f"{name}: combo {combo_idx}/{total_combos}: "
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

    last_minute_by_secid: pd.DataFrame | None = None

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
                f"{name}: combo {combo_idx}/{total_combos}, "
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

        increment_position_holding_steps(
            open_positions=open_positions,
            begin=begin,
        )

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
        if last_minute_by_secid is None:
            raise RuntimeError(
                "Невозможно закрыть позиции в конце периода: "
                "нет последней minute_by_secid."
            )

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
        f"{name}: combo {combo_idx}/{total_combos} завершена: "
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
        "_combo_idx": combo_idx,
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
        model_name=model_name,
    )

    return summary_row


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

    Если BACKTEST_N_JOBS > 1, разные комбинации threshold/max_positions/cost
    выполняются параллельно.
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

    unique_minutes = work["begin"].nunique()
    final_time = work["begin"].iloc[-1]

    log(f"{name}: work shape={work.shape}")
    log(f"{name}: уникальных минут={unique_minutes}")
    log(f"{name}: final_time={final_time}")
    log(f"{name}: комбинаций параметров={len(parameter_grid)}")
    log(
        f"{name}: BACKTEST_N_JOBS={BACKTEST_N_JOBS}, "
        f"BACKTEST_PARALLEL_BACKEND={BACKTEST_PARALLEL_BACKEND}"
    )

    summary_rows: list[dict[str, Any]] = []
    total_combos = len(parameter_grid)

    if BACKTEST_N_JOBS <= 1 or total_combos <= 1:
        for combo_idx, params in enumerate(parameter_grid, start=1):
            summary_rows.append(run_single_parameter_combo(
                work=work,
                name=name,
                params=params,
                combo_idx=combo_idx,
                total_combos=total_combos,
                progress_every_minutes=progress_every_minutes,
                save_details=save_details,
                model_name=model_name,
                results_dir=results_dir,
                initial_cash=initial_cash,
            ))
    else:
        max_workers = min(int(BACKTEST_N_JOBS), total_combos)

        task_kwargs = [
            {
                "name": name,
                "params": params,
                "combo_idx": combo_idx,
                "total_combos": total_combos,
                "progress_every_minutes": progress_every_minutes,
                "save_details": save_details,
                "model_name": model_name,
                "results_dir": results_dir,
                "initial_cash": initial_cash,
            }
            for combo_idx, params in enumerate(parameter_grid, start=1)
        ]

        log(
            f"{name}: запускаю параллельный backtest: "
            f"workers={max_workers}, backend={BACKTEST_PARALLEL_BACKEND}"
        )

        if BACKTEST_PARALLEL_BACKEND == "thread":
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        run_single_parameter_combo,
                        work=work,
                        **kwargs,
                    )
                    for kwargs in task_kwargs
                ]

                for future in as_completed(futures):
                    summary_rows.append(future.result())

        elif BACKTEST_PARALLEL_BACKEND == "process":
            with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_parallel_backtest_worker,
                    initargs=(work,),
            ) as executor:
                futures = [
                    executor.submit(
                        _run_single_parameter_combo_from_global,
                        kwargs,
                    )
                    for kwargs in task_kwargs
                ]

                for future in as_completed(futures):
                    summary_rows.append(future.result())

        else:
            raise ValueError(
                "BACKTEST_PARALLEL_BACKEND должен быть 'process' или 'thread', "
                f"получено: {BACKTEST_PARALLEL_BACKEND!r}"
            )

    summary_rows = sorted(summary_rows, key=lambda row: row["_combo_idx"])

    for row in summary_rows:
        row.pop("_combo_idx", None)

    summary = pd.DataFrame(summary_rows)

    log(f"{name}: все комбинации параметров завершены")

    safe_print(f"\nРезультаты backtest: {name}")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 300,
        "display.float_format", "{:.6f}".format,
    ):
        safe_print(summary.to_string(index=False))

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


def select_best_validation_configs_by_cost(
        valid_summary: pd.DataFrame,
) -> list[dict[str, float | int]]:
    """
    Для каждой пары издержек отдельно выбирает лучшие threshold_bp
    и max_positions на validation. Эти же параметры затем используются на test.
    """
    if valid_summary.empty:
        raise ValueError("valid_summary пустой, выбрать параметры невозможно")

    candidates = valid_summary.copy()

    if "trades" in candidates.columns:
        candidates = candidates[candidates["trades"] > 0]

    if candidates.empty:
        raise ValueError("На validation нет ни одной конфигурации со сделками")

    best_configs: list[dict[str, float | int]] = []

    grouped = candidates.groupby(
        ["buy_cost_bp", "sell_cost_bp"],
        sort=True,
        dropna=False,
    )

    for (buy_cost_bp, sell_cost_bp), group in grouped:
        sorted_group = group.sort_values(
            by=["total_return_pct", "max_drawdown_pct"],
            ascending=[False, False],
        )

        best = sorted_group.iloc[0]

        best_config = {
            "threshold_bp": float(best["threshold_bp"]),
            "max_positions": int(best["max_positions"]),
            "buy_cost_bp": float(buy_cost_bp),
            "sell_cost_bp": float(sell_cost_bp),
        }

        log(
            "Лучшая validation-конфигурация для издержек "
            f"buy={buy_cost_bp}bp, sell={sell_cost_bp}bp: {best_config}"
        )

        best_configs.append(best_config)

    return best_configs


# ============================================================
# Rule-based стратегии: чтение данных и построение score
# ============================================================


def get_split_data_candidates(split_name: str) -> list[Path]:
    """
    Ищет данные для rule-based стратегий.

    Лучший вариант — отдельный parquet с признаками valid/test. Если его нет,
    пробуем использовать prediction-файлы обучаемых моделей, если в них остались
    базовые колонки.
    """
    candidates = [
        DATA_ROOT / f"{split_name}_dataset.parquet",
        DATA_ROOT / f"{split_name}.parquet",
        DATA_ROOT / "datasets" / f"{split_name}_dataset.parquet",
        DATA_ROOT / "datasets" / f"{split_name}.parquet",
        DATA_ROOT / "features" / f"{split_name}_features.parquet",
        DATA_ROOT / "features" / f"{split_name}.parquet",
    ]

    for model_name in [
        "ridge",
        "catboost",
        "random_forest",
        "decision_tree",
        "gru",
        "lstm",
        "transformer",
        "tcn",
    ]:
        candidates.append(DATA_ROOT / model_name / f"{split_name}_predictions.parquet")

    candidates.extend([
        Path(f"{split_name}_dataset.parquet"),
        Path(f"{split_name}.parquet"),
        Path(f"{split_name}_predictions.parquet"),
    ])

    return candidates


def find_split_data_path(split_name: str) -> Path:
    for path in get_split_data_candidates(split_name):
        if path.exists():
            return path

    candidates_text = "\n".join(str(path) for path in get_split_data_candidates(split_name))
    raise FileNotFoundError(
        f"Не найден файл данных для rule-based split={split_name}. Проверялись пути:\n{candidates_text}"
    )


def read_split_dataset(split_name: str, debug_max_minutes: int | None) -> pd.DataFrame:
    path = find_split_data_path(split_name)
    log(f"Читаю данные для rule-based split={split_name}: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    for col in ["begin", "secid", "close"]:
        if col not in df.columns:
            raise ValueError(f"В файле {path} нет колонки {col}")

    df = df.copy()
    df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
    df = df.dropna(subset=["begin", "secid", "close"]).copy()
    df = df.sort_values(["begin", "secid"]).reset_index(drop=True)

    if debug_max_minutes is not None:
        unique_minutes = df["begin"].drop_duplicates()
        if len(unique_minutes) > debug_max_minutes:
            last_allowed_minute = unique_minutes.iloc[debug_max_minutes - 1]
            df = df[df["begin"] <= last_allowed_minute].copy()
            log(
                f"{split_name}: debug_max_minutes={debug_max_minutes}, "
                f"last_allowed_minute={last_allowed_minute}, shape={df.shape}"
            )

    log(
        f"{split_name}: shape={df.shape}, "
        f"begin={df['begin'].min()} -> {df['begin'].max()}, "
        f"минут={df['begin'].nunique()}, бумаг={df['secid'].nunique()}"
    )

    return df


def ensure_trade_date(df: pd.DataFrame) -> None:
    if "trade_date" not in df.columns or df["trade_date"].isna().any():
        df["trade_date"] = df["begin"].dt.date


def ensure_next_open(df: pd.DataFrame) -> None:
    if "next_open" in df.columns:
        return

    if "open" not in df.columns:
        raise ValueError(
            "Для backtest нужна колонка next_open. Её нет, а open тоже нет, "
            "поэтому next_open невозможно восстановить."
        )

    group_keys = ["secid", "trade_date"]
    df["next_open"] = df.groupby(group_keys, sort=False)["open"].shift(-1)


def add_return_feature(df: pd.DataFrame, lag: int) -> None:
    col = f"return_{lag}"
    if col in df.columns:
        return

    group_keys = ["secid", "trade_date"]
    df[col] = (
        df.groupby(group_keys, sort=False)["close"]
        .pct_change(periods=lag)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def add_ma_feature(df: pd.DataFrame, window: int) -> None:
    ma_col = f"ma_close_{window}"
    if ma_col not in df.columns:
        group_keys = ["secid", "trade_date"]
        df[ma_col] = (
            df.groupby(group_keys, sort=False)["close"]
            .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        )

    ratio_col = f"close_to_ma_{window}"
    if ratio_col not in df.columns:
        df[ratio_col] = (
            df["close"] / df[ma_col].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_breakout_feature(df: pd.DataFrame, window: int = 15) -> None:
    col = f"breakout_{window}"
    if col in df.columns:
        return

    group_keys = ["secid", "trade_date"]
    high_col = "high" if "high" in df.columns else "close"
    rolling_high_col = f"rolling_high_{window}"

    df[rolling_high_col] = (
        df.groupby(group_keys, sort=False)[high_col]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).max())
    )

    df[col] = (
        df["close"] / df[rolling_high_col].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_rule_based_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ensure_trade_date(df)
    ensure_next_open(df)
    add_return_feature(df, 15)
    for window in [5, 15, 30]:
        add_ma_feature(df, window)
    add_breakout_feature(df, window=15)
    return df


def get_strategy_score(df: pd.DataFrame, strategy_name: str) -> pd.Series:
    normalized_name = normalize_model_name(strategy_name)

    if normalized_name == "momentum":
        score = df["return_15"]
    elif normalized_name == "mean_reversion":
        score = -df["close_to_ma_15"]
    elif normalized_name == "ma_trend":
        score = df["ma_close_5"] / df["ma_close_30"].replace(0, np.nan) - 1.0
    elif normalized_name == "breakout":
        score = df["breakout_15"]
    else:
        raise ValueError(f"Неизвестная rule-based стратегия: {strategy_name}")

    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def build_rule_based_backtest_dataset(
        base_df: pd.DataFrame,
        strategy_name: str,
) -> pd.DataFrame:
    df = add_rule_based_features(base_df)
    score = get_strategy_score(df, strategy_name)

    out = pd.DataFrame({
        "begin": df["begin"],
        "secid": df["secid"],
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "next_open": pd.to_numeric(df["next_open"], errors="coerce"),
        "open_score": score,
        "open_horizon": HORIZON,
        "hold_score": score,
        "hold_horizon": HORIZON,
    })

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

    out = out.dropna(subset=required_cols).copy()
    numeric_cols = [
        "close",
        "next_open",
        "open_score",
        "open_horizon",
        "hold_score",
        "hold_horizon",
    ]
    finite_mask = np.isfinite(out[numeric_cols].to_numpy()).all(axis=1)
    out = out[finite_mask].copy()
    out = out.sort_values(["begin", "secid"]).reset_index(drop=True)

    return out


def get_prediction_data_path(
        model_name: str,
        split_name: str,
) -> Path:
    """
    Возвращает путь к parquet-файлу с прогнозами модели для указанной выборки.
    """
    normalized_name = normalize_model_name(model_name)

    return DATA_ROOT / normalized_name / f"{split_name}_predictions.parquet"


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
    Запускает backtest для одной выборки.

    Для model-based стратегий читает готовые prediction-файлы из data/<model>/.
    Для rule-based стратегий строит open_score/hold_score из исторических признаков.
    """
    normalized_name = normalize_model_name(model_name)
    log(f"{normalized_name}: начинаю обработку split={split_name}")

    if is_rule_based_strategy(normalized_name):
        base_df = read_split_dataset(
            split_name=split_name,
            debug_max_minutes=active_settings["debug_max_minutes"],
        )
        df = build_rule_based_backtest_dataset(
            base_df=base_df,
            strategy_name=normalized_name,
        )
    else:
        df = read_prediction_dataset(
            model_name=normalized_name,
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
        model_name=normalized_name,
        results_dir=results_dir,
        initial_cash=INITIAL_CASH,
    )

    log(f"{normalized_name}: обработка split={split_name} завершена")

    return summary


def format_parameter_config(config: dict[str, float | int]) -> str:
    """
    Форматирует одну комбинацию параметров для консоли и txt-файла.
    """
    return (
        f"threshold_bp={float(config['threshold_bp']):.6g}, "
        f"max_positions={int(config['max_positions'])}, "
        f"buy_cost_bp={float(config['buy_cost_bp']):.6g}, "
        f"sell_cost_bp={float(config['sell_cost_bp']):.6g}"
    )


def safe_print_best_validation_configs(
        model_name: str,
        selected_configs_by_cost: list[dict[str, float | int]],
) -> None:
    """
    Выводит лучшие validation-параметры модели в консоль.
    """
    safe_print()
    safe_print("=" * 80)
    safe_print(f"Лучшие validation-гиперпараметры для модели {model_name}")

    for config in selected_configs_by_cost:
        safe_print("  " + format_parameter_config(config))

    safe_print("=" * 80)
    safe_print()


def build_best_hyperparameters_text(
        best_parameter_grids_by_model: dict[str, list[dict[str, float | int]]],
) -> str:
    """
    Создаёт человекочитаемый текст с лучшими параметрами.
    """
    lines = [
        "Лучшие торговые гиперпараметры по результатам validation",
        "",
    ]

    for model_name, configs in best_parameter_grids_by_model.items():
        lines.append(f"[{model_name}]")

        for config in configs:
            lines.append("  " + format_parameter_config(config))

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_best_hyperparameters_py(
        best_parameter_grids_by_model: dict[str, list[dict[str, float | int]]],
) -> str:
    """
    Создаёт Python-файл, которым можно заменить validation_hyperparameters.py.
    """
    import pprint

    models = list(best_parameter_grids_by_model.keys())
    models_repr = pprint.pformat(models, width=88, sort_dicts=False)
    grids_repr = pprint.pformat(
        best_parameter_grids_by_model,
        width=88,
        sort_dicts=False,
    )

    return (
        "from __future__ import annotations\n\n"
        "# Этот файл автоматически создан backtest_strategy.py по результатам validation.\n"
        "# Им можно заменить validation_hyperparameters.py, чтобы прогнать только\n"
        "# лучшие найденные параметры и затем выполнить test backtest.\n\n"
        "# Этот файл предназначен для финального test-прогона без повторной validation.\n"
        "RUN_VALIDATION_BACKTEST = False\n"
        "RUN_TEST_BACKTEST = True\n\n"
        f"BACKTEST_STRATEGIES = {models_repr}\n\n"
        f"VALIDATION_PARAMETER_GRIDS = {grids_repr}\n\n"
        "BEST_VALIDATION_CONFIGS_BY_MODEL = VALIDATION_PARAMETER_GRIDS\n"
    )


def save_best_validation_hyperparameter_files(
        best_parameter_grids_by_model: dict[str, list[dict[str, float | int]]],
) -> None:
    """
    Сохраняет лучшие validation-параметры в txt и в Python-модуль.
    """
    if not best_parameter_grids_by_model:
        return

    text_content = build_best_hyperparameters_text(best_parameter_grids_by_model)
    py_content = build_best_hyperparameters_py(best_parameter_grids_by_model)

    BEST_VALIDATION_HYPERPARAMETERS_TEXT_PATH.write_text(
        text_content,
        encoding="utf-8",
    )
    BEST_VALIDATION_HYPERPARAMETERS_PY_PATH.write_text(
        py_content,
        encoding="utf-8",
    )


    log(
        "Лучшие validation-гиперпараметры сохранены: "
        f"{BEST_VALIDATION_HYPERPARAMETERS_TEXT_PATH}, "
        f"{BEST_VALIDATION_HYPERPARAMETERS_PY_PATH}"
    )


def save_run_settings(
        active_settings: dict[str, Any],
        selected_configs_by_cost: list[dict[str, float | int]],
        selected_configs_source: str,
        model_name: str,
        results_dir: Path,
) -> None:
    """
    Сохраняет настройки запуска и список конфигураций,
    выбранных на validation отдельно для каждой пары издержек.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    run_settings = {
        "model_name": model_name,
        "partial_backtest": PARTIAL_BACKTEST,
        "run_validation_backtest": RUN_VALIDATION_BACKTEST,
        "run_test_backtest": RUN_TEST_BACKTEST,
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
        "selected_configs_source": selected_configs_source,
        "selected_configs_by_cost": selected_configs_by_cost,
    }

    file_prefix = get_model_file_prefix(model_name)
    path = results_dir / f"{file_prefix}_run_settings.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_settings, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Выполняет только backtest на уже готовых prediction-файлах.

    train_models.py отвечает за обучение моделей и создание:
    1. data/<model_name>/valid_predictions.parquet
       — прогнозы validation-модели, обученной на train.
    2. data/<model_name>/test_predictions.parquet
       — прогнозы final-модели, обученной на train + validation.

    backtest_strategy.py:
    1. берёт grid торговых параметров из validation_hyperparameters.py;
    2. если RUN_VALIDATION_BACKTEST=True, подбирает threshold_bp и
       max_positions на validation;
    3. если validation запускалась, сохраняет лучшие параметры в
       best_validation_hyperparameters.py и best_validation_hyperparameters.txt;
    4. если RUN_TEST_BACKTEST=True, прогоняет test:
       - после validation — на лучших validation-параметрах;
       - без validation — на параметрах из VALIDATION_PARAMETER_GRIDS.
    """
    reset_log_file()
    log("Старт backtest_strategy.py")

    active_settings = get_active_backtest_settings()

    log(
        f"Режим тестирования: {active_settings['name']} "
        f"(PARTIAL_BACKTEST={PARTIAL_BACKTEST})"
    )
    log(f"RUN_VALIDATION_BACKTEST={RUN_VALIDATION_BACKTEST}")
    log(f"RUN_TEST_BACKTEST={RUN_TEST_BACKTEST}")
    log(f"Стратегии/модели для backtest: {BACKTEST_STRATEGIES}")
    log("Grid торговых гиперпараметров: validation_hyperparameters.py")
    log("Обучение моделей в этом файле не выполняется")
    log(f"debug_max_minutes: {active_settings['debug_max_minutes']}")
    log(f"BACKTEST_N_JOBS={BACKTEST_N_JOBS}")
    log(f"BACKTEST_PARALLEL_BACKEND={BACKTEST_PARALLEL_BACKEND}")
    log(f"LOG_FILE_PATH={LOG_FILE_PATH}")

    available_grid_names = {
        normalize_model_name(name)
        for name in VALIDATION_PARAMETER_GRIDS.keys()
    }

    active_backtest_strategies = [
        strategy
        for strategy in BACKTEST_STRATEGIES
        if normalize_model_name(strategy) in available_grid_names
    ]

    skipped_strategies = [
        strategy
        for strategy in BACKTEST_STRATEGIES
        if normalize_model_name(strategy) not in available_grid_names
    ]

    if skipped_strategies:
        log(
            "Пропускаю стратегии/модели без grid в validation_hyperparameters.py: "
            f"{skipped_strategies}"
        )

    if not active_backtest_strategies:
        raise ValueError(
            "Нет ни одной стратегии/модели с grid в validation_hyperparameters.py."
        )

    all_valid_summaries = []
    all_test_summaries = []
    best_parameter_grids_by_model: dict[str, list[dict[str, float | int]]] = {}

    for model_name in active_backtest_strategies:
        normalized_model_name = normalize_model_name(model_name)

        log("=" * 80)
        log(f"Запускаю backtest для модели: {model_name}")

        model_active_settings = get_model_backtest_settings(
            active_settings=active_settings,
            model_name=normalized_model_name,
        )

        validation_grid = get_model_parameter_grid(normalized_model_name)

        log(f"{model_name}: Validation-grid: {len(validation_grid)} конфигураций")
        log(f"{model_name}: Validation-grid configs: {validation_grid}")

        model_file_prefix = get_model_file_prefix(normalized_model_name)

        model_results_dir = RESULTS_DIR / normalized_model_name
        model_results_dir.mkdir(parents=True, exist_ok=True)

        valid_summary = None

        if RUN_VALIDATION_BACKTEST:
            log(f"{model_name}: запускаю validation backtest")

            valid_summary = process_split(
                split_name="valid",
                model_name=normalized_model_name,
                parameter_grid=validation_grid,
                active_settings=model_active_settings,
                results_dir=model_results_dir,
            )

            selected_configs_by_cost = select_best_validation_configs_by_cost(
                valid_summary
            )

            best_parameter_grids_by_model[normalized_model_name] = (
                selected_configs_by_cost
            )

            safe_print_best_validation_configs(
                model_name=normalized_model_name,
                selected_configs_by_cost=selected_configs_by_cost,
            )

            valid_summary_path = (
                model_results_dir / f"{model_file_prefix}_valid_summary.csv"
            )
            log(f"{model_name}: сохраняю {valid_summary_path.name}")
            valid_summary.to_csv(
                valid_summary_path,
                index=False,
                encoding="utf-8-sig",
            )

            selected_configs_source = "validation"
        else:
            log(
                f"{model_name}: RUN_VALIDATION_BACKTEST=False, "
                "пропускаю validation backtest"
            )
            log(
                f"{model_name}: для test будут использованы параметры "
                "из VALIDATION_PARAMETER_GRIDS без дополнительного отбора"
            )

            selected_configs_by_cost = validation_grid
            selected_configs_source = "validation_hyperparameters.py"

        test_grid = selected_configs_by_cost

        selected_configs_path = (
            model_results_dir
            / f"{model_file_prefix}_selected_configs_by_cost.json"
        )
        log(f"{model_name}: сохраняю {selected_configs_path.name}")
        with open(
                selected_configs_path,
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(selected_configs_by_cost, f, ensure_ascii=False, indent=2)

        selected_test_grid_path = (
            model_results_dir
            / f"{model_file_prefix}_selected_test_grid.json"
        )
        log(f"{model_name}: сохраняю {selected_test_grid_path.name}")
        with open(selected_test_grid_path, "w", encoding="utf-8") as f:
            json.dump(test_grid, f, ensure_ascii=False, indent=2)

        test_summary = None

        if RUN_TEST_BACKTEST:
            log(
                f"{model_name}: запускаю test backtest: "
                f"источник параметров — {selected_configs_source}"
            )

            test_summary = process_split(
                split_name="test",
                model_name=normalized_model_name,
                parameter_grid=test_grid,
                active_settings=model_active_settings,
                results_dir=model_results_dir,
            )

            test_summary_path = model_results_dir / f"{model_file_prefix}_test_summary.csv"
            log(f"{model_name}: сохраняю {test_summary_path.name}")
            test_summary.to_csv(
                test_summary_path,
                index=False,
                encoding="utf-8-sig",
            )
        else:
            log(
                f"{model_name}: RUN_TEST_BACKTEST=False, пропускаю test backtest"
            )

        log(f"{model_name}: сохраняю {model_file_prefix}_run_settings.json")
        save_run_settings(
            active_settings=model_active_settings,
            selected_configs_by_cost=selected_configs_by_cost,
            selected_configs_source=selected_configs_source,
            model_name=normalized_model_name,
            results_dir=model_results_dir,
        )

        if valid_summary is not None:
            all_valid_summaries.append(valid_summary)

        if test_summary is not None:
            all_test_summaries.append(test_summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if all_valid_summaries:
        combined_valid = pd.concat(all_valid_summaries, ignore_index=True)
        combined_valid.to_csv(
            RESULTS_DIR / "all_strategies_valid_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_test_summaries:
        combined_test = pd.concat(all_test_summaries, ignore_index=True)
        combined_test.to_csv(
            RESULTS_DIR / "all_strategies_test_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_best_validation_hyperparameter_files(
        best_parameter_grids_by_model=best_parameter_grids_by_model,
    )

    log("Backtest завершён")
    log(f"Итоги сохранены в папку: {RESULTS_DIR}")

if __name__ == "__main__":
    main()