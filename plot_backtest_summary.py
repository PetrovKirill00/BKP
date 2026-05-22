from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Настройки
# ============================================================

PRIMARY_RESULTS_DIR = Path("backtest_results")
FALLBACK_RESULTS_DIR = Path(".")
RESULTS_DIR = PRIMARY_RESULTS_DIR
PLOTS_DIR = PRIMARY_RESULTS_DIR / "plots"

SUMMARY_CANDIDATES = [
    PRIMARY_RESULTS_DIR / "all_strategies_test_summary.csv",
    FALLBACK_RESULTS_DIR / "all_strategies_test_summary.csv",
    PRIMARY_RESULTS_DIR / "all_models_test_summary.csv",
    FALLBACK_RESULTS_DIR / "all_models_test_summary.csv",
]

# Берём один уровень издержек, чтобы на каждом графике было ровно 12 столбцов.
# 3 б.п. — средний сценарий из трёх рассмотренных: 1, 3 и 5 б.п.
SELECTED_COST_BP = 3.0

STRATEGY_ORDER = [
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

# Все подписи на русском, кроме CatBoost, TCN, GRU и LSTM.
STRATEGY_LABELS = {
    "momentum": "Моментум",
    "mean_reversion": "Возврат к среднему",
    "ma_trend": "Тренд по средним",
    "breakout": "Пробой",
    "ridge": "Гребневая регрессия",
    "catboost": "CatBoost",
    "random_forest": "Случайный лес",
    "decision_tree": "Дерево решений",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Трансформер",
    "tcn": "TCN",
}

CATEGORY_BOUNDARIES = [3.5, 7.5]
CATEGORY_LABELS = [
    ("Правиловые стратегии", 1.5),
    ("Машинное обучение", 5.5),
    ("Нейросетевые стратегии", 9.5),
]

# Контрастные цвета по группам:
# 4 rule-based, 4 ML, 4 neural.
BAR_COLORS = [
    "#1f77b4", "#2ca02c", "#17becf", "#9467bd",
    "#d62728", "#ff7f0e", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#005f73", "#9b2226",
]


# ============================================================
# Чтение summary
# ============================================================


def read_summary() -> pd.DataFrame:
    global RESULTS_DIR, PLOTS_DIR

    summary_path: Path | None = None

    for candidate in SUMMARY_CANDIDATES:
        if candidate.exists():
            summary_path = candidate
            break

    if summary_path is None:
        raise FileNotFoundError(
            "Не найден all_strategies_test_summary.csv. "
            "Сначала запусти backtest_strategy.py."
        )

    if summary_path.parent == FALLBACK_RESULTS_DIR:
        RESULTS_DIR = FALLBACK_RESULTS_DIR
        PLOTS_DIR = RESULTS_DIR / "plots"

    df = pd.read_csv(summary_path)

    required_cols = [
        "model_name",
        "split",
        "buy_cost_bp",
        "sell_cost_bp",
        "total_return_pct",
        "max_drawdown_pct",
        "trades",
        "total_cost_pct",
        "profit_factor",
        "mean_holding_minutes",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"В summary нет обязательных колонок: {missing_cols}")

    if "time_in_market_share" not in df.columns and "time_in_market_pct" not in df.columns:
        raise ValueError(
            "В summary нет колонки time_in_market_share или time_in_market_pct."
        )

    numeric_cols = [
        "buy_cost_bp",
        "sell_cost_bp",
        "total_return_pct",
        "max_drawdown_pct",
        "trades",
        "total_cost_pct",
        "profit_factor",
        "mean_holding_minutes",
        "time_in_market_share",
        "time_in_market_pct",
        "final_equity",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "time_in_market_pct" not in df.columns:
        df["time_in_market_pct"] = df["time_in_market_share"] * 100.0

    df["model_name"] = df["model_name"].astype(str).str.lower()
    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    df = df[df["buy_cost_bp"] == df["sell_cost_bp"]].copy()
    df = df[df["buy_cost_bp"] == SELECTED_COST_BP].copy()
    df = df[df["model_name"].isin(STRATEGY_ORDER)].copy()

    if df.empty:
        raise ValueError(
            f"В summary нет test-строк для издержек {SELECTED_COST_BP} б.п. "
            "и выбранных 12 стратегий."
        )

    # Если для одной стратегии почему-то несколько строк, оставляем лучшую
    # по total_return_pct, а если такой колонки нет — первую.
    df = df.sort_values(
        ["model_name", "total_return_pct"],
        ascending=[True, False],
    )
    df = df.drop_duplicates(["model_name"], keep="first")

    df["model_name"] = pd.Categorical(
        df["model_name"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    df = df.sort_values("model_name").reset_index(drop=True)

    if len(df) != len(STRATEGY_ORDER):
        present = set(df["model_name"].astype(str))
        missing = [name for name in STRATEGY_ORDER if name not in present]
        raise ValueError(f"В summary нет строк для стратегий: {missing}")

    return df


# ============================================================
# Визуальные helper-функции
# ============================================================


def add_group_separators(ax) -> None:
    for x in CATEGORY_BOUNDARIES:
        ax.axvline(
            x,
            linestyle="--",
            linewidth=1,
            alpha=0.45,
            color="#5dade2",
        )

    y_min, y_max = ax.get_ylim()
    y_text = y_max - (y_max - y_min) * 0.04

    for label, x in CATEGORY_LABELS:
        ax.text(
            x,
            y_text,
            label,
            ha="center",
            va="top",
            fontsize=9,
        )


def format_axes(ax, title: str, ylabel: str) -> None:
    x = np.arange(len(STRATEGY_ORDER))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [STRATEGY_LABELS[name] for name in STRATEGY_ORDER],
        rotation=25,
        ha="right",
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    add_group_separators(ax)


def save_barplot(
        df: pd.DataFrame,
        value_col: str,
        title: str,
        ylabel: str,
        filename: str,
        transform_values: Callable[[np.ndarray], np.ndarray] | None = None,
        baseline: float | None = None,
        log_scale: bool = False,
) -> Path:
    x = np.arange(len(STRATEGY_ORDER))
    values = df[value_col].astype(float).to_numpy()

    if transform_values is not None:
        values = transform_values(values)

    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)

    ax.bar(
        x,
        values,
        color=BAR_COLORS,
        edgecolor="black",
        linewidth=0.5,
    )

    if baseline is not None:
        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1.2,
            color="#333333",
        )

    if log_scale:
        ax.set_yscale("log")

    format_axes(ax=ax, title=title, ylabel=ylabel)

    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return out_path


# ============================================================
# Построение отдельных графиков для показателей из раздела 3.4
# ============================================================


def save_total_return_plot(df: pd.DataFrame) -> Path:
    return save_barplot(
        df=df,
        value_col="total_return_pct",
        title=f"Итоговая доходность стратегий на тестовой выборке, издержки {SELECTED_COST_BP:g} б.п.",
        ylabel="Итоговая доходность, %",
        filename="total_return_by_strategy_3bp_v2.png",
        baseline=0.0,
    )


def save_max_drawdown_plot(df: pd.DataFrame) -> Path:
    return save_barplot(
        df=df,
        value_col="max_drawdown_pct",
        title=f"Максимальная просадка стратегий на тестовой выборке, издержки {SELECTED_COST_BP:g} б.п.",
        ylabel="Максимальная просадка, %",
        filename="max_drawdown_by_strategy_3bp_v2.png",
        transform_values=np.abs,
    )


def save_trades_plot(df: pd.DataFrame) -> Path:
    return save_barplot(
        df=df,
        value_col="trades",
        title=f"Суммарное количество сделок на тестовой выборке, издержки {SELECTED_COST_BP:g} б.п.",
        ylabel="Количество закрытых сделок",
        filename="trades_by_strategy_3bp_v2.png",
        log_scale=True,
    )


def save_total_cost_plot(df: pd.DataFrame) -> Path:
    return save_barplot(
        df=df,
        value_col="total_cost_pct",
        title=f"Суммарные торговые издержки на тестовой выборке, издержки {SELECTED_COST_BP:g} б.п.",
        ylabel="Издержки, % от начального капитала",
        filename="total_cost_by_strategy_3bp_v2.png",
    )


def save_profit_factor_plot(df: pd.DataFrame) -> Path:
    return save_barplot(
        df=df,
        value_col="profit_factor",
        title=f"Фактор прибыльности стратегий на тестовой выборке, издержки {SELECTED_COST_BP:g} б.п.",
        ylabel="Фактор прибыльности",
        filename="profit_factor_by_strategy_3bp_v2.png",
        baseline=1.0,
    )


def save_mean_holding_minutes_plot(df: pd.DataFrame) -> Path:
    return save_barplot(
        df=df,
        value_col="mean_holding_minutes",
        title=f"Среднее время удержания позиции на тестовой выборке, издержки {SELECTED_COST_BP:g} б.п.",
        ylabel="Среднее время удержания, минут",
        filename="mean_holding_minutes_by_strategy_3bp_v2.png",
    )


def save_time_in_market_plot(df: pd.DataFrame) -> Path:
    return save_barplot(
        df=df,
        value_col="time_in_market_pct",
        title=f"Доля времени нахождения в рынке на тестовой выборке, издержки {SELECTED_COST_BP:g} б.п.",
        ylabel="Доля времени в рынке, %",
        filename="time_in_market_by_strategy_3bp_v2.png",
    )


def main() -> None:
    df = read_summary()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = [
        save_total_return_plot(df),
        save_max_drawdown_plot(df),
        save_trades_plot(df),
        save_total_cost_plot(df),
        save_profit_factor_plot(df),
        save_mean_holding_minutes_plot(df),
        save_time_in_market_plot(df),
    ]

    for path in paths:
        print(f"Сохранено: {path}")


if __name__ == "__main__":
    main()
