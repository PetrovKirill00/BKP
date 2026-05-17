from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from constants import HORIZON
except Exception:
    HORIZON = 15

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

FILE_PREFIXES = {
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

COST_ORDER = [1.0, 3.0, 5.0]
COST_LABELS = {1.0: "0,01%", 3.0: "0,03%", 5.0: "0,05%"}


def read_summary() -> pd.DataFrame:
    summary_path = None
    global RESULTS_DIR, PLOTS_DIR

    for candidate in SUMMARY_CANDIDATES:
        if candidate.exists():
            summary_path = candidate
            break

    if summary_path is None:
        raise FileNotFoundError(
            "Не найден all_strategies_test_summary.csv. "
            "Сначала запусти backtest_strategy.py"
        )

    if summary_path.parent == FALLBACK_RESULTS_DIR:
        RESULTS_DIR = FALLBACK_RESULTS_DIR
        PLOTS_DIR = RESULTS_DIR / "plots"

    df = pd.read_csv(summary_path)

    numeric_cols = [
        "buy_cost_bp",
        "sell_cost_bp",
        "threshold_bp",
        "max_positions",
        "total_return_pct",
        "max_drawdown_pct",
        "profit_factor",
        "trades",
        "total_cost_pct",
        "mean_holding_minutes",
        "time_in_market_share",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["model_name"] = df["model_name"].astype(str).str.lower()
    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    df = df[df["buy_cost_bp"] == df["sell_cost_bp"]].copy()
    df = df[df["buy_cost_bp"].isin(COST_ORDER)].copy()
    df = df[df["model_name"].isin(STRATEGY_ORDER)].copy()

    if df.empty:
        raise ValueError(
            "В summary нет test-строк для выбранных стратегий и издержек "
            "0,01%, 0,03%, 0,05%."
        )

    df["cost_bp"] = df["buy_cost_bp"]
    if "time_in_market_share" in df.columns:
        df["time_in_market_pct"] = df["time_in_market_share"] * 100.0

    df["model_name"] = pd.Categorical(
        df["model_name"],
        STRATEGY_ORDER,
        ordered=True,
    )
    df["cost_bp"] = pd.Categorical(df["cost_bp"], COST_ORDER, ordered=True)
    df = df.sort_values(["model_name", "cost_bp"]).reset_index(drop=True)

    return df


def grouped_values(df: pd.DataFrame, value_col: str) -> dict[float, list[float]]:
    values: dict[float, list[float]] = {}
    for cost in COST_ORDER:
        part = df[df["cost_bp"] == cost].set_index("model_name")
        row_values = []
        for model in STRATEGY_ORDER:
            if model in part.index:
                value = part.loc[model, value_col]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                row_values.append(float(value))
            else:
                row_values.append(np.nan)
        values[cost] = row_values
    return values


def add_group_separators(ax) -> None:
    # 4 rule-based, 4 ML, 4 neural.
    for x in [3.5, 7.5]:
        ax.axvline(x, linestyle="--", linewidth=1, alpha=0.35)

    y_min, y_max = ax.get_ylim()
    y_text = y_max - (y_max - y_min) * 0.04
    ax.text(1.5, y_text, "Rule-based", ha="center", va="top", fontsize=8)
    ax.text(5.5, y_text, "ML", ha="center", va="top", fontsize=8)
    ax.text(9.5, y_text, "Neural", ha="center", va="top", fontsize=8)


def plot_grouped_bars(
        ax,
        df: pd.DataFrame,
        value_col: str,
        title: str,
        ylabel: str,
        log_scale: bool = False,
        abs_value: bool = False,
        baseline: float | None = None,
) -> None:
    x = np.arange(len(STRATEGY_ORDER))
    width = 0.22
    values = grouped_values(df, value_col)

    for idx, cost in enumerate(COST_ORDER):
        y = np.array(values[cost], dtype=float)
        if abs_value:
            y = np.abs(y)
        ax.bar(x + (idx - 1) * width, y, width=width, label=COST_LABELS[cost])

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[m] for m in STRATEGY_ORDER], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    add_group_separators(ax)
    ax.legend(fontsize=8)


def build_equity_file_path(row: pd.Series) -> Path:
    model = str(row["model_name"]).lower()
    prefix = FILE_PREFIXES[model]
    threshold_bp = float(row["threshold_bp"])
    max_positions = int(row["max_positions"])
    buy_cost_bp = float(row["buy_cost_bp"])
    sell_cost_bp = float(row["sell_cost_bp"])

    filename = (
        f"{prefix}_test_"
        f"thr_{threshold_bp}_"
        f"pos_{max_positions}_"
        f"buy_{buy_cost_bp}_"
        f"sell_{sell_cost_bp}_equity.csv"
    )
    return RESULTS_DIR / model / filename


def load_equity_curves_for_1bp(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    cost_df = df[df["cost_bp"] == 1.0]

    for _, row in cost_df.iterrows():
        model = str(row["model_name"]).lower()
        path = build_equity_file_path(row)
        if not path.exists():
            print(f"[WARN] Не найден equity-файл: {path}")
            continue

        equity_df = pd.read_csv(path)
        if "begin" in equity_df.columns:
            equity_df["begin"] = pd.to_datetime(equity_df["begin"], errors="coerce")
        if "equity" in equity_df.columns:
            equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce")
        equity_df = equity_df.dropna(subset=["begin", "equity"]).copy()
        if equity_df.empty:
            continue

        equity_df["equity_norm"] = equity_df["equity"] / float(equity_df["equity"].iloc[0])
        equity_df["drawdown_pct"] = (
            equity_df["equity"] / equity_df["equity"].cummax() - 1.0
        ) * 100.0
        curves[model] = equity_df

    return curves


def plot_equity_curves(ax, curves: dict[str, pd.DataFrame]) -> None:
    for model in STRATEGY_ORDER:
        if model not in curves:
            continue
        eq = curves[model]
        ax.plot(
            eq["begin"],
            eq["equity_norm"],
            label=STRATEGY_LABELS[model],
            linewidth=1.1,
        )

    ax.set_title("Кривая капитала, лучшие конфигурации при издержках 0,01%")
    ax.set_ylabel("Капитал / начальный капитал")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=7, ncol=2)


def save_slide_1(df: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

    plot_grouped_bars(
        axes[0, 0],
        df,
        value_col="total_return_pct",
        title="Итоговая доходность на тестовой выборке",
        ylabel="Доходность, %",
    )
    plot_grouped_bars(
        axes[0, 1],
        df,
        value_col="max_drawdown_pct",
        title="Максимальная просадка",
        ylabel="|Просадка|, %",
        abs_value=True,
    )
    plot_grouped_bars(
        axes[1, 0],
        df,
        value_col="profit_factor",
        title="Прибыль / убыток по сделкам",
        ylabel="Суммарная прибыль / суммарный убыток",
        baseline=1.0,
    )
    plot_equity_curves(axes[1, 1], curves)

    out_path = PLOTS_DIR / "slide1_results_with_all_strategies_2x2.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_slide_2(df: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

    plot_grouped_bars(
        axes[0, 0],
        df,
        value_col="trades",
        title="Количество сделок",
        ylabel="Сделки, логарифмическая шкала",
        log_scale=True,
    )
    plot_grouped_bars(
        axes[0, 1],
        df,
        value_col="total_cost_pct",
        title="Суммарные торговые издержки",
        ylabel="Издержки, % от капитала",
    )
    plot_grouped_bars(
        axes[1, 0],
        df,
        value_col="mean_holding_minutes",
        title="Средняя длительность удержания позиции",
        ylabel="Минуты",
    )
    plot_grouped_bars(
        axes[1, 1],
        df,
        value_col="time_in_market_pct",
        title="Доля времени в рынке",
        ylabel="% минут с открытыми позициями",
    )

    out_path = PLOTS_DIR / "slide2_explanation_with_all_strategies_2x2.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    df = read_summary()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    curves = load_equity_curves_for_1bp(df)

    slide1 = save_slide_1(df, curves)
    slide2 = save_slide_2(df, curves)

    print(f"Сохранено: {slide1}")
    print(f"Сохранено: {slide2}")


if __name__ == "__main__":
    main()
