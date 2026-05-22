from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

INITIAL_CASH = 1_000_000.0
SELECTION_COST_BP = 3.0
TOP_PER_CATEGORY = 2

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

RULE_BASED_STRATEGIES = ["momentum", "mean_reversion", "ma_trend", "breakout"]
ML_STRATEGIES = ["ridge", "catboost", "random_forest", "decision_tree"]
NEURAL_STRATEGIES = ["gru", "lstm", "transformer", "tcn"]

CATEGORY_TO_MODELS = {
    "Правиловые стратегии": RULE_BASED_STRATEGIES,
    "Машинное обучение": ML_STRATEGIES,
    "Нейросети": NEURAL_STRATEGIES,
}
CATEGORY_ORDER = ["Правиловые стратегии", "Машинное обучение", "Нейросети"]

COST_ORDER = [1.0, 3.0, 5.0]
COST_LABELS = {1.0: "0,01%", 3.0: "0,03%", 5.0: "0,05%"}
COST_TITLES = {1.0: "1 б.п.", 3.0: "3 б.п.", 5.0: "5 б.п."}

# Более контрастная палитра.
CONTRAST_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
]

ACTIVE_STRATEGY_ORDER = list(STRATEGY_ORDER)
ACTIVE_COLORS_BY_MODEL: dict[str, str] = {}
ACTIVE_CATEGORY_COUNTS: list[int] = [4, 4, 4]
ACTIVE_CATEGORY_SELECTIONS: dict[str, list[str]] = {
    category: list(models)
    for category, models in CATEGORY_TO_MODELS.items()
}


# ============================================================
# Чтение summary
# ============================================================


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
        "final_equity",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = [
        "model_name",
        "split",
        "buy_cost_bp",
        "sell_cost_bp",
        "threshold_bp",
        "max_positions",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"В summary нет обязательных колонок: {missing_cols}")

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

    df["cost_bp"] = df["buy_cost_bp"].astype(float)

    if "time_in_market_share" in df.columns:
        df["time_in_market_pct"] = df["time_in_market_share"] * 100.0

    sort_cols = ["model_name", "cost_bp"]
    if "total_return_pct" in df.columns:
        df = df.sort_values(
            sort_cols + ["total_return_pct"],
            ascending=[True, True, False],
        )
    df = df.drop_duplicates(sort_cols, keep="first")

    df["model_name"] = pd.Categorical(
        df["model_name"],
        STRATEGY_ORDER,
        ordered=True,
    )
    df["cost_bp"] = pd.Categorical(df["cost_bp"], COST_ORDER, ordered=True)
    df = df.sort_values(["model_name", "cost_bp"]).reset_index(drop=True)

    return df


# ============================================================
# Выбор топ-2 стратегий из каждой категории
# ============================================================


def _metric_for_selection(part: pd.DataFrame) -> pd.Series:
    if "total_return_pct" in part.columns and part["total_return_pct"].notna().any():
        return part["total_return_pct"]

    if "final_equity" in part.columns and part["final_equity"].notna().any():
        return part["final_equity"] / INITIAL_CASH - 1.0

    raise ValueError(
        "Для отбора лучших стратегий нужна колонка total_return_pct "
        "или final_equity."
    )


def select_top_strategies_by_category(
        df: pd.DataFrame,
        selection_cost_bp: float = SELECTION_COST_BP,
        top_per_category: int = TOP_PER_CATEGORY,
) -> pd.DataFrame:
    global ACTIVE_STRATEGY_ORDER, ACTIVE_COLORS_BY_MODEL
    global ACTIVE_CATEGORY_COUNTS, ACTIVE_CATEGORY_SELECTIONS

    selected_order: list[str] = []
    category_counts: list[int] = []
    category_selections: dict[str, list[str]] = {}

    selection_df = df[df["cost_bp"] == selection_cost_bp].copy()
    if selection_df.empty:
        raise ValueError(
            f"В summary нет строк для издержек {selection_cost_bp} bp, "
            "по которым нужно отобрать лучшие стратегии."
        )

    for category in CATEGORY_ORDER:
        category_models = CATEGORY_TO_MODELS[category]
        part = selection_df[selection_df["model_name"].isin(category_models)].copy()

        if part.empty:
            raise ValueError(
                f"В summary нет строк для категории {category} "
                f"при издержках {selection_cost_bp} bp."
            )

        part["selection_metric"] = _metric_for_selection(part)
        part = part.sort_values(
            by=["selection_metric", "profit_factor", "trades"],
            ascending=[False, False, False],
        )

        selected = part["model_name"].astype(str).head(top_per_category).tolist()
        if len(selected) < top_per_category:
            raise ValueError(
                f"Для категории {category} найдено только {len(selected)} "
                f"стратегий вместо {top_per_category}."
            )

        category_selections[category] = selected
        selected_order.extend(selected)
        category_counts.append(len(selected))

    ACTIVE_STRATEGY_ORDER = selected_order
    ACTIVE_CATEGORY_COUNTS = category_counts
    ACTIVE_CATEGORY_SELECTIONS = category_selections
    ACTIVE_COLORS_BY_MODEL = {
        model: CONTRAST_COLORS[idx % len(CONTRAST_COLORS)]
        for idx, model in enumerate(ACTIVE_STRATEGY_ORDER)
    }

    filtered_df = df[df["model_name"].astype(str).isin(ACTIVE_STRATEGY_ORDER)].copy()
    filtered_df["model_name"] = pd.Categorical(
        filtered_df["model_name"].astype(str),
        ACTIVE_STRATEGY_ORDER,
        ordered=True,
    )
    filtered_df = filtered_df.sort_values(["model_name", "cost_bp"]).reset_index(drop=True)

    return filtered_df


# ============================================================
# Пути к подробным файлам backtest
# ============================================================


def _format_float_for_filename(value: float | int) -> str:
    return str(float(value))


def build_details_file_path(row: pd.Series, suffix: str) -> Path:
    model = str(row["model_name"]).lower()
    prefix = FILE_PREFIXES[model]
    threshold_bp = _format_float_for_filename(row["threshold_bp"])
    max_positions = int(row["max_positions"])
    buy_cost_bp = _format_float_for_filename(row["buy_cost_bp"])
    sell_cost_bp = _format_float_for_filename(row["sell_cost_bp"])

    filename = (
        f"{prefix}_test_"
        f"thr_{threshold_bp}_"
        f"pos_{max_positions}_"
        f"buy_{buy_cost_bp}_"
        f"sell_{sell_cost_bp}_{suffix}.csv"
    )
    return RESULTS_DIR / model / filename


def build_equity_file_path(row: pd.Series) -> Path:
    return build_details_file_path(row=row, suffix="equity")


def build_trades_file_path(row: pd.Series) -> Path:
    return build_details_file_path(row=row, suffix="trades")


# ============================================================
# Загрузка time-series для графиков
# ============================================================


def load_equity_curves_by_cost(
        df: pd.DataFrame,
) -> dict[float, dict[str, pd.DataFrame]]:
    curves: dict[float, dict[str, pd.DataFrame]] = {cost: {} for cost in COST_ORDER}

    for _, row in df.iterrows():
        model = str(row["model_name"]).lower()
        cost = float(row["cost_bp"])
        path = build_equity_file_path(row)

        if not path.exists():
            print(f"[WARN] Не найден equity-файл: {path}")
            continue

        equity_df = pd.read_csv(path)
        if "begin" not in equity_df.columns or "equity" not in equity_df.columns:
            print(f"[WARN] В equity-файле нет begin/equity: {path}")
            continue

        equity_df["begin"] = pd.to_datetime(equity_df["begin"], errors="coerce")
        equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce")
        equity_df = equity_df.dropna(subset=["begin", "equity"]).copy()
        equity_df = equity_df.sort_values("begin").reset_index(drop=True)

        if equity_df.empty:
            print(f"[WARN] Пустой equity-файл после очистки: {path}")
            continue

        equity_df["equity_factor"] = equity_df["equity"] / INITIAL_CASH
        curves[cost][model] = equity_df[["begin", "equity", "equity_factor"]]

    return curves


def _make_zero_trade_curve(
        equity_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if equity_df is None or equity_df.empty:
        return pd.DataFrame(columns=["begin", "trade_count"])

    start_time = equity_df["begin"].iloc[0]
    end_time = equity_df["begin"].iloc[-1]
    return pd.DataFrame({
        "begin": [start_time, end_time],
        "trade_count": [0, 0],
    })


def load_trade_count_curves_by_cost(
        df: pd.DataFrame,
        equity_curves: dict[float, dict[str, pd.DataFrame]],
) -> dict[float, dict[str, pd.DataFrame]]:
    curves: dict[float, dict[str, pd.DataFrame]] = {cost: {} for cost in COST_ORDER}

    for _, row in df.iterrows():
        model = str(row["model_name"]).lower()
        cost = float(row["cost_bp"])
        path = build_trades_file_path(row)
        equity_df = equity_curves.get(cost, {}).get(model)

        if not path.exists():
            print(f"[WARN] Не найден trades-файл: {path}")
            curves[cost][model] = _make_zero_trade_curve(equity_df)
            continue

        trades_df = pd.read_csv(path)

        if trades_df.empty:
            curves[cost][model] = _make_zero_trade_curve(equity_df)
            continue

        time_col = "exit_time" if "exit_time" in trades_df.columns else None
        if time_col is None and "time" in trades_df.columns:
            time_col = "time"

        if time_col is None:
            print(f"[WARN] В trades-файле нет exit_time/time: {path}")
            curves[cost][model] = _make_zero_trade_curve(equity_df)
            continue

        times = pd.to_datetime(trades_df[time_col], errors="coerce").dropna()

        if times.empty:
            curves[cost][model] = _make_zero_trade_curve(equity_df)
            continue

        counts = (
            times.value_counts()
            .sort_index()
            .cumsum()
            .rename("trade_count")
            .reset_index()
        )
        counts.columns = ["begin", "trade_count"]

        if equity_df is not None and not equity_df.empty:
            start_time = equity_df["begin"].iloc[0]
            end_time = equity_df["begin"].iloc[-1]
            start_row = pd.DataFrame({"begin": [start_time], "trade_count": [0]})
            counts = pd.concat([start_row, counts], ignore_index=True)

            last_count = int(counts["trade_count"].iloc[-1])
            if counts["begin"].iloc[-1] < end_time:
                end_row = pd.DataFrame({
                    "begin": [end_time],
                    "trade_count": [last_count],
                })
                counts = pd.concat([counts, end_row], ignore_index=True)

        counts = counts.sort_values("begin").reset_index(drop=True)
        curves[cost][model] = counts

    return curves


# ============================================================
# Вспомогательные bar charts
# ============================================================


def get_strategy_metric_values(
        df: pd.DataFrame,
        cost_bp: float,
        value_col: str,
        fallback: float = np.nan,
) -> list[float]:
    part = df[df["cost_bp"] == cost_bp].set_index("model_name")
    values = []

    for model in ACTIVE_STRATEGY_ORDER:
        if model not in part.index or value_col not in part.columns:
            values.append(fallback)
            continue

        value = part.loc[model, value_col]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        values.append(float(value))

    return values


def get_final_equity_factors_for_cost(df: pd.DataFrame, cost_bp: float) -> list[float]:
    if "total_return_pct" in df.columns:
        returns = get_strategy_metric_values(df, cost_bp, "total_return_pct")
        return [1.0 + value / 100.0 if np.isfinite(value) else np.nan for value in returns]

    if "final_equity" in df.columns:
        final_equity = get_strategy_metric_values(df, cost_bp, "final_equity")
        return [value / INITIAL_CASH if np.isfinite(value) else np.nan for value in final_equity]

    raise ValueError(
        "Для графика итогового фактора доходности нужна колонка "
        "total_return_pct или final_equity."
    )


def add_group_separators(ax) -> None:
    cumulative = np.cumsum(ACTIVE_CATEGORY_COUNTS)

    for boundary in cumulative[:-1]:
        ax.axvline(boundary - 0.5, linestyle="--", linewidth=1, alpha=0.35, color="#7aa6d8")

    y_min, y_max = ax.get_ylim()
    y_text = y_max - (y_max - y_min) * 0.04

    start = 0
    for category, count in zip(CATEGORY_ORDER, ACTIVE_CATEGORY_COUNTS):
        center = start + (count - 1) / 2
        ax.text(center, y_text, category, ha="center", va="top", fontsize=9)
        start += count


def plot_single_cost_bar(
        ax,
        values: list[float],
        title: str,
        ylabel: str,
        baseline: float | None = None,
        log_scale: bool = False,
) -> None:
    x = np.arange(len(ACTIVE_STRATEGY_ORDER))
    colors = [ACTIVE_COLORS_BY_MODEL[model] for model in ACTIVE_STRATEGY_ORDER]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [STRATEGY_LABELS[m] for m in ACTIVE_STRATEGY_ORDER],
        rotation=25,
        ha="right",
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1, color="#1f77b4")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    add_group_separators(ax)


# ============================================================
# Графики time-series
# ============================================================


def plot_equity_curves_for_cost(
        ax,
        curves_by_model: dict[str, pd.DataFrame],
        cost_bp: float,
) -> None:
    for model in ACTIVE_STRATEGY_ORDER:
        if model not in curves_by_model:
            continue

        eq = curves_by_model[model]
        ax.plot(
            eq["begin"],
            eq["equity_factor"],
            label=STRATEGY_LABELS[model],
            linewidth=2.0,
            color=ACTIVE_COLORS_BY_MODEL[model],
        )

    ax.set_title(f"Капитал / начальный капитал, издержки {COST_TITLES[cost_bp]}")
    ax.set_xlabel("Время")
    ax.set_ylabel("Капитал / начальный капитал")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8, ncol=2)



def plot_trade_count_curves_for_cost(
        ax,
        curves_by_model: dict[str, pd.DataFrame],
        cost_bp: float,
) -> None:
    for model in ACTIVE_STRATEGY_ORDER:
        if model not in curves_by_model:
            continue

        trades = curves_by_model[model]
        if trades.empty:
            continue

        ax.plot(
            trades["begin"],
            trades["trade_count"],
            label=STRATEGY_LABELS[model],
            linewidth=2.0,
            color=ACTIVE_COLORS_BY_MODEL[model],
        )

    ax.set_title(f"Накопленное количество сделок, издержки {COST_TITLES[cost_bp]}")
    ax.set_xlabel("Время")
    ax.set_ylabel("Количество закрытых сделок")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8, ncol=2)


# ============================================================
# Сохранение страниц
# ============================================================


def save_page_1(
        df: pd.DataFrame,
        equity_curves: dict[float, dict[str, pd.DataFrame]],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

    plot_equity_curves_for_cost(axes[0, 0], equity_curves[1.0], 1.0)
    plot_equity_curves_for_cost(axes[0, 1], equity_curves[3.0], 3.0)
    plot_equity_curves_for_cost(axes[1, 0], equity_curves[5.0], 5.0)

    final_factors_3bp = get_final_equity_factors_for_cost(df, cost_bp=3.0)
    plot_single_cost_bar(
        axes[1, 1],
        values=final_factors_3bp,
        title="Итоговый фактор доходности при издержках 3 б.п.",
        ylabel="Финальный капитал / начальный капитал",
        baseline=1.0,
    )

    fig.suptitle(
        "Топ-2 стратегии из каждой категории по результату издержках 0.03%",
        fontsize=14,
    )

    out_path = PLOTS_DIR / "page1_equity_and_final_factor.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path



def save_page_2(
        df: pd.DataFrame,
        trade_curves: dict[float, dict[str, pd.DataFrame]],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

    plot_trade_count_curves_for_cost(axes[0, 0], trade_curves[1.0], 1.0)
    plot_trade_count_curves_for_cost(axes[0, 1], trade_curves[3.0], 3.0)
    plot_trade_count_curves_for_cost(axes[1, 0], trade_curves[5.0], 5.0)

    trades_3bp = get_strategy_metric_values(df, cost_bp=3.0, value_col="trades")
    plot_single_cost_bar(
        axes[1, 1],
        values=trades_3bp,
        title="Суммарное количество сделок при издержках 3 б.п.",
        ylabel="Количество закрытых сделок",
        log_scale=True,
    )

    fig.suptitle(
        "Топ-2 стратегии из каждой категории по результату при издержках 0.03%",
        fontsize=14,
    )

    out_path = PLOTS_DIR / "page2_trade_counts.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path



def save_pdf_report(page_1: Path, page_2: Path) -> Path:
    pdf_path = PLOTS_DIR / "backtest_test_pages.pdf"

    with PdfPages(pdf_path) as pdf:
        for page_path in [page_1, page_2]:
            image = plt.imread(page_path)
            fig, ax = plt.subplots(figsize=(18, 10))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    return pdf_path


# ============================================================
# main
# ============================================================


def print_selection_summary() -> None:
    print("Выбраны стратегии для графиков:")
    for category in CATEGORY_ORDER:
        selected = ACTIVE_CATEGORY_SELECTIONS.get(category, [])
        labels = [STRATEGY_LABELS[name] for name in selected]
        print(f"  {category}: {', '.join(labels)}")



def main() -> None:
    df = read_summary()
    df = select_top_strategies_by_category(
        df=df,
        selection_cost_bp=SELECTION_COST_BP,
        top_per_category=TOP_PER_CATEGORY,
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print_selection_summary()

    equity_curves = load_equity_curves_by_cost(df)
    trade_curves = load_trade_count_curves_by_cost(
        df=df,
        equity_curves=equity_curves,
    )

    page_1 = save_page_1(df=df, equity_curves=equity_curves)
    page_2 = save_page_2(df=df, trade_curves=trade_curves)
    pdf_report = save_pdf_report(page_1=page_1, page_2=page_2)

    print(f"Сохранено: {page_1}")
    print(f"Сохранено: {page_2}")
    print(f"Сохранено: {pdf_report}")


if __name__ == "__main__":
    main()
