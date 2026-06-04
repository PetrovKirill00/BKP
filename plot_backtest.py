from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


PRIMARY_RESULTS_DIR = Path("backtest_results")
FALLBACK_RESULTS_DIR = Path(".")
SUMMARY_CANDIDATES = [
    PRIMARY_RESULTS_DIR / "all_strategies_test_summary.csv",
    FALLBACK_RESULTS_DIR / "all_strategies_test_summary.csv",
    PRIMARY_RESULTS_DIR / "all_models_test_summary.csv",
    FALLBACK_RESULTS_DIR / "all_models_test_summary.csv",
]

INITIAL_CASH = 1_000_000.0
TARGET_COST_BP = 3.0  # 0,03%
TOP_PER_CATEGORY = 2

# Формат 16:9 — удобно вставлять на слайды.
FIGSIZE = (13.333, 7.5)
DPI = 200

RESULTS_DIR = PRIMARY_RESULTS_DIR
PLOTS_DIR = PRIMARY_RESULTS_DIR / "plots_single"

RULE_BASED_STRATEGIES = ["momentum", "mean_reversion", "ma_trend", "breakout"]
ML_STRATEGIES = ["ridge", "catboost", "random_forest", "decision_tree"]
NEURAL_STRATEGIES = ["gru", "lstm", "transformer", "tcn"]

CATEGORY_TO_MODELS = {
    "Стратегии на основе правил": RULE_BASED_STRATEGIES,
    "Методы машинного обучения": ML_STRATEGIES,
    "Нейросетевые модели": NEURAL_STRATEGIES,
}
CATEGORY_ORDER = list(CATEGORY_TO_MODELS.keys())

STRATEGY_LABELS = {
    "momentum": "Моментум",
    "mean_reversion": "Возврат к среднему",
    "ma_trend": "Тренд по скользящим средним",
    "breakout": "Пробой уровня",
    "ridge": "Гребневая регрессия",
    "catboost": "Градиентный бустинг",
    "random_forest": "Случайный лес",
    "decision_tree": "Дерево решений",
    "gru": "Рекуррентная модель GRU",
    "lstm": "Рекуррентная модель LSTM",
    "transformer": "Трансформер",
    "tcn": "Временная свёрточная сеть",
}

CATEGORY_COLORS = {
    "Стратегии на основе правил": "#4e79a7",
    "Методы машинного обучения": "#f28e2b",
    "Нейросетевые модели": "#59a14f",
}


def read_summary() -> pd.DataFrame:
    global RESULTS_DIR, PLOTS_DIR

    summary_path = None
    for candidate in SUMMARY_CANDIDATES:
        if candidate.exists():
            summary_path = candidate
            break

    if summary_path is None:
        raise FileNotFoundError(
            "Не найден файл all_strategies_test_summary.csv. "
            "Сначала запусти backtest_strategy.py."
        )

    RESULTS_DIR = summary_path.parent
    PLOTS_DIR = RESULTS_DIR / "plots_single"

    df = pd.read_csv(summary_path)

    numeric_cols = [
        "buy_cost_bp",
        "sell_cost_bp",
        "threshold_bp",
        "max_positions",
        "total_return_pct",
        "max_drawdown_pct",
        "return_to_drawdown",
        "profit_factor",
        "trades",
        "turnover",
        "total_cost_pct",
        "final_equity",
        "win_rate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "model_name" not in df.columns:
        raise ValueError("В summary отсутствует колонка model_name.")

    df["model_name"] = df["model_name"].astype(str).str.lower()

    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "test"].copy()

    df = df[df["buy_cost_bp"] == df["sell_cost_bp"]].copy()
    df = df[df["buy_cost_bp"] == TARGET_COST_BP].copy()

    allowed_models = set(RULE_BASED_STRATEGIES + ML_STRATEGIES + NEURAL_STRATEGIES)
    df = df[df["model_name"].isin(allowed_models)].copy()

    if df.empty:
        raise ValueError(
            "После фильтрации не осталось строк для test и издержек 0,03%."
        )

    # Оставляем для каждой стратегии одну лучшую конфигурацию.
    sort_cols = []
    ascending = []

    if "final_equity" in df.columns:
        sort_cols.append("final_equity")
        ascending.append(False)
    elif "total_return_pct" in df.columns:
        sort_cols.append("total_return_pct")
        ascending.append(False)
    else:
        raise ValueError(
            "В summary нужна хотя бы одна из колонок: final_equity или total_return_pct."
        )

    if "return_to_drawdown" in df.columns:
        sort_cols.append("return_to_drawdown")
        ascending.append(False)

    if "profit_factor" in df.columns:
        sort_cols.append("profit_factor")
        ascending.append(False)

    if "trades" in df.columns:
        sort_cols.append("trades")
        ascending.append(False)

    df = df.sort_values(["model_name"] + sort_cols, ascending=[True] + ascending)
    df = df.drop_duplicates(subset=["model_name"], keep="first").reset_index(drop=True)

    return df



def select_top_strategies_by_category(df: pd.DataFrame) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []

    for category in CATEGORY_ORDER:
        models = CATEGORY_TO_MODELS[category]
        part = df[df["model_name"].isin(models)].copy()
        if part.empty:
            continue

        if "final_equity" in part.columns:
            part["selection_score"] = part["final_equity"] / INITIAL_CASH
        else:
            part["selection_score"] = 1.0 + part["total_return_pct"] / 100.0

        sort_cols = ["selection_score"]
        ascending = [False]

        if "return_to_drawdown" in part.columns:
            sort_cols.append("return_to_drawdown")
            ascending.append(False)

        if "profit_factor" in part.columns:
            sort_cols.append("profit_factor")
            ascending.append(False)

        part = part.sort_values(sort_cols, ascending=ascending)
        part = part.head(TOP_PER_CATEGORY).copy()
        part["category"] = category
        selected_parts.append(part)

    if not selected_parts:
        raise ValueError("Не удалось выбрать стратегии для построения графиков.")

    result = pd.concat(selected_parts, ignore_index=True)
    result["strategy_label"] = result["model_name"].map(STRATEGY_LABELS)
    result["color"] = result["category"].map(CATEGORY_COLORS)

    if "final_equity" in result.columns:
        result["equity_factor"] = result["final_equity"] / INITIAL_CASH
    else:
        result["equity_factor"] = 1.0 + result["total_return_pct"] / 100.0

    result["max_drawdown_abs_pct"] = result["max_drawdown_pct"].abs()

    if "return_to_drawdown" not in result.columns:
        total_return = result["equity_factor"] - 1.0
        drawdown_frac = result["max_drawdown_abs_pct"] / 100.0
        result["return_to_drawdown"] = np.where(
            drawdown_frac > 0,
            total_return / drawdown_frac,
            np.nan,
        )

    if "profit_factor" not in result.columns:
        result["profit_factor"] = np.nan

    if "trades" not in result.columns:
        result["trades"] = np.nan

    if "win_rate" in result.columns:
        result["win_rate_pct"] = result["win_rate"] * 100.0
    else:
        result["win_rate_pct"] = np.nan

    # Порядок: сначала категории, внутри — по отношению конечного капитала к начальному.
    ordered_rows = []
    for category in CATEGORY_ORDER:
        part = result[result["category"] == category].copy()
        if part.empty:
            continue
        part = part.sort_values("equity_factor", ascending=False)
        ordered_rows.append(part)

    result = pd.concat(ordered_rows, ignore_index=True)
    return result



def format_thousands(value: float, _pos: int) -> str:
    if not np.isfinite(value):
        return ""
    return f"{int(round(value)):,}".replace(",", " ")



def add_value_labels(ax, values: list[float], fmt) -> None:
    if not values:
        return

    finite_values = [v for v in values if np.isfinite(v)]
    if not finite_values:
        return

    max_value = max(finite_values)
    offset = max_value * 0.015 if max_value != 0 else 0.02

    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        ax.text(value + offset, idx, fmt(value), va="center", fontsize=11)



def plot_horizontal_bar(
    df: pd.DataFrame,
    value_col: str,
    xlabel: str,
    filename: str,
    formatter,
    baseline: float | None = None,
    note: str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    labels = df["strategy_label"].tolist()
    values = df[value_col].astype(float).tolist()
    colors = df["color"].tolist()
    y = np.arange(len(df))

    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=13)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    if baseline is not None:
        ax.axvline(baseline, linestyle="--", linewidth=1.5, color="black", alpha=0.7)

    add_value_labels(ax, values, formatter)

    if value_col == "trades":
        ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))

    if note:
        ax.text(
            0.99,
            0.02,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="dimgray",
        )

    out_path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path



def plot_risk_return_scatter(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = df["max_drawdown_abs_pct"].astype(float).to_numpy()
    y = df["equity_factor"].astype(float).to_numpy()
    trades = df["trades"].fillna(0).astype(float).to_numpy()
    colors = df["color"].tolist()
    labels = df["strategy_label"].tolist()

    # Размер маркера зависит от числа сделок.
    if np.nanmax(trades) > 0:
        sizes = 200 + 1800 * (trades / np.nanmax(trades))
    else:
        sizes = np.full_like(trades, 400.0)

    ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.85,
    )

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(
            label,
            (xi, yi),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=11,
        )

    ax.set_xlabel("Максимальная просадка, %", fontsize=13)
    ax.set_ylabel("Конечный капитал / начальный капитал", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black", alpha=0.7)

    ax.text(
        0.99,
        0.02,
        "Размер точки пропорционален числу сделок",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="dimgray",
    )

    out_path = PLOTS_DIR / "06_risk_return_scatter_0_03pct.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path





def add_subplot_value_labels(ax, values: list[float], fmt) -> None:
    if not values:
        return

    finite_values = [v for v in values if np.isfinite(v)]
    if not finite_values:
        return

    max_value = max(finite_values)
    offset = max_value * 0.02 if max_value != 0 else 0.02

    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        ax.text(value + offset, idx, fmt(value), va="center", fontsize=8)



def plot_metric_on_axis(
    ax,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str,
    formatter,
    baseline: float | None = None,
    note: str | None = None,
) -> None:
    labels = df["strategy_label"].tolist()
    values = df[value_col].astype(float).tolist()
    colors = df["color"].tolist()
    y = np.arange(len(df))

    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    if baseline is not None:
        ax.axvline(baseline, linestyle="--", linewidth=1.1, color="black", alpha=0.7)

    if value_col == "trades":
        ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))

    add_subplot_value_labels(ax, values, formatter)

    if note:
        ax.text(
            0.98,
            0.03,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="dimgray",
        )



def plot_effectiveness_summary_2x2(df: pd.DataFrame) -> Path:
    """
    Строит дополнительный сводный график 2×2.

    Старые отдельные графики остаются без заголовков и без изменений.
    Здесь заголовки специально добавлены, потому что это один общий рисунок
    для отдельного слайда.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

    plot_metric_on_axis(
        ax=axes[0, 0],
        df=df,
        value_col="equity_factor",
        title="Отношение конечного капитала к начальному",
        xlabel="Капитал / начальный капитал",
        formatter=lambda v: f"{v:.3f}",
        baseline=1.0,
    )

    plot_metric_on_axis(
        ax=axes[0, 1],
        df=df,
        value_col="max_drawdown_abs_pct",
        title="Максимальная просадка",
        xlabel="Просадка, %",
        formatter=lambda v: f"{v:.2f}%",
        note="Меньше — лучше",
    )

    plot_metric_on_axis(
        ax=axes[1, 0],
        df=df,
        value_col="trades",
        title="Количество сделок",
        xlabel="Количество закрытых сделок",
        formatter=lambda v: f"{int(round(v)):,}".replace(",", " "),
    )

    plot_metric_on_axis(
        ax=axes[1, 1],
        df=df,
        value_col="return_to_drawdown",
        title="Доходность на единицу просадки",
        xlabel="Доходность / максимальная просадка",
        formatter=lambda v: f"{v:.3f}",
        note="Больше — лучше",
    )

    fig.suptitle("Сравнение стратегий при издержках 0,03%", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = PLOTS_DIR / "07_effectiveness_summary_2x2_0_03pct.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_pdf(image_paths: list[Path]) -> Path:
    pdf_path = PLOTS_DIR / "single_plots_0_03pct_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for image_path in image_paths:
            image = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    return pdf_path



def main() -> None:
    df = read_summary()
    selected = select_top_strategies_by_category(df)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Сохраняем компактную таблицу, чтобы можно было проверить значения.
    save_columns = [
        "category",
        "model_name",
        "strategy_label",
        "threshold_bp",
        "max_positions",
        "buy_cost_bp",
        "sell_cost_bp",
        "equity_factor",
        "total_return_pct",
        "max_drawdown_pct",
        "max_drawdown_abs_pct",
        "return_to_drawdown",
        "profit_factor",
        "trades",
        "turnover",
        "total_cost_pct",
        "win_rate_pct",
    ]
    save_columns = [col for col in save_columns if col in selected.columns]
    selected_to_save = selected[save_columns].copy()
    selected_csv_path = PLOTS_DIR / "selected_strategies_0_03pct.csv"
    selected_to_save.to_csv(selected_csv_path, index=False, encoding="utf-8-sig")

    image_paths: list[Path] = []

    image_paths.append(
        plot_horizontal_bar(
            df=selected,
            value_col="equity_factor",
            xlabel="Конечный капитал / начальный капитал",
            filename="01_final_equity_factor_0_03pct.png",
            formatter=lambda v: f"{v:.3f}",
            baseline=1.0,
        )
    )

    image_paths.append(
        plot_horizontal_bar(
            df=selected,
            value_col="trades",
            xlabel="Количество закрытых сделок",
            filename="02_trades_0_03pct.png",
            formatter=lambda v: f"{int(round(v)):,}".replace(",", " "),
        )
    )

    image_paths.append(
        plot_horizontal_bar(
            df=selected,
            value_col="max_drawdown_abs_pct",
            xlabel="Максимальная просадка, %",
            filename="03_max_drawdown_0_03pct.png",
            formatter=lambda v: f"{v:.2f}%",
            note="Меньше — лучше",
        )
    )

    image_paths.append(
        plot_horizontal_bar(
            df=selected,
            value_col="return_to_drawdown",
            xlabel="Доходность на единицу просадки",
            filename="04_return_to_drawdown_0_03pct.png",
            formatter=lambda v: f"{v:.3f}",
            note="Больше — лучше",
        )
    )

    image_paths.append(
        plot_horizontal_bar(
            df=selected,
            value_col="profit_factor",
            xlabel="Фактор прибыли",
            filename="05_profit_factor_0_03pct.png",
            formatter=lambda v: f"{v:.3f}",
            baseline=1.0,
            note="Больше 1 — прибыльная стратегия",
        )
    )

    image_paths.append(plot_risk_return_scatter(selected))

    # Новый дополнительный график 2×2 с заголовками.
    # Старые отдельные графики выше остаются без заголовков и без изменений.
    image_paths.append(plot_effectiveness_summary_2x2(selected))

    pdf_path = save_pdf(image_paths)

    print("Выбраны стратегии:")
    for _, row in selected.iterrows():
        trades_value = int(row["trades"]) if pd.notna(row["trades"]) else "nan"
        print(
            f"  {row['category']}: {row['strategy_label']} | "
            f"capital_ratio={row['equity_factor']:.4f}, "
            f"MDD={row['max_drawdown_abs_pct']:.4f}%, "
            f"trades={trades_value}"
        )

    print(f"Сохранена таблица: {selected_csv_path}")
    for image_path in image_paths:
        print(f"Сохранён график: {image_path}")
    print(f"Сохранён PDF-отчёт: {pdf_path}")


if __name__ == "__main__":
    main()
