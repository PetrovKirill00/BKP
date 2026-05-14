from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("backtest_results")
SUMMARY_PATH = RESULTS_DIR / "all_models_test_summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

INITIAL_CASH = 1_000_000.0

MODEL_ORDER = ["ridge", "gru", "lstm", "transformer", "arima"]
MODEL_LABELS = {
    "ridge": "Ridge",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "arima": "ARIMA",
}

MODEL_FILE_PREFIX = {
    "ridge": "ridge",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "transformer",
    "arima": "ARIMA",
}

# Для слайда обычно достаточно показать equity/drawdown при минимальных издержках.
EQUITY_CURVE_COST_BP = 1.0
MAX_CURVE_POINTS = 2_500


def normalize_model_name(model_name: str) -> str:
    return str(model_name).lower()


def ensure_output_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def read_test_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл {SUMMARY_PATH}. Сначала запусти backtest_strategy.py."
        )

    df = pd.read_csv(SUMMARY_PATH)
    df["model_name"] = df["model_name"].map(normalize_model_name)

    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()

    df = df[df["model_name"].isin(MODEL_ORDER)].copy()
    df["model_order"] = df["model_name"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df["model_label"] = df["model_name"].map(MODEL_LABELS)

    # В твоём grid buy_cost_bp == sell_cost_bp. Если когда-нибудь они разойдутся,
    # подпись всё равно останется корректной.
    same_cost = np.isclose(df["buy_cost_bp"], df["sell_cost_bp"])
    df["cost_bp"] = np.where(same_cost, df["buy_cost_bp"], np.nan)
    df["cost_label"] = np.where(
        same_cost,
        df["buy_cost_bp"].map(lambda x: f"{x:g} bp"),
        df.apply(lambda r: f"buy {r['buy_cost_bp']:g} / sell {r['sell_cost_bp']:g} bp", axis=1),
    )

    df = df.sort_values(["cost_bp", "model_order"]).reset_index(drop=True)
    return df


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="model_label",
        columns="cost_label",
        values=metric,
        aggfunc="first",
    )

    ordered_model_labels = [MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in pivot.index]
    ordered_cost_labels = [f"{c:g} bp" for c in sorted(df["buy_cost_bp"].dropna().unique())]
    ordered_cost_labels = [c for c in ordered_cost_labels if c in pivot.columns]

    return pivot.loc[ordered_model_labels, ordered_cost_labels]


def plot_grouped_bar(
        ax: plt.Axes,
        df: pd.DataFrame,
        metric: str,
        title: str,
        ylabel: str,
        abs_values: bool = False,
        log_y: bool = False,
        hline: float | None = None,
) -> None:
    pivot = pivot_metric(df, metric)

    if abs_values:
        pivot = pivot.abs()

    x = np.arange(len(pivot.index))
    n_cols = len(pivot.columns)
    width = 0.8 / max(n_cols, 1)

    for i, col in enumerate(pivot.columns):
        offset = (i - (n_cols - 1) / 2) * width
        ax.bar(x + offset, pivot[col].to_numpy(), width=width, label=col)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)

    if log_y:
        ax.set_yscale("log")

    if hline is not None:
        ax.axhline(hline, linewidth=1, linestyle="--")

    ax.legend(fontsize=8)


def plot_return_sensitivity(ax: plt.Axes, df: pd.DataFrame) -> None:
    for model_name in MODEL_ORDER:
        part = df[df["model_name"] == model_name].sort_values("cost_bp")
        if part.empty:
            continue

        ax.plot(
            part["cost_bp"],
            part["total_return_pct"],
            marker="o",
            label=MODEL_LABELS[model_name],
        )

    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_title("Чувствительность доходности к издержкам")
    ax.set_xlabel("Издержки на покупку и продажу, bp")
    ax.set_ylabel("Доходность, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def format_float_for_filename(value: float) -> str:
    # backtest_strategy.py формирует имя через обычный f-string от float:
    # 1.0 -> "1.0", 0.5 -> "0.5", 25.0 -> "25.0".
    return str(float(value))


def find_equity_file(row: pd.Series) -> Path | None:
    model_name = str(row["model_name"])
    file_prefix = MODEL_FILE_PREFIX[model_name]
    model_dir = RESULTS_DIR / model_name

    threshold = format_float_for_filename(row["threshold_bp"])
    max_positions = str(int(row["max_positions"]))
    buy_cost = format_float_for_filename(row["buy_cost_bp"])
    sell_cost = format_float_for_filename(row["sell_cost_bp"])

    expected_name = (
        f"{file_prefix}_test_"
        f"thr_{threshold}_"
        f"pos_{max_positions}_"
        f"buy_{buy_cost}_"
        f"sell_{sell_cost}_equity.csv"
    )
    expected_path = model_dir / expected_name

    if expected_path.exists():
        return expected_path

    # Fallback на случай, если float в имени был записан без .0 или иначе.
    pattern = (
        f"{file_prefix}_test_"
        f"thr_*_"
        f"pos_{max_positions}_"
        f"buy_*_"
        f"sell_*_equity.csv"
    )

    candidates = []
    for path in model_dir.glob(pattern):
        name = path.name
        parsed = re.search(
            r"thr_([^_]+)_pos_([^_]+)_buy_([^_]+)_sell_([^_]+)_equity\.csv$",
            name,
        )
        if parsed is None:
            continue

        try:
            parsed_threshold = float(parsed.group(1))
            parsed_pos = int(parsed.group(2))
            parsed_buy = float(parsed.group(3))
            parsed_sell = float(parsed.group(4))
        except ValueError:
            continue

        if (
                np.isclose(parsed_threshold, float(row["threshold_bp"]))
                and parsed_pos == int(row["max_positions"])
                and np.isclose(parsed_buy, float(row["buy_cost_bp"]))
                and np.isclose(parsed_sell, float(row["sell_cost_bp"]))
        ):
            candidates.append(path)

    if candidates:
        return candidates[0]

    return None


def read_equity_curve(row: pd.Series) -> pd.DataFrame | None:
    path = find_equity_file(row)
    if path is None:
        print(
            "Не нашёл equity-файл для",
            row["model_name"],
            "threshold=", row["threshold_bp"],
            "pos=", row["max_positions"],
            "buy=", row["buy_cost_bp"],
            "sell=", row["sell_cost_bp"],
        )
        return None

    equity = pd.read_csv(path, parse_dates=["begin"])
    if len(equity) > MAX_CURVE_POINTS:
        step = int(np.ceil(len(equity) / MAX_CURVE_POINTS))
        equity = equity.iloc[::step].copy()

    return equity


def plot_equity_curves(ax: plt.Axes, df: pd.DataFrame) -> None:
    selected = df[np.isclose(df["buy_cost_bp"], EQUITY_CURVE_COST_BP)].copy()
    selected = selected.sort_values("model_order")

    for _, row in selected.iterrows():
        equity = read_equity_curve(row)
        if equity is None or equity.empty:
            continue

        ax.plot(
            equity["begin"],
            equity["equity"] / INITIAL_CASH,
            label=MODEL_LABELS[str(row["model_name"])],
        )

    ax.set_title(f"Кривая капитала, {EQUITY_CURVE_COST_BP:g} bp")
    ax.set_ylabel("Капитал / начальный капитал")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)


def plot_drawdown_curves(ax: plt.Axes, df: pd.DataFrame) -> None:
    selected = df[np.isclose(df["buy_cost_bp"], EQUITY_CURVE_COST_BP)].copy()
    selected = selected.sort_values("model_order")

    for _, row in selected.iterrows():
        equity = read_equity_curve(row)
        if equity is None or equity.empty:
            continue

        curve = equity["equity"].astype(float)
        drawdown_pct = (curve / curve.cummax() - 1.0) * 100.0

        ax.plot(
            equity["begin"],
            drawdown_pct,
            label=MODEL_LABELS[str(row["model_name"])],
        )

    ax.set_title(f"Просадка во времени, {EQUITY_CURVE_COST_BP:g} bp")
    ax.set_ylabel("Drawdown, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)


def save_single_plot(filename: str, draw_func) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_func(ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_dashboard(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    plot_grouped_bar(
        axes[0], df,
        metric="total_return_pct",
        title="Доходность на test",
        ylabel="Доходность, %",
    )
    plot_return_sensitivity(axes[1], df)
    plot_grouped_bar(
        axes[2], df,
        metric="max_drawdown_pct",
        title="Максимальная просадка",
        ylabel="|Max drawdown|, %",
        abs_values=True,
    )
    plot_grouped_bar(
        axes[3], df,
        metric="profit_factor",
        title="Profit factor",
        ylabel="Profit factor",
        hline=1.0,
    )
    plot_grouped_bar(
        axes[4], df,
        metric="trades",
        title="Количество сделок",
        ylabel="Сделки, log scale",
        log_y=True,
    )
    plot_grouped_bar(
        axes[5], df,
        metric="turnover",
        title="Оборот стратегии",
        ylabel="Turnover, разы от капитала",
        log_y=True,
    )
    plot_equity_curves(axes[6], df)
    plot_drawdown_curves(axes[7], df)

    fig.suptitle(
        "Сравнение моделей торговой стратегии на тестовой выборке, горизонт 15 минут",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS_DIR / "00_h15_test_dashboard_2x4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_individual_plots(df: pd.DataFrame) -> None:
    save_single_plot(
        "01_total_return_by_model_and_cost.png",
        lambda ax: plot_grouped_bar(
            ax, df, "total_return_pct", "Доходность на test", "Доходность, %"
        ),
    )
    save_single_plot(
        "02_return_sensitivity_to_costs.png",
        lambda ax: plot_return_sensitivity(ax, df),
    )
    save_single_plot(
        "03_max_drawdown_by_model_and_cost.png",
        lambda ax: plot_grouped_bar(
            ax, df, "max_drawdown_pct", "Максимальная просадка", "|Max drawdown|, %", abs_values=True
        ),
    )
    save_single_plot(
        "04_profit_factor_by_model_and_cost.png",
        lambda ax: plot_grouped_bar(
            ax, df, "profit_factor", "Profit factor", "Profit factor", hline=1.0
        ),
    )
    save_single_plot(
        "05_trades_by_model_and_cost.png",
        lambda ax: plot_grouped_bar(
            ax, df, "trades", "Количество сделок", "Сделки, log scale", log_y=True
        ),
    )
    save_single_plot(
        "06_turnover_by_model_and_cost.png",
        lambda ax: plot_grouped_bar(
            ax, df, "turnover", "Оборот стратегии", "Turnover, разы от капитала", log_y=True
        ),
    )
    save_single_plot(
        "07_equity_curves_1bp.png",
        lambda ax: plot_equity_curves(ax, df),
    )
    save_single_plot(
        "08_drawdown_curves_1bp.png",
        lambda ax: plot_drawdown_curves(ax, df),
    )


def main() -> None:
    ensure_output_dir()
    df = read_test_summary()

    build_dashboard(df)
    build_individual_plots(df)

    print(f"Графики сохранены в: {PLOTS_DIR}")
    print(f"Главный файл для слайда: {PLOTS_DIR / '00_h15_test_dashboard_2x4.png'}")


if __name__ == "__main__":
    main()
