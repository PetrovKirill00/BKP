import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from constants import HORIZON, TRAIN_UNTIL, VALIDATE_UNTIL

DATASET_PATH = "minute_dataset.parquet"
TARGET_COL = f"target_return_{HORIZON}"

BACKTEST_THRESHOLDS_BP = [1.5, 2.0, 2.5, 3.0]
TRADE_COST_BP = 2.0

def read_dataset():
    return pd.read_parquet(DATASET_PATH)

def get_feature_columns(df):
    exclude = {
        "secid", "begin", "end", "trade_date",
        "open", "high", "low", "close", "volume", "value",
        "next_open", "next_high", "next_low", "next_close",
        "next_volume", "next_value",
        "next_log_volume", "next_log_value",
        "ma_close_5", "ma_close_15", "ma_close_30",
        TARGET_COL,
    }
    return [col for col in df.columns if col not in exclude]

def split_by_time(df):
    train = df[df["begin"] < TRAIN_UNTIL]
    valid = df[(df["begin"] >= TRAIN_UNTIL) & (df["begin"] < VALIDATE_UNTIL)]
    test = df[df["begin"] >= VALIDATE_UNTIL]
    return train, valid, test

def safe_correlation(y_true, y_pred):
    true_std = np.std(y_true)
    pred_std = np.std(y_pred)

    if true_std == 0 or pred_std == 0:
        return np.nan

    return np.corrcoef(y_true, y_pred)[0, 1]

def print_regression_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    corr = safe_correlation(y_true, y_pred)

    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    direction_accuracy = np.mean(direction_true == direction_pred)

    print(f"\n{name}")
    print(f"MAE:                 {mae:.8f}")
    print(f"RMSE:                {rmse:.8f}")
    print(f"R2:                  {r2:.8f}")
    if np.isnan(corr):
        print("Correlation:         undefined")
    else:
        print(f"Correlation:         {corr:.8f}")
    print(f"Direction accuracy:  {direction_accuracy:.4f}")
    print(f"Pred mean:           {np.mean(y_pred):.8f}")
    print(f"Pred std:            {np.std(y_pred):.8f}")
    print(f"True mean:           {np.mean(y_true):.8f}")
    print(f"True std:            {np.std(y_true):.8f}")



def train_naive_model(valid):
    y_valid = valid[TARGET_COL].to_numpy()
    pred_valid = np.zeros_like(y_valid)

    print_regression_metrics("Наивный прогноз на valid", y_valid, pred_valid)


def analyze_prediction_buckets(y_true, y_pred, n_bins=10):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })

    df["bucket"] = pd.qcut(df["y_pred"], q=n_bins, duplicates="drop")

    result = df.groupby("bucket", observed=True).agg(
        count=("y_true", "size"),
        mean_pred=("y_pred", "mean"),
        mean_true=("y_true", "mean"),
        median_true=("y_true", "median"),
        positive_share=("y_true", lambda x: np.mean(x > 0)),
    )

    result["mean_pred_bp"] = result["mean_pred"] * 10_000
    result["mean_true_bp"] = result["mean_true"] * 10_000
    result["median_true_bp"] = result["median_true"] * 10_000

    result = result[
        [
            "count",
            "mean_pred",
            "mean_true",
            "median_true",
            "positive_share",
            "mean_pred_bp",
            "mean_true_bp",
            "median_true_bp",
        ]
    ]

    print("\nАнализ по группам прогноза:")

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.float_format", "{:.8f}".format,
    ):
        print(result.to_string())

def analyze_long_thresholds(y_true, y_pred, thresholds_bp, cost_bp=2.0):
    rows = []

    for threshold_bp in thresholds_bp:
        threshold = threshold_bp / 10_000
        mask = y_pred >= threshold

        if mask.sum() == 0:
            rows.append({
                "threshold_bp": threshold_bp,
                "count": 0,
                "mean_true_bp": np.nan,
                "median_true_bp": np.nan,
                "positive_share": np.nan,
            })
            continue

        selected_true = y_true[mask]

        rows.append({
            "threshold_bp": threshold_bp,
            "count": mask.sum(),
            "mean_true_bp": selected_true.mean() * 10_000,
            "median_true_bp": np.median(selected_true) * 10_000,
            "positive_share": np.mean(selected_true > 0),
            "mean_net_bp": selected_true.mean() * 10_000 - cost_bp,
            "median_net_bp": np.median(selected_true) * 10_000 - cost_bp,
        })

    result = pd.DataFrame(rows)

    print("\nАнализ long-сигналов по порогам прогноза:")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.6f}".format,
    ):
        print(result.to_string(index=False))


def train_ridge_regression(train, valid, feature_cols, alpha=1.0):
    x_train = train[feature_cols].to_numpy()
    y_train = train[TARGET_COL].to_numpy()

    x_valid = valid[feature_cols].to_numpy()
    y_valid = valid[TARGET_COL].to_numpy()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)

    model = Ridge(alpha=alpha)
    model.fit(x_train_scaled, y_train)

    pred_valid = model.predict(x_valid_scaled)

    print_regression_metrics(f"Ridge alpha={alpha} на valid", y_valid, pred_valid)
    analyze_prediction_buckets(y_valid, pred_valid)
    analyze_long_thresholds(
        y_valid,
        pred_valid,
        thresholds_bp=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    )

    return model, scaler, pred_valid


def evaluate_model_on_split(name, model, scaler, data, feature_cols, thresholds_bp):
    x = data[feature_cols].to_numpy()
    y = data[TARGET_COL].to_numpy()

    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)

    print_regression_metrics(name, y, pred)
    analyze_prediction_buckets(y, pred)
    analyze_long_thresholds(y, pred, thresholds_bp, cost_bp=2.0)

    return pred

def run_simple_long_backtest(name, data, y_pred, thresholds_bp, cost_bp=2.0):
    """
    Простейший long-only backtest.

    Если прогноз >= threshold, считаем, что открывается long-сделка
    на один шаг прогноза. Доходность сделки равна target_return - cost.

    Упрощения:
    - Каждая строка с сигналом считается отдельной сделкой;
    - если в одну минуту есть несколько сигналов по разным бумагам,
      их доходности усредняются
    - капитал, лимиты позиций и состояние портфеля пока не моделируются
    """
    bt = data[["begin", "secid", TARGET_COL]].copy()
    bt["forecast"] = y_pred

    rows = []

    for threshold_bp in thresholds_bp:
        threshold = threshold_bp / 10_000
        cost = cost_bp / 10_000

        trades = bt[bt["forecast"] >= threshold].copy()

        if trades.empty:
            rows.append({
                "threshold_bp": threshold_bp,
                "cost_bp": cost_bp,
                "signals": 0,
                "active_minutes": 0,
                "avg_signals_per_minute": np.nan,
                "mean_gross_bp": np.nan,
                "mean_net_bp": np.nan,
                "win_rate_net": np.nan,
                "total_return_pct": np.nan,
                "max_drawdown_pct": np.nan,
            })
            continue

        trades["gross_return"] = trades[TARGET_COL]
        trades["net_return"] = trades["gross_return"] - cost

        # Если в одну минуту несколько бумаг дали сигнал,
        # считаем равновзвешенную доходность по этим сигналам
        minute_returns = (
            trades
            .groupby("begin")["net_return"]
            .mean()
            .sort_index()
        )

        equity = (1.0 + minute_returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0

        signals_per_minute = trades.groupby("begin").size()

        rows.append({
            "threshold_bp": threshold_bp,
            "cost_bp": cost_bp,
            "signals": len(trades),
            "active_minutes": len(minute_returns),
            "avg_signals_per_minute": signals_per_minute.mean(),
            "mean_gross_bp": trades["gross_return"].mean() * 10_000,
            "mean_net_bp": trades["net_return"].mean() * 10_000,
            "win_rate_net": np.mean(trades["net_return"] > 0),
            "total_return_pct": (equity.iloc[-1] - 1.0) * 100,
            "max_drawdown_pct": drawdown.min() * 100,
        })

    result = pd.DataFrame(rows)

    print(f"\nПростой backtest: {name}")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.float_format", "{:.6f}".format,
    ):
        print(result.to_string(index=False))

def run_topk_long_backtest(
        name,
        data,
        y_pred,
        thresholds_bp,
        max_positions_list,
        cost_bp=2.0
):
    """
    Упрощенный портфельный long-only backtest.

    На каждой минуте:
    - берём бумаги с прогнозом выше порога;
    - выбираем top-K по прогнозу;
    - считаем равновзвешенную доходность выбранных бумаг;
    - из каждой сделки вычитаем cost_bp.

    Упрощения:
    - позиция удерживается ровно один шаг прогноза;
    - позиции не переносятся на следующую минуту;
    - капитал распределяется поровну между выбранными бумагами;
    - ликвидность стакана и реальное исполнение пока не моделируются.
    """

    bt = data[["begin", "secid", TARGET_COL]].copy()
    bt["forecast"] = y_pred

    all_minutes = pd.Index(sorted(data["begin"].unique()))
    rows = []

    for threshold_bp in thresholds_bp:
        threshold = threshold_bp / 10_000
        cost = cost_bp / 10_000

        candidates = bt[bt["forecast"] >= threshold].copy()

        for max_positions in max_positions_list:
            if candidates.empty:
                rows.append({
                    "threshold_bp": threshold_bp,
                    "max_positions": max_positions,
                    "cost_bp": cost_bp,
                    "signals": 0,
                    "active_minutes": 0,
                    "mean_net_bp": np.nan,
                    "median_net_bp": np.nan,
                    "win_rate_minute": np.nan,
                    "total_return_pct": np.nan,
                    "log_total_return": np.nan,
                    "max_drawdown_pct": np.nan,
                    "avg_positions_per_active_minute": np.nan,
                })
                continue

            selected = (
                candidates
                .sort_values(["begin", "forecast"], ascending=[True, False])
                .groupby("begin")
                .head(max_positions)
                .copy()
            )

            selected["net_return"] = selected[TARGET_COL] - cost

            minute_returns = (
                selected
                .groupby("begin")["net_return"]
                .mean()
                .reindex(all_minutes, fill_value=0.0)
                .sort_index()
            )

            active_minute_returns = minute_returns[minute_returns != 0.0]

            equity = (1.0 + minute_returns).cumprod()
            drawdown = equity / equity.cummax() - 1.0

            positions_per_minute = selected.groupby("begin").size()

            rows.append({
                "threshold_bp": threshold_bp,
                "max_positions": max_positions,
                "cost_bp": cost_bp,
                "signals": len(selected),
                "active_minutes": len(active_minute_returns),
                "mean_net_bp": active_minute_returns.mean() * 10_000,
                "median_net_bp": active_minute_returns.median() * 10_000,
                "win_rate_minute": np.mean(active_minute_returns > 0),
                "total_return_pct": (equity.iloc[-1] - 1.0) * 100,
                "log_total_return": np.log(equity.iloc[-1]),
                "max_drawdown_pct": drawdown.min() * 100,
                "avg_positions_per_active_minute": positions_per_minute.mean(),
            })
    result = pd.DataFrame(rows)

    print(f"\nTop-K портфельный backtest: {name}")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 260,
        "display.float_format", "{:.6f}".format,
    ):
        print(result.to_string(index=False))


def main():
    print("Загружаю датасет")
    df = read_dataset()

    print("Размер датасета:", df.shape)
    print("Целевая колонка:", TARGET_COL)

    df = df.dropna(subset=[TARGET_COL]).copy()

    feature_cols = get_feature_columns(df)

    print("Количество признаков:", len(feature_cols))
    print("Первые признаки:", feature_cols[:10])

    train, valid, test = split_by_time(df)

    print("Train:", train.shape)
    print("Valid:", valid.shape)
    print("Test: ", test.shape)

    train_naive_model(valid)
    model, scaler, pred_valid = train_ridge_regression(
        train,
        valid,
        feature_cols,
        alpha=1.0
    )

    run_simple_long_backtest(
        name="valid",
        data=valid,
        y_pred=pred_valid,
        thresholds_bp=BACKTEST_THRESHOLDS_BP,
        cost_bp=TRADE_COST_BP,
    )

    pred_test = evaluate_model_on_split(
        "Ridge alpha=1.0 на test",
        model,
        scaler,
        test,
        feature_cols,
        thresholds_bp=BACKTEST_THRESHOLDS_BP,
    )

    run_simple_long_backtest(
        name="test",
        data=test,
        y_pred=pred_test,
        thresholds_bp=BACKTEST_THRESHOLDS_BP,
        cost_bp=TRADE_COST_BP,
    )
    run_topk_long_backtest(
        name="valid",
        data=valid,
        y_pred=pred_valid,
        thresholds_bp=[2.0, 2.5, 3.0],
        max_positions_list=[1, 3, 5],
        cost_bp=TRADE_COST_BP,
    )

    run_topk_long_backtest(
        name="test",
        data=test,
        y_pred=pred_test,
        thresholds_bp=[2.0, 2.5, 3.0],
        max_positions_list=[1, 3, 5],
        cost_bp=TRADE_COST_BP,
    )


if __name__ == "__main__":
    main()
