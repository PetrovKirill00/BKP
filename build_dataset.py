import pandas as pd
import numpy as np
from pathlib import Path

from constants import MOEX_CANDLES_DIR, FILTERED_SUMMARY_PATH, HORIZON

DEBUG = True
DEBUG_MAX_SECIDS = None

RETURN_LAGS = [1, 2, 3, 5, 10, 15, 30]
CLOSE_MA_WINDOWS = [5, 15, 30]
VOLATILITY_WINDOWS = [5, 15]
VOLUME_MA_WINDOWS = [5]


def read_liquid_secids():
    summary = pd.read_csv(FILTERED_SUMMARY_PATH)
    secids = summary["secid"].tolist()

    if DEBUG_MAX_SECIDS is not None:
        secids = secids[:DEBUG_MAX_SECIDS]

    secids.sort()

    return secids


def read_secid_file(secid):
    path = MOEX_CANDLES_DIR / f"{secid}_TQBR_1m.csv"

    # Считывание
    df = pd.read_csv(path)
    df["begin"] = pd.to_datetime(df["begin"])
    df["end"] = pd.to_datetime(df["end"])

    numeric_cols = ["open", "close", "high", "low", "value", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    if DEBUG:
        # sanity check
        df2 = df.sort_values("begin").drop_duplicates(subset=["begin"], keep="last")
        if not df.equals(df2):
            raise Exception("Данные почему-то не отсортированы по времени")

    return df


def build_minute_grid(df):
    df["trade_date"] = df["begin"].dt.date

    parts = []
    for date, day_df in df.groupby("trade_date", sort=True):
        if DEBUG:
            # sanity check
            df2 = day_df.sort_values("begin")
            if not day_df.equals(df2):
                raise Exception("Данные внутри одного дня почему-то не отсортированы по времени")

        start = day_df["begin"].iloc[0]
        finish = day_df["begin"].iloc[-1]

        full_index = pd.date_range(
            start=start,
            end=finish,
            freq="1min"
        )

        day_df = day_df.set_index("begin")
        day_df = day_df.reindex(full_index)

        day_df.index.name = "begin"
        day_df = day_df.reset_index()

        day_df["is_imputed"] = day_df["open"].isna().astype(int)

        fill_cols = ["open", "close", "high", "low", "value", "volume"]
        day_df[fill_cols] = day_df[fill_cols].ffill()

        parts.append(day_df)

    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["begin"].dt.date

    return df


def restore_end(df):
    df["end"] = df["begin"] + pd.Timedelta(minutes=1)


def add_return_lag_features(df):
    grouped_close = df.groupby("trade_date")["close"]

    for lag in RETURN_LAGS:
        col = f"return_{lag}"
        # pct_change возвращает дробное изменение. Т.е. y[t] = (x[t] / x[t-lag]) - 1
        # В начале дня недостаточно истории для расчёта лаговой доходности,
        # поэтому пропуски заполняются нулём.
        df[col] = grouped_close.pct_change(periods=lag).fillna(0.0)


def add_volume_features(df):
    grouped = df.groupby("trade_date", group_keys=False)

    df["log_volume_diff_1"] = grouped["log_volume"].diff().fillna(0.0)
    df["log_value_diff_1"] = grouped["log_value"].diff().fillna(0.0)

    for window in VOLUME_MA_WINDOWS:
        df[f"log_volume_ma_{window}"] = (
            grouped["log_volume"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"log_value_ma_{window}"] = (
            grouped["log_value"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0,drop=True)
        )

def add_rolling_features(df):
    grouped = df.groupby("trade_date", group_keys=False)

    for window in CLOSE_MA_WINDOWS:
        ma_col = f"ma_close_{window}"
        df[ma_col] = (
            grouped["close"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"close_to_ma_{window}"] = (
            df["close"] / df[ma_col] - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

def add_volatility_features(df):
    grouped = df.groupby("trade_date", group_keys=False)

    for window in VOLATILITY_WINDOWS:
        col = f"volatility_{window}"
        df[col] = (
            grouped["return_1"]
            .rolling(window=window, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    df["hl_range"] = (
        (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    ).fillna(0.0)


def add_candle_shape_features(df):
    base = df["open"].replace(0, np.nan)
    upper_base = np.maximum(df["open"], df["close"])
    lower_base = np.minimum(df["open"], df["close"])
    df["body"] = ((df["close"] - df["open"]) / base).fillna(0.0)
    df["upper_wick"] = ((df["high"] - upper_base) / base).fillna(0.0)
    df["lower_wick"] = ((lower_base - df["low"]) / base).fillna(0.0)


def add_basic_time_features(df):
    df["hour"] = df["begin"].dt.hour
    df["minute"] = df["begin"].dt.minute

    df["minute_from_midnight"] = df["hour"] * 60 + df["minute"]

    # 0 = понедельник, 6 = воскресенье
    df["day_of_week"] = df["begin"].dt.dayofweek

    df["day_of_month"] = df["begin"].dt.day
    df["month"] = df["begin"].dt.month
    df["days_in_month"] = df["begin"].dt.days_in_month

    df["is_month_start"] = df["begin"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["begin"].dt.is_month_end.astype(int)


def add_cyclic_time_features(df):
    # Час внутри суток
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Минута внутри часа
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    # Положение внутри суток
    df["minute_from_midnight_sin"] = np.sin(
        2 * np.pi * df["minute_from_midnight"] / (24 * 60)
    )
    df["minute_from_midnight_cos"] = np.cos(
        2 * np.pi * df["minute_from_midnight"] / (24 * 60)
    )

    # День недели
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Месяц
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # День внутри месяца
    df["day_of_month_sin"] = np.sin(
        2 * np.pi * (df["day_of_month"] - 1) / df["days_in_month"]
    )
    df["day_of_month_cos"] = np.cos(
        2 * np.pi * (df["day_of_month"] - 1) / df["days_in_month"]
    )


def add_log_features(df):
    df["log_volume"] = np.log1p(df["volume"])
    df["log_value"] = np.log1p(df["value"])


def add_next_minute_targets(df):
    target_cols = ["open", "high", "low", "close", "volume", "value"]
    for col in target_cols:
        df[f"next_{col}"] = df.groupby("trade_date")[col].shift(-1)

    df["next_log_volume"] = np.log1p(df["next_volume"])
    df["next_log_value"] = np.log1p(df["next_value"])

def add_target_return(df, horizon=HORIZON):
    future_close = df.groupby("trade_date")["close"].shift(-horizon)
    df[f"target_return_{horizon}"] = future_close / df["close"] - 1.0


def drop_rows_without_target(df, horizon=HORIZON):
    target_col = f"target_return_{horizon}"
    df.dropna(subset=[target_col], inplace=True)


def process_secid_file(secid: str) -> pd.DataFrame:
    if DEBUG:
        print(f"{secid}: начинаю чтение файла")

    df = read_secid_file(secid)

    if DEBUG:
        print(f"{secid}: строю поминутную сетку по дням")
    df = build_minute_grid(df)      # тут нужен return, потому что внутри собирается новый df
    restore_end(df)            # если restore_end тоже возвращает новый df

    if DEBUG:
        print(f"{secid}: добавляю временные признаки")
    add_basic_time_features(df)     # in-place
    add_cyclic_time_features(df)    # in-place

    if DEBUG:
        print(f"{secid}: добавляю лаги доходности и объёмные признаки")
    add_log_features(df)            # in-place
    add_return_lag_features(df)     # in-place
    add_volume_features(df)         # in-place

    if DEBUG:
        print(f"{secid}: добавляю скользящие, волатильность и геометрию свечи")
    add_rolling_features(df)        # in-place
    add_volatility_features(df)     # in-place
    add_candle_shape_features(df)   # in-place

    if DEBUG:
        print(f"{secid}: добавляю целевые значения")
    add_next_minute_targets(df)     # in-place
    add_target_return(df, HORIZON)  # in-place
    drop_rows_without_target(df)   # in-place

    df["secid"] = secid

    ordered_cols = [
        "secid",
        "begin", "end", "trade_date", "is_imputed",

        "hour", "minute", "minute_from_midnight",
        "day_of_week", "day_of_month", "month", "days_in_month",
        "is_month_start", "is_month_end",

        "hour_sin", "hour_cos",
        "minute_sin", "minute_cos",
        "minute_from_midnight_sin", "minute_from_midnight_cos",
        "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos",
        "day_of_month_sin", "day_of_month_cos",

        "open", "high", "low", "close", "volume", "value",
        "log_volume", "log_value",

        "return_1", "return_2", "return_3", "return_5",
        "return_10", "return_15", "return_30",

        "log_volume_diff_1", "log_value_diff_1",
        "log_volume_ma_5", "log_value_ma_5",

        "ma_close_5", "ma_close_15", "ma_close_30",
        "close_to_ma_5", "close_to_ma_15", "close_to_ma_30",

        "volatility_5", "volatility_15", "hl_range",

        "body", "upper_wick", "lower_wick",

        "next_open", "next_high", "next_low", "next_close",
        "next_volume", "next_value",
        "next_log_volume", "next_log_value",

        f"target_return_{HORIZON}",
    ]

    existing_cols = [c for c in ordered_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    return df[existing_cols + other_cols]


def build_dataset():
    secids = read_liquid_secids()
    if DEBUG:
        print("Считались secid, их", len(secids))

    parts = []

    for i, secid in enumerate(secids, start=1):
        if DEBUG:
            print(f"[{i}/{len(secids)}] Обрабатываю {secid}")

        data = process_secid_file(secid)
        parts.append(data)

    if DEBUG:
        print("Собираю данные вместе")
    ds = pd.concat(parts, ignore_index=True)
    ds = ds.sort_values(["secid", "begin"]).reset_index(drop=True)

    return ds

def main():
    ds = build_dataset()

    print("\nИтоговый размер датасета:")
    print(ds.shape)
    print("\nКолонки:")
    print(ds.columns.tolist())
    print("\nПервые строки:")
    print(ds.head())

    output_parquet_path = Path("minute_dataset.parquet")

    print(f"\nСохраняю в {output_parquet_path}")
    ds.to_parquet(output_parquet_path, index=False)

    output_csv_path = Path("minute_dataset.csv")
    if DEBUG:
        ds.head(10000).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено в {output_csv_path}")


if __name__ == "__main__":
    main()