import pandas as pd
import numpy as np
from pathlib import Path

from constants import MOEX_CANDLES_DIR, FILTERED_SUMMARY_PATH


DEBUG = True
DEBUG_MAX_SECIDS = None


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

        fill_cols = ["open", "close", "high", "low", "value", "volume"]
        day_df[fill_cols] = day_df[fill_cols].ffill()

        parts.append(day_df)

    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["begin"].dt.date


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


def drop_last_row_of_each_day(df):
    """
    У последней минуты дня нет целевой переменной внутри этого же дня,
    поэтому такие строки удаляем.
    """
    target_cols = [
        "next_open", "next_high", "next_low", "next_close",
        "next_volume", "next_value", "next_log_volume",
        "next_log_value"
    ]
    df.dropna(subset=target_cols, inplace=True)


def process_secid_file(secid: str) -> pd.DataFrame:
    """
    Полный pipeline для одной бумаги.
    """
    if DEBUG:
        print(f"{secid}: начинаю чтение файла")

    df = read_secid_file(secid)

    if DEBUG:
        print(f"{secid}: строю поминутную сетку по дням")
    build_minute_grid(df)

    if DEBUG:
        print(f"{secid}: добавляю временные признаки")
    add_basic_time_features(df)
    add_cyclic_time_features(df)

    if DEBUG:
        print(f"{secid}: добавляю логарифмы и целевые значения")
    add_log_features(df)
    add_next_minute_targets(df)
    drop_last_row_of_each_day(df)

    df["secid"] = secid

    ordered_cols = [
        "secid",
        "begin", "end", "trade_date",
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
        "next_open", "next_high", "next_low", "next_close",
        "next_volume", "next_value",
        "next_log_volume", "next_log_value",
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