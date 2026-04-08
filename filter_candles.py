from constants import MOEX_CANDLES_DIR, SUMMARY_PATH, FILTERED_SUMMARY_PATH
import pandas as pd

def build_candles_summary(candles_dir):
    records = []

    for path in sorted(candles_dir.glob("*_TQBR_1m.csv")):
        secid = path.name.split("_")[0]
        print(f"Обрабатываю {secid}")

        df = pd.read_csv(path)

        df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
        df = df.dropna(subset=["begin"])

        last_timestamp = pd.Timestamp("2026-03-30 23:59:59")
        df = df[df["begin"] <= last_timestamp]

        df["date"] = df["begin"].dt.date
        df["month"] = df["begin"].dt.to_period('M')

        total_rows = len(df)
        active_days = df["date"].nunique()
        active_months = df["month"].nunique()

        records.append({
            "secid": secid,
            "total_rows": total_rows,
            "first_date": df["begin"].min().date(),
            "last_date": df["begin"].max().date(),
            "active_days": active_days,
            "active_months": active_months,
            "avg_rows_per_active_day": total_rows / active_days
        })
    summary = pd.DataFrame(records)
    summary = summary.sort_values(
        by=["avg_rows_per_active_day", "total_rows"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return summary



def main():
    if not SUMMARY_PATH.exists():
        print("Строю summary по скачанным бумагам.")
        summary = build_candles_summary(MOEX_CANDLES_DIR)
        summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    else:
        print("Summary уже построен.")
        summary = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")

    summary["first_date"] = pd.to_datetime(summary["first_date"])

    filtered = summary[
        (summary["first_date"] <= pd.Timestamp("2022-07-01")) &
        (summary["active_months"] >= 45) &
        (summary["avg_rows_per_active_day"] >= 500)
    ].copy()

    filtered = filtered.sort_values(
        by=["avg_rows_per_active_day", "total_rows"],
        ascending=[False, False]
    ).reset_index(drop=True)

    filtered.to_csv(FILTERED_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("Топ-20 отфильтрованных компаний:")
    print(filtered.head(20))
    print()
    print(f"Всего ценных бумаг: {len(filtered)}")


if __name__ == "__main__":
    main()