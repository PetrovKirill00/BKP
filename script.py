from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path("data/sequence_shards/horizon_15_window_30/train")

HARD_SHARDS = [31, 32, 7, 20]

for shard_id in HARD_SHARDS:
    path = BASE_DIR / f"shard_{shard_id:05d}.npz"

    with np.load(path, allow_pickle=False) as data:
        secids = data["group_secids"].astype(str)
        dates = pd.to_datetime(data["group_trade_dates"].astype(str))

    print("=" * 80)
    print(path.name)
    print("Общий диапазон дат:", dates.min().date(), "->", dates.max().date())
    print("Количество торговых дней-групп:", len(dates))
    print("Бумаги:", sorted(set(secids)))

    print("\nДиапазоны по бумагам:")
    df = pd.DataFrame({
        "secid": secids,
        "trade_date": dates,
    })

    ranges = (
        df.groupby("secid")["trade_date"]
        .agg(["min", "max", "count"])
        .reset_index()
    )

    for _, row in ranges.iterrows():
        print(
            row["secid"],
            row["min"].date(),
            "->",
            row["max"].date(),
            f"days={row['count']}",
        )