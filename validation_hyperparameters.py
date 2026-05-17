from __future__ import annotations

# backtest_strategy.py содержит главный список BACKTEST_STRATEGIES.
# Здесь остаются только режим запуска и grids торговых параметров.
RUN_VALIDATION_BACKTEST = True
RUN_TEST_BACKTEST = True

COST_BP_PAIRS = [
    (1.0, 1.0),
    (3.0, 3.0),
    (5.0, 5.0),
]

MAX_POSITIONS_LIST = [1, 3, 5, 7, 10]

# Пороги задаются в базисных пунктах: 1 bp = 0,01%.
MODEL_THRESHOLDS_BP = [0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 12.0, 16.0, 24.0, 32.0]
RULE_BASED_THRESHOLDS_BP = {
    "momentum": [0.0, 2.0, 5.0, 10.0, 20.0, 30.0],
    "mean_reversion": [0.0, 2.0, 5.0, 10.0, 20.0, 30.0],
    "ma_trend": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
    "breakout": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
}

MODEL_NAMES = [
    "ridge",
    "catboost",
    "random_forest",
    "decision_tree",
    "gru",
    "lstm",
    "transformer",
    "tcn",
]

RULE_BASED_NAMES = [
    "momentum",
    "mean_reversion",
    "ma_trend",
    "breakout",
]


def build_grid(thresholds_bp: list[float]) -> list[dict[str, float | int]]:
    grid = []
    for threshold_bp in thresholds_bp:
        for max_positions in MAX_POSITIONS_LIST:
            for buy_cost_bp, sell_cost_bp in COST_BP_PAIRS:
                grid.append({
                    "threshold_bp": float(threshold_bp),
                    "max_positions": int(max_positions),
                    "buy_cost_bp": float(buy_cost_bp),
                    "sell_cost_bp": float(sell_cost_bp),
                })
    return grid


VALIDATION_PARAMETER_GRIDS = {}

for name in RULE_BASED_NAMES:
    VALIDATION_PARAMETER_GRIDS[name] = build_grid(RULE_BASED_THRESHOLDS_BP[name])

for name in MODEL_NAMES:
    VALIDATION_PARAMETER_GRIDS[name] = build_grid(MODEL_THRESHOLDS_BP)
