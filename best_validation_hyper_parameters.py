from __future__ import annotations

# Этот файл будет автоматически перезаписан backtest_strategy.py после validation.
# Пока он просто импортирует полный grid из validation_hyperparameters.py.

from validation_hyperparameters import (  # noqa: F401
    RUN_TEST_BACKTEST,
    RUN_VALIDATION_BACKTEST,
    VALIDATION_PARAMETER_GRIDS,
)

BEST_VALIDATION_CONFIGS_BY_MODEL = VALIDATION_PARAMETER_GRIDS
