from __future__ import annotations

# Этот файл автоматически создан backtest_strategy.py по результатам validation.
# Им можно заменить validation_hyperparameters.py, чтобы прогнать только
# лучшие найденные параметры и затем выполнить test backtest.

# После подбора на validation обычно нужен финальный test-прогон.
RUN_TEST_BACKTEST = True
RUN_VALIDATION_BACKTEST = False

TO_BACKTEST = ['ridge', 'gru', 'lstm', 'transformer', 'arima']

VALIDATION_PARAMETER_GRIDS = {'ridge': [{'threshold_bp': 0.5,
            'max_positions': 7,
            'buy_cost_bp': 1.0,
            'sell_cost_bp': 1.0},
           {'threshold_bp': 25.0,
            'max_positions': 1,
            'buy_cost_bp': 3.0,
            'sell_cost_bp': 3.0},
           {'threshold_bp': 25.0,
            'max_positions': 1,
            'buy_cost_bp': 5.0,
            'sell_cost_bp': 5.0}],
 'gru': [{'threshold_bp': 6.0,
          'max_positions': 8,
          'buy_cost_bp': 1.0,
          'sell_cost_bp': 1.0},
         {'threshold_bp': 24.0,
          'max_positions': 2,
          'buy_cost_bp': 3.0,
          'sell_cost_bp': 3.0},
         {'threshold_bp': 28.0,
          'max_positions': 1,
          'buy_cost_bp': 5.0,
          'sell_cost_bp': 5.0}],
 'lstm': [{'threshold_bp': 0.5,
           'max_positions': 6,
           'buy_cost_bp': 1.0,
           'sell_cost_bp': 1.0},
          {'threshold_bp': 12.0,
           'max_positions': 2,
           'buy_cost_bp': 3.0,
           'sell_cost_bp': 3.0},
          {'threshold_bp': 30.0,
           'max_positions': 4,
           'buy_cost_bp': 5.0,
           'sell_cost_bp': 5.0}],
 'transformer': [{'threshold_bp': 0.0,
                  'max_positions': 11,
                  'buy_cost_bp': 1.0,
                  'sell_cost_bp': 1.0},
                 {'threshold_bp': 16.0,
                  'max_positions': 5,
                  'buy_cost_bp': 3.0,
                  'sell_cost_bp': 3.0},
                 {'threshold_bp': 38.0,
                  'max_positions': 1,
                  'buy_cost_bp': 5.0,
                  'sell_cost_bp': 5.0}],
 'arima': [{'threshold_bp': 8.5,
            'max_positions': 1,
            'buy_cost_bp': 1.0,
            'sell_cost_bp': 1.0},
           {'threshold_bp': 25.0,
            'max_positions': 1,
            'buy_cost_bp': 3.0,
            'sell_cost_bp': 3.0},
           {'threshold_bp': 25.0,
            'max_positions': 1,
            'buy_cost_bp': 5.0,
            'sell_cost_bp': 5.0}]}

BEST_VALIDATION_CONFIGS_BY_MODEL = VALIDATION_PARAMETER_GRIDS
