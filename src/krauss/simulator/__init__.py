"""Public simulator API.

Frontend code (e.g. ``app/pages/7_Simulator.py``) should import from here::

    from krauss.simulator import run_backtest, run_simulation, BacktestResult

and not reach into ``krauss.backtest.*`` or ``krauss.data.*`` directly.
"""

from krauss.simulator.api import (  # noqa: F401
    FAMILIES,
    SCHEMES,
    BacktestResult,
    run_backtest,
    run_simulation,
    valid_scheme_for_era,
)
