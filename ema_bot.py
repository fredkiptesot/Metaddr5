"""
EMA crossover + ATR stop forex bot skeleton (OANDA, practice account).

SETUP
    pip install oandapyV20 pandas numpy
    export OANDA_TOKEN="your-practice-api-token"      # never hardcode keys
    export OANDA_ACCOUNT_ID="101-xxx-xxxxxxx-xxx"
    python ema_bot.py

KILL SWITCH
    Create a file named STOP in the same folder and the bot exits cleanly.

LIMITATIONS (read before extending)
    - Position sizing assumes an XXX/USD pair (e.g. EUR_USD) on a USD account.
      Other pairs/account currencies need a pip-value conversion.
    - This is a learning skeleton, not a profitable strategy. Backtest it first.
"""

import os
import time
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
log = logging.getLogger("bot")


# ----------------------------------------------------------------- config
@dataclass
class Config:
    instrument: str = "EUR_USD"
    granularity: str = "H1"
    fast: int = 20
    slow: int = 50
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    risk_per_trade: float = 0.005     # 0.5% of balance per trade
    max_daily_loss: float = 0.02      # halt for the day at -2% NAV
    max_units: int = 50_000           # hard cap on position size
    poll_seconds: int = 30
    candles: int = 200


# ------------------------------------------------------------- data feed
class DataFeed:
    def __init__(self, client: API, cfg: Config):
        self.client, self.cfg = client, cfg

    def get_candles(self) -> pd.DataFrame:
        r = instruments.InstrumentsCandles(
            instrument=self.cfg.instrument,
            params={"granularity": self.cfg.granularity,
                    "count": self.cfg.candles, "price": "M"},
        )
        self.client.request(r)
        rows = [
            {
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
            }
            for c in r.response["candles"] if c["complete"]  # closed bars only
        ]
        return pd.DataFrame(rows)


# ------------------------------------------------------------- strategy
class Strategy:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.cfg.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.cfg.slow, adjust=False).mean()
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / self.cfg.atr_period, adjust=False).mean()
        return df

    def signal(self, df: pd.DataFrame) -> int:
        """+1 = bullish cross, -1 = bearish cross, 0 = nothing."""
        df = self.add_indicators(df)
        prev, last = df.iloc[-2], df.iloc[-1]
        if prev.ema_fast <= prev.ema_slow and last.ema_fast > last.ema_slow:
            return 1
        if prev.ema_fast >= prev.ema_slow and last.ema_fast < last.ema_slow:
            return -1
        return 0


# --------------------------------------------------------- risk manager
class RiskManager:
    def __init__(self, cfg: Config, start_nav: float):
        self.cfg = cfg
        self.day_start_nav = start_nav
        self.day = time.strftime("%Y-%m-%d")

    def roll_day(self, nav: float):
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.day_start_nav = today, nav

    def trading_allowed(self, nav: float) -> bool:
        self.roll_day(nav)
        loss = (self.day_start_nav - nav) / self.day_start_nav
        if loss >= self.cfg.max_daily_loss:
            log.warning("Daily loss limit hit (%.2f%%). No new trades today.", loss * 100)
            return False
        return True

    def position_size(self, balance: float, stop_distance: float) -> int:
        """Units such that hitting the stop loses ~risk_per_trade of balance."""
        if stop_distance <= 0:
            return 0
        units = (balance * self.cfg.risk_per_trade) / stop_distance
        return int(min(units, self.cfg.max_units))


# ------------------------------------------------------------- executor
class Executor:
    def __init__(self, client: API, account_id: str, cfg: Config):
        self.client, self.account_id, self.cfg = client, account_id, cfg

    def account(self) -> dict:
        r = accounts.AccountSummary(self.account_id)
        self.client.request(r)
        a = r.response["account"]
        return {"balance": float(a["balance"]), "nav": float(a["NAV"])}

    def open_side(self):
        """Return 'long', 'short' or None for the configured instrument."""
        r = positions.OpenPositions(self.account_id)
        self.client.request(r)
        for p in r.response["positions"]:
            if p["instrument"] == self.cfg.instrument:
                if float(p["long"]["units"]) > 0:
                    return "long"
                if float(p["short"]["units"]) < 0:
                    return "short"
        return None

    def market_order(self, units: int, stop_price: float):
        data = {"order": {
            "type": "MARKET",
            "instrument": self.cfg.instrument,
            "units": str(units),  # negative = sell
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{stop_price:.5f}"},
        }}
        r = orders.OrderCreate(self.account_id, data=data)
        self.client.request(r)
        log.info("Order sent: units=%s stop=%.5f", units, stop_price)

    def close_all(self, side: str):
        data = {"longUnits": "ALL"} if side == "long" else {"shortUnits": "ALL"}
        r = positions.PositionClose(self.account_id, self.cfg.instrument, data=data)
        self.client.request(r)
        log.info("Closed %s position", side)


# ----------------------------------------------------------------- main
def main():
    token = os.environ["OANDA_TOKEN"]
    account_id = os.environ["OANDA_ACCOUNT_ID"]
    cfg = Config()

    client = API(access_token=token, environment="practice")  # practice only!
    feed = DataFeed(client, cfg)
    strategy = Strategy(cfg)
    executor = Executor(client, account_id, cfg)
    risk = RiskManager(cfg, executor.account()["nav"])

    last_bar = None
    log.info("Bot started on %s %s", cfg.instrument, cfg.granularity)

    while True:
        if os.path.exists("STOP"):
            log.info("STOP file found. Exiting.")
            break
        try:
            df = feed.get_candles()
            bar_time = df.iloc[-1]["time"]

            if bar_time != last_bar:          # act once per new closed candle
                last_bar = bar_time
                sig = strategy.signal(df)
                acct = executor.account()
                side = executor.open_side()
                log.info("Bar %s | signal=%s | position=%s | NAV=%.2f",
                         bar_time, sig, side, acct["nav"])

                # exit on opposite cross
                if (side == "long" and sig == -1) or (side == "short" and sig == 1):
                    executor.close_all(side)
                    side = None

                # entry
                if side is None and sig != 0 and risk.trading_allowed(acct["nav"]):
                    ind = strategy.add_indicators(df).iloc[-1]
                    stop_dist = ind.atr * cfg.atr_stop_mult
                    units = risk.position_size(acct["balance"], stop_dist)
                    if units > 0:
                        price = ind.close
                        if sig == 1:
                            executor.market_order(units, price - stop_dist)
                        else:
                            executor.market_order(-units, price + stop_dist)

        except V20Error as e:
            log.error("OANDA API error: %s", e)
        except Exception:
            log.exception("Unexpected error; continuing after pause")

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
