from datetime import datetime as dt
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).parent.parent.parent / "build" / "cpp"))

import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np

import heston


def get_time_to_expiry(date: str, today):
    return (dt.strptime(date, "%Y-%m-%d").date() - today).days / 365.25


class MarketData:
    def __init__(self, ticker: str):
        self.today = dt.today().date()
        self.ticker = yf.Ticker(ticker)
        expiries = self.ticker.options  # Option expiries available

        self.expiries = []
        for expiry in expiries:
            if 30 / 365.25 <= get_time_to_expiry(expiry, self.today) <= 2:
                # Only using expiries from 1 month to 2 year for calibration
                self.expiries.append(expiry)

        # Spot price of underlying: S0 in Heston model
        self.spot = self.ticker.history(period="1d")["Close"].iloc[-1]

        irx = yf.Ticker("^IRX")
        # US 13 Week Treasury Bill: r in Heston model
        self.interest_rate = irx.history(period="1d")["Close"].iloc[-1] / 100

    def get_calls_one_date(self, date: str):
        option_chain = self.ticker.option_chain(date)

        calls = option_chain.calls[["strike", "lastPrice", "bid", "ask"]]
        calls = calls[ (0.8 * self.spot <= calls["strike"]) &
                       (calls["strike"] <= 1.2 * self.spot) ] # Filtering out deep OTM and ITM options

        calls["marketPrice"] = np.where(
            (calls["bid"] > 0) & (calls["ask"] > 0),
            (calls["bid"] + calls["ask"])/2,
            calls["lastPrice"]
        )  # Set last price as the market price in case of illiquid options

        calls["timeToExpiry"] = get_time_to_expiry(date, self.today)

        # Cleaning data
        calls = calls[calls["marketPrice"] > 1]               # Filtering cheap deep OTM options for SPX
        calls = calls.dropna(subset=["marketPrice"])
        calls.reset_index(drop=True, inplace=True)

        if len(calls) > 15:
            indices = np.linspace(0, len(calls) - 1, 15, dtype=int)
            calls = calls.iloc[indices]

        return calls[["strike", "marketPrice", "timeToExpiry"]]

    def get_implied_vol(self, row):
        def objective(imp_vol):
            return (heston.black_scholes_price(self.spot, row.strike, row.timeToExpiry, self.interest_rate, imp_vol) -
                    row.marketPrice)

        try:
            return brentq(f=objective, a=1e-6, b=5)
        except ValueError:
            return np.nan

    def get_calls(self):
        # Using multiple threads to do API calls
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.get_calls_one_date, self.expiries))

        calls = pd.concat(results, ignore_index=True)

        calls["impliedVol"] = [self.get_implied_vol(row) for row in calls.itertuples(index=False)]

        calls.dropna(subset=["impliedVol"], inplace=True)
        calls.reset_index(drop=True, inplace=True)
        return calls[["strike", "timeToExpiry", "impliedVol"]]

    def vega(self, K, T, r, sigma):
        d1 = (np.log(self.spot / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return self.spot * norm.pdf(d1) * np.sqrt(T)
