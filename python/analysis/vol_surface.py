import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "build" / "cpp"))
sys.path.append(str(Path(__file__).parent.parent)) # Append python directory

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from calibration.calibrate import calibrate_params, implied_vol_from_price, ETA, N, ALPHA
from data.market_data import MarketData
import heston

market_data = MarketData("^SPX")
calls_data = market_data.get_calls()

strikes = calls_data["strike"].values
maturities = calls_data["timeToExpiry"].values
unique_maturities = calls_data["timeToExpiry"].unique()

params = calibrate_params(market_data, calls_data) # Calibrated params (v0, rho, kappa, theta , xi)
print(params)

heston_prices = np.empty(len(calls_data))
for T in unique_maturities:
    indices = np.isclose(calls_data["timeToExpiry"], T)
    this_strikes = calls_data.loc[indices, "strike"].values
    price = heston.heston_fft_price(market_data.spot, this_strikes, params[0], ETA, ALPHA,
                                     market_data.interest_rate,
                                     params[1], params[2], params[3], params[4], T, N)
    heston_prices[indices] = price

calls_data = calls_data.copy()
calls_data["hestonIV"] = [
    implied_vol_from_price(market_data.spot, K, T, market_data.interest_rate, heston_price)
    for K, T, heston_price in zip(strikes, maturities, heston_prices)
]

market_pivot = calls_data.pivot(index="strike", columns="timeToExpiry", values="impliedVol")
heston_pivot = calls_data.pivot(index="strike", columns="timeToExpiry", values="hestonIV")

market_pivot = market_pivot.interpolate(axis=0).interpolate(axis=1)
heston_pivot = heston_pivot.interpolate(axis=0).interpolate(axis=1)

diff_pivot = heston_pivot - market_pivot

x_labels = [f"{t:.2f}y" for t in market_pivot.columns]
y_labels = [f"{int(k)}" for k in market_pivot.index]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Market IV with Heston Contours", "Heston − Market IV Error"),
    horizontal_spacing=0.2
)

# Left: Market IV heatmap + Heston contours
fig.add_trace(go.Heatmap(
    z=market_pivot.values,
    x=x_labels,
    y=y_labels,
    colorscale="RdYlGn_r",
    colorbar=dict(x=0.41, title="Market IV", len=0.9),
    hovertemplate="Strike: %{y}<br>Maturity: %{x}<br>Market IV: %{z:.4f}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Contour(
    z=heston_pivot.values,
    x=x_labels,
    y=y_labels,
    showscale=False,
    contours=dict(coloring="lines", showlabels=True, labelfont=dict(size=10, color="black")),
    line=dict(color="black", width=1.5),
    hovertemplate="Heston IV: %{z:.4f}<extra></extra>"
), row=1, col=1)

# Right: Difference heatmap
max_err = np.nanmax(np.abs(diff_pivot.values))
fig.add_trace(go.Heatmap(
    z=diff_pivot.values,
    x=x_labels,
    y=y_labels,
    colorscale="RdBu",
    zmid=0,
    zmin=-max_err,
    zmax=max_err,
    colorbar=dict(x=1.01, title="Error", len=0.9),
    hovertemplate="Strike: %{y}<br>Maturity: %{x}<br>Error: %{z:.4f}<extra></extra>"
), row=1, col=2)

fig.update_layout(
    title=dict(text="SPX Implied Volatility Surface & Heston Calibration", font=dict(size=22)),
    height=600,
    hoverlabel=dict(font_size=13),
    font=dict(color="#e0e0e0"),
    hovermode="x unified",
    template="plotly_dark"
)

fig.update_xaxes(title_text="Maturity", tickangle=45)
fig.update_yaxes(title_text="Strike ($)")

fig.show()
