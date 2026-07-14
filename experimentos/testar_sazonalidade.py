import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_loader import load_processed
from preprocessing import clean, get_store_series
from evaluate import compute_metrics, train_test_split_temporal

STORE_ID, TEST_DAYS = 1, 42
HOLIDAYS = pd.DataFrame([
    {"holiday":"state_holiday","ds":pd.Timestamp("2013-12-25"),"lower_window":0,"upper_window":1},
    {"holiday":"state_holiday","ds":pd.Timestamp("2014-04-18"),"lower_window":-1,"upper_window":1},
    {"holiday":"state_holiday","ds":pd.Timestamp("2014-12-25"),"lower_window":0,"upper_window":1},
    {"holiday":"state_holiday","ds":pd.Timestamp("2015-05-01"),"lower_window":0,"upper_window":0},
])

def build(cfg):
    m = Prophet(seasonality_mode=cfg["seasonality_mode"], yearly_seasonality=True,
        weekly_seasonality=cfg.get("weekly_auto", True), daily_seasonality=False,
        holidays=HOLIDAYS, changepoint_prior_scale=cfg.get("cps", 0.05),
        seasonality_prior_scale=cfg.get("sps", 10.0))
    if not cfg.get("weekly_auto", True):
        m.add_seasonality(name="weekly", period=7,
            fourier_order=cfg.get("wf", 3), prior_scale=cfg.get("sps", 10.0))
    return m

def amp(fc):
    return float(fc["weekly"].max()-fc["weekly"].min()) if "weekly" in fc.columns else float("nan")

print("\nCarregando dados...")
df = clean(load_processed("rossmann_merged.csv"))
series = get_store_series(df, STORE_ID)
tr, te = train_test_split_temporal(series, test_days=TEST_DAYS)
real = series.copy()
real["dow"] = real["ds"].dt.dayofweek
dm = real.groupby("dow")["y"].mean()
amp_real = dm.max() - dm.min()
print(f"\n>>> Amplitude semanal REAL (referencia): {amp_real:,.0f} EUR")
print("    (o modelo ideal se aproxima desse valor)\n")

configs = {
 "1. ATUAL  (mult,  prior=10, fourier=auto)": {"seasonality_mode":"multiplicative","sps":10.0,"weekly_auto":True},
 "2. Additive, prior=10, fourier=auto":       {"seasonality_mode":"additive","sps":10.0,"weekly_auto":True},
 "3. Additive, prior=25, fourier=auto":       {"seasonality_mode":"additive","sps":25.0,"weekly_auto":True},
 "4. Additive, prior=25, fourier=10":         {"seasonality_mode":"additive","sps":25.0,"weekly_auto":False,"wf":10},
}

print(f"{'Config':<44} {'MAPE':>7} {'MAE':>8} {'RMSE':>8} {'AmpSem':>9}")
print("-" * 78)
res = []
for n, c in configs.items():
    m = build(c)
    m.fit(tr)
    fc = m.predict(m.make_future_dataframe(periods=TEST_DAYS, freq="D"))
    mg = te.merge(fc[["ds","yhat"]], on="ds")
    mt = compute_metrics(mg["y"], mg["yhat"])
    a  = amp(fc)
    res.append((n, mt["MAPE (%)"], a))
    print(f"{n:<44} {mt['MAPE (%)']:>6.2f}% {mt['MAE']:>8,.0f} {mt['RMSE']:>8,.0f} {a:>9,.0f}")

best = min(res, key=lambda r: r[1])
print("-" * 78)
print(f"\n>>> MELHOR MAPE  : {best[0]}")
print(f"    MAPE         : {best[1]:.2f}%")
print(f"    AmpSem modelo: {best[2]:,.0f} EUR  (real ~ {amp_real:,.0f} EUR)")
