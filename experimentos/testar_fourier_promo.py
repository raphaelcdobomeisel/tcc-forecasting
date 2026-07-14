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

df = clean(load_processed("rossmann_merged.csv"))
store = df[df["Store"] == STORE_ID].sort_values("Date")
data = store[["Date","Sales","Promo"]].rename(columns={"Date":"ds","Sales":"y"})
tr, te = train_test_split_temporal(data, test_days=TEST_DAYS)

real = data.copy(); real["dow"] = real["ds"].dt.dayofweek
dm = real.groupby("dow")["y"].mean(); amp_real = dm.max()-dm.min()
print(f"\n>>> Amplitude semanal REAL: {amp_real:,.0f} EUR\n")

def build(cfg):
    m = Prophet(seasonality_mode="additive", yearly_seasonality=True,
        weekly_seasonality=(cfg["wf"] is None), daily_seasonality=False,
        holidays=HOLIDAYS, changepoint_prior_scale=0.05,
        seasonality_prior_scale=cfg["sps"])
    if cfg["wf"] is not None:
        m.add_seasonality(name="weekly", period=7, fourier_order=cfg["wf"], prior_scale=cfg["sps"])
    if cfg["promo"]:
        m.add_regressor("Promo")
    return m

def amp(fc):
    return float(fc["weekly"].max()-fc["weekly"].min()) if "weekly" in fc.columns else float("nan")

configs = {
 "A. base (F=3, sps=10, sem Promo)": {"wf":None,"sps":10,"promo":False},
 "B. F=6,  sps=10, sem Promo":       {"wf":6,  "sps":10,"promo":False},
 "C. F=10, sps=10, sem Promo":       {"wf":10, "sps":10,"promo":False},
 "D. F=10, sps=20, sem Promo":       {"wf":10, "sps":20,"promo":False},
 "E. base (F=3)  + PROMO":           {"wf":None,"sps":10,"promo":True},
 "F. F=10, sps=20 + PROMO":          {"wf":10, "sps":20,"promo":True},
}

print(f"{'Config':<38}{'MAPE':>8}{'MAE':>8}{'RMSE':>8}{'AmpSem':>9}")
print("-"*71)
res=[]
for n,c in configs.items():
    m=build(c); m.fit(tr)
    fut=m.make_future_dataframe(periods=TEST_DAYS, freq="D")
    if c["promo"]:
        pm=data.set_index("ds")["Promo"].to_dict()
        fut["Promo"]=fut["ds"].map(pm).fillna(0).astype(int)
    fc=m.predict(fut)
    mg=te.merge(fc[["ds","yhat"]], on="ds"); mt=compute_metrics(mg["y"],mg["yhat"])
    a=amp(fc); res.append((n,mt["MAPE (%)"],a))
    print(f"{n:<38}{mt['MAPE (%)']:>7.2f}%{mt['MAE']:>8,.0f}{mt['RMSE']:>8,.0f}{a:>9,.0f}")

best=min(res,key=lambda r:r[1])
print("-"*71)
print(f"\n>>> MELHOR: {best[0]}  (MAPE {best[1]:.2f}%, AmpSem {best[2]:,.0f} / real {amp_real:,.0f})")
print("\nNOTA: se MAPE PIORA com Fourier maior => OVERFIT => usar Fourier menor.")
