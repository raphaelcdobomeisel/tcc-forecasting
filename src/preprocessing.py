import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_processed, save_processed


def clean(df):
    n = len(df)
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    print("  Removidas", n - len(df), "linhas fechadas")
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")
    hm = {"0": "None", "a": "Public", "b": "Easter", "c": "Christmas"}
    df["StateHoliday"] = df["StateHoliday"].map(hm).fillna("None")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfMonth"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)
    df["CompetitionOpen"] = (
        (df["CompetitionOpenSinceYear"] > 0) &
        (
            (df["Year"] > df["CompetitionOpenSinceYear"]) |
            (
                (df["Year"] == df["CompetitionOpenSinceYear"]) &
                (df["Month"] >= df["CompetitionOpenSinceMonth"])
            )
        )
    ).astype(int)
    df["LogSales"] = np.log1p(df["Sales"])
    print("  Shape:", df.shape, "| Lojas:", df["Store"].nunique())
    return df


def get_store_series(df, store_id):
    s = df[df["Store"] == store_id].sort_values("Date")
    return s[["Date", "Sales", "Promo"]].rename(columns={"Date": "ds", "Sales": "y"})


def promo_pattern(series):
    s = series.copy()
    s["weekday"] = s["ds"].dt.weekday
    return s.groupby("weekday")["Promo"].mean().to_dict()


if __name__ == "__main__":
    raw = load_processed("rossmann_merged.csv")
    clean_df = clean(raw)
    save_processed(clean_df, "rossmann_clean.csv")
    s1 = get_store_series(clean_df, store_id=1)
    print("Serie loja 1:", len(s1), "obs")