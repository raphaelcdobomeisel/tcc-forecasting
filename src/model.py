"""
src/model.py
------------
Treinamento e previsão com Prophet + regressor externo Promo.

Decisão de modelagem (documentada via experimento em experimentos/testar_fourier_promo.py):
  - seasonality_mode = 'additive'  (multiplicative zerava a amplitude semanal)
  - regressor Promo  = principal driver: derrubou MAPE de 15.47% -> 7.54%
  - Fourier order e seasonality_prior_scale extras nao ajudaram -> mantidos no padrao
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")

# Feriados alemães relevantes para o dataset Rossmann
GERMAN_SCHOOL_HOLIDAYS = [
    {"holiday": "school_holiday", "ds": pd.Timestamp("2013-07-01"), "lower_window": 0, "upper_window": 42},
    {"holiday": "school_holiday", "ds": pd.Timestamp("2013-10-02"), "lower_window": 0, "upper_window": 16},
    {"holiday": "school_holiday", "ds": pd.Timestamp("2013-12-23"), "lower_window": 0, "upper_window": 13},
    {"holiday": "school_holiday", "ds": pd.Timestamp("2014-03-03"), "lower_window": 0, "upper_window": 4},
    {"holiday": "school_holiday", "ds": pd.Timestamp("2014-04-14"), "lower_window": 0, "upper_window": 18},
    {"holiday": "school_holiday", "ds": pd.Timestamp("2014-07-07"), "lower_window": 0, "upper_window": 42},
    {"holiday": "state_holiday",  "ds": pd.Timestamp("2013-12-25"), "lower_window": 0,  "upper_window": 1},
    {"holiday": "state_holiday",  "ds": pd.Timestamp("2014-04-18"), "lower_window": -1, "upper_window": 1},
    {"holiday": "state_holiday",  "ds": pd.Timestamp("2014-05-01"), "lower_window": 0,  "upper_window": 0},
    {"holiday": "state_holiday",  "ds": pd.Timestamp("2014-10-03"), "lower_window": 0,  "upper_window": 0},
    {"holiday": "state_holiday",  "ds": pd.Timestamp("2014-12-25"), "lower_window": 0,  "upper_window": 1},
]


def build_model() -> Prophet:
    """
    Configura o Prophet com:
      - sazonalidade aditiva (escolhida por experimento)
      - sazonalidades semanal e anual
      - feriados alemães
      - regressor externo Promo (add_regressor chamado aqui)
    """
    holidays_df = pd.DataFrame(GERMAN_SCHOOL_HOLIDAYS)
    model = Prophet(
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    # Regressor externo: flag de promoção (0 ou 1)
    model.add_regressor("Promo")
    return model


def train(df_prophet: pd.DataFrame, store_id: int) -> Prophet:
    """
    Treina o modelo.
    df_prophet deve ter colunas: ds (datetime), y (float), Promo (int 0/1)
    """
    print(f"🤖 Treinando Prophet para loja {store_id} (com regressor Promo)...")
    model = build_model()
    model.fit(df_prophet)
    print("   ✅ Treinamento concluído.")
    return model


def make_future_with_promo(
    model: Prophet,
    historical: pd.DataFrame,
    periods: int,
    promo_scenario: str = "historical_pattern",
    custom_promo: pd.Series = None,
) -> pd.DataFrame:
    """
    Cria o dataframe futuro com o campo Promo preenchido.

    Parâmetros
    ----------
    model            : modelo Prophet treinado
    historical       : série histórica com colunas ds + Promo
    periods          : dias a prever
    promo_scenario   : 'historical_pattern' | 'all_promo' | 'no_promo' | 'custom'
    custom_promo     : pd.Series com index=ds e values=0/1 (só usado quando scenario='custom')

    Cenários disponíveis (para o dashboard what-if):
      'historical_pattern' — repete a taxa histórica de promo por dia da semana
      'all_promo'          — assume promoção em todos os dias futuros
      'no_promo'           — assume sem promoção nos dias futuros
      'custom'             — usa custom_promo fornecido pelo usuário
    """
    future = model.make_future_dataframe(periods=periods, freq="D")

    # Preenche Promo nas datas HISTÓRICAS com o valor real
    promo_map = historical.set_index("ds")["Promo"].to_dict()
    future["Promo"] = future["ds"].map(promo_map)

    # Para datas FUTURAS (NaN após o map), aplica o cenário
    future_mask = future["Promo"].isna()

    if promo_scenario == "all_promo":
        future.loc[future_mask, "Promo"] = 1

    elif promo_scenario == "no_promo":
        future.loc[future_mask, "Promo"] = 0

    elif promo_scenario == "custom" and custom_promo is not None:
        future.loc[future_mask, "Promo"] = future.loc[future_mask, "ds"].map(custom_promo).fillna(0)

    else:  # 'historical_pattern' — default
        # Taxa histórica por dia da semana (0=Seg, 6=Dom)
        historical_copy = historical.copy()
        historical_copy["weekday"] = historical_copy["ds"].dt.weekday
        rate = historical_copy.groupby("weekday")["Promo"].mean()
        # Converte para 0/1 arredondando (>0.5 = promo)
        future.loc[future_mask, "Promo"] = (
            future.loc[future_mask, "ds"].dt.weekday.map(rate).fillna(0).round().astype(int)
        )

    future["Promo"] = future["Promo"].fillna(0).astype(int)
    return future


def predict(
    model: Prophet,
    historical: pd.DataFrame,
    periods: int = 42,
    promo_scenario: str = "historical_pattern",
    custom_promo: pd.Series = None,
) -> pd.DataFrame:
    """
    Gera previsão. Retorna colunas: ds, yhat, yhat_lower, yhat_upper, Promo.
    """
    future = make_future_with_promo(model, historical, periods, promo_scenario, custom_promo)
    # Guarda o Promo antes de prever (prophet pode dropar a coluna)
    promo_col = future[["ds", "Promo"]].copy()
    forecast = model.predict(future)
    # Merge seguro: renomeia para evitar colisao se Promo ja existir no forecast
    if "Promo" in forecast.columns:
        forecast = forecast.drop(columns=["Promo"])
    forecast = forecast.merge(promo_col, on="ds", how="left")
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "Promo"]]


def save_model(model: Prophet, store_id: int, out_dir: Path = None) -> Path:
    """Salva modelo em disco (.pkl)."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"prophet_store_{store_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Modelo salvo: {path}")
    return path


def load_model(store_id: int, model_dir: Path = None) -> Prophet:
    """Carrega modelo salvo."""
    if model_dir is None:
        model_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    path = model_dir / f"prophet_store_{store_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_processed
    from preprocessing import clean, get_store_series
    from evaluate import compute_metrics, train_test_split_temporal

    STORE_ID = 1
    df      = clean(load_processed("rossmann_merged.csv"))
    series  = get_store_series(df, store_id=STORE_ID)   # tem colunas ds, y, Promo
    train_df, test_df = train_test_split_temporal(series, test_days=42)

    model    = train(train_df, store_id=STORE_ID)

    # Previsao usando o Promo REAL do periodo de teste (para medir acuracia)
    forecast = predict(model, historical=series, periods=42,
                       promo_scenario="historical_pattern")

    merged  = test_df.merge(forecast[["ds", "yhat"]], on="ds")
    metrics = compute_metrics(merged["y"], merged["yhat"])

    print(f"\n📐 Métricas (loja {STORE_ID}, 42 dias hold-out, cenario=historical_pattern):")
    print(f"  MAE  = {metrics['MAE']:.2f}")
    print(f"  RMSE = {metrics['RMSE']:.2f}")
    print(f"  MAPE = {metrics['MAPE (%)']:.2f}%")

    save_model(model, store_id=STORE_ID)
