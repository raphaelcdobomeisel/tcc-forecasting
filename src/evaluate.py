"""
src/evaluate.py
---------------
Funções de avaliação e geração de relatório de métricas.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calcula MAE, RMSE e MAPE entre valores reais e previstos.

    Args:
        y_true: série com valores reais
        y_pred: série com valores previstos

    Returns:
        dict com mae, rmse, mape
    """
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100)

    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


def train_test_split_temporal(
    df: pd.DataFrame,
    test_days: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal: os últimos `test_days` dias vão para o conjunto de teste.
    df deve ter coluna 'ds' (datetime).
    """
    cutoff = df["ds"].max() - pd.Timedelta(days=test_days)
    train  = df[df["ds"] <= cutoff].copy()
    test   = df[df["ds"] >  cutoff].copy()
    return train, test


def metrics_report(metrics: dict, store_id: int) -> str:
    """Formata o relatório de métricas como string legível."""
    lines = [
        f"=== Métricas do Modelo — Loja {store_id} ===",
        f"  MAE    : {metrics['MAE']:.2f}  (Erro médio absoluto em R$)",
        f"  RMSE   : {metrics['RMSE']:.2f}  (Raiz do erro quadrático médio)",
        f"  MAPE   : {metrics['MAPE (%)']:.2f}%  (Erro percentual médio)",
    ]
    interpretation = ""
    mape = metrics["MAPE (%)"]
    if mape < 10:
        interpretation = "✅ Excelente — MAPE < 10%"
    elif mape < 20:
        interpretation = "👍 Bom — MAPE entre 10% e 20%"
    else:
        interpretation = "⚠️ Razoável — MAPE > 20%, considere ajustar hiperparâmetros"
    lines.append(f"  Avaliação: {interpretation}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Teste rápido com dados sintéticos
    np.random.seed(42)
    y_true = pd.Series(np.random.uniform(3000, 8000, 42))
    noise  = np.random.normal(0, 300, 42)
    y_pred = y_true + noise

    metrics = compute_metrics(y_true, y_pred)
    print(metrics_report(metrics, store_id=1))
