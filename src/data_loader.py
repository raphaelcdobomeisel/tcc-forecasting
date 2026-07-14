"""
src/data_loader.py
------------------
Carrega e faz o merge dos arquivos brutos do Rossmann Store Sales.
Saída: data/processed/rossmann_merged.csv

Dataset esperado em data/raw/:
  - train.csv
  - store.csv
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_raw() -> pd.DataFrame:
    """Carrega train.csv e store.csv e realiza o merge."""
    print("📂 Carregando dados brutos...")

    train = pd.read_csv(
        RAW_DIR / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False,
    )
    store = pd.read_csv(RAW_DIR / "store.csv", low_memory=False)

    print(f"  train.csv : {train.shape[0]:,} linhas, {train.shape[1]} colunas")
    print(f"  store.csv : {store.shape[0]:,} linhas, {store.shape[1]} colunas")

    # Merge pela coluna Store
    df = train.merge(store, on="Store", how="left")
    print(f"  Merged    : {df.shape[0]:,} linhas, {df.shape[1]} colunas")

    return df


def save_processed(df: pd.DataFrame, filename: str = "rossmann_merged.csv") -> Path:
    """Salva o dataframe processado em data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"✅ Salvo em: {out_path}")
    return out_path


def load_processed(filename: str = "rossmann_merged.csv") -> pd.DataFrame:
    """Lê o CSV processado. Útil para outros módulos."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Execute primeiro: python src/data_loader.py"
        )
    return pd.read_csv(path, parse_dates=["Date"])


if __name__ == "__main__":
    df = load_raw()
    save_processed(df)
    print("\n📊 Amostra:")
    print(df.head(3).to_string())
    print(f"\nColunas: {list(df.columns)}")
