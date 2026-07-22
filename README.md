# 📊 TCC — Previsão de Vendas com Prophet

Projeto de Conclusão de Curso em Ciência de Dados.
Pipeline completo de **coleta → tratamento → análise → modelagem → dashboard interativo**.

## 🎯 Objetivo

Prever as vendas diárias de lojas de varejo usando o modelo **Prophet** (Meta/Facebook),
com análise de sazonalidade, tendências e feriados.

**Dataset:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data)

---

## 🗂️ Estrutura do Projeto

```
tcc-forecasting/
├── data/
│   ├── raw/             # Dados brutos do Kaggle (train.csv, store.csv)
│   └── processed/       # Dados tratados e modelos salvos
├── notebooks/
│   └── 01_eda.ipynb     # Análise Exploratória de Dados
├── src/
│   ├── data_loader.py   # Carregamento e merge dos dados
│   ├── preprocessing.py # Limpeza e feature engineering
│   ├── model.py         # Treinamento e previsão com Prophet
│   └── evaluate.py      # Métricas (MAE, RMSE, MAPE)
├── app/
│   └── streamlit_app.py # Dashboard interativo
├── reports/
│   └── figures/         # Gráficos exportados
├── requirements.txt
└── README.md
```

---

## 🚀 Como Rodar

### 1. Clone o repositório
```bash
git clone https://github.com/raphaelcdobomeisel/tcc-forecasting.git
cd tcc-forecasting
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Baixe os dados
Acesse [Rossmann Store Sales no Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data),
baixe `train.csv` e `store.csv` e coloque em `data/raw/`.

### 4. Execute o pipeline
```bash
python src/data_loader.py
python src/preprocessing.py
python src/model.py
```

### 5. Abra o dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🤖 Modelo

- **Algoritmo:** [Prophet](https://facebook.github.io/prophet/) (Meta/Facebook, 2017)
- **Componentes:** Tendência, Sazonalidade semanal, Sazonalidade anual, Feriados
- **Avaliação:** MAE, RMSE, MAPE — validação temporal (hold-out)

---

## 📈 Dashboard

- Seleção de loja e período de previsão
- Gráfico histórico + previsão com intervalo de confiança
- Decomposição de componentes (tendência, sazonalidade)
- Métricas de desempenho e tabela real vs previsto

---

## 🛠️ Tecnologias

| Ferramenta     | Uso                         |
|----------------|-----------------------------|
| Python 3.9    | Linguagem principal          |
| Prophet        | Modelagem de séries temporais|
| Pandas / NumPy | Tratamento de dados          |
| Plotly         | Visualizações interativas    |
| Streamlit      | Dashboard web               |
| Scikit-learn   | Métricas de avaliação        |

---

## 👤 Autor

Raphael Bomeisel — TCC Ciência de Dados
