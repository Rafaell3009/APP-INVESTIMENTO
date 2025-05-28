import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

st.set_page_config(page_title="Análise Intraday com IA", layout="wide")

st.title("📊 Análise Intraday com Machine Learning (5 Min / 30 Dias)")

# 🔥 Sidebar - seleção de ativos
ativos = st.sidebar.multiselect(
    "Escolha os ativos:",
    ['AAPL', 'MSFT', 'PETR4.SA', 'VALE3.SA', 'BTC-USD', 'ETH-USD', 'GOOGL', 'TSLA', 'META'],
    default=['AAPL', 'BTC-USD']
)

st.sidebar.write("Período fixo: Últimos 30 dias")
st.sidebar.write("Intervalo fixo: 5 minutos")

if st.sidebar.button("🔍 Analisar"):

    resultados = []

    with st.spinner("🔄 Coletando dados e processando..."):

        for ticker in ativos:
            try:
                dados = yf.download(ticker, period='30d', interval='5m', progress=False)

                if dados.empty or len(dados) < 100:
                    st.warning(f"❌ Dados insuficientes para {ticker}")
                    continue

                dados['Retorno'] = dados['Close'].pct_change()
                dados['SMA_10'] = dados['Close'].rolling(window=10).mean()
                dados['SMA_50'] = dados['Close'].rolling(window=50).mean()
                dados['Hora'] = dados.index.hour
                dados.dropna(inplace=True)

                dados['Target'] = np.where(dados['Close'].shift(-1) > dados['Close'], 1, 0)

                X = dados[['Retorno', 'SMA_10', 'SMA_50']].fillna(0)
                y = dados['Target']

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, shuffle=False
                )

                modelo = LogisticRegression()
                modelo.fit(X_train, y_train)

                acc = accuracy_score(y_test, modelo.predict(X_test)) * 100
                prob_subir = modelo.predict_proba([X_scaled[-1]])[0][1] * 100

                grupo_horas = dados.groupby('Hora')['Retorno'].agg(['mean', 'std', 'count'])
                grupo_horas['Prob_Positivo'] = 1 - norm.cdf(0, grupo_horas['mean'], grupo_horas['std'])

                melhor_hora = grupo_horas['Prob_Positivo'].idxmax()
                melhor_prob = grupo_horas.loc[melhor_hora, 'Prob_Positivo'] * 100

                resultados.append({
                    'Ativo': ticker,
                    'Acuracia (%)': round(acc, 2),
                    'Prob. Alta Próximo Candle (%)': round(prob_subir, 2),
                    'Melhor Hora': f'{melhor_hora}:00',
                    'Prob. Melhor Hora (%)': round(melhor_prob, 2)
                })

            except Exception as e:
                st.error(f"⚠️ Erro no ativo {ticker}: {e}")

    if resultados:
        df_resultados = pd.DataFrame(resultados)

        st.subheader("📈 Resultados da Análise")
        st.dataframe(df_resultados)

        st.subheader("📊 Gráfico das Probabilidades de Alta no Próximo Candle")

        plt.figure(figsize=(10,5))
        sns.barplot(
            x='Ativo', y='Prob. Alta Próximo Candle (%)',
            data=df_resultados.sort_values(by='Prob. Alta Próximo Candle (%)', ascending=False),
            palette="viridis"
        )
        plt.title('Probabilidade de Alta no Próximo Candle (5 Min / 30 Dias)')
        plt.ylabel('Probabilidade (%)')
        plt.xlabel('Ativo')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

    else:
        st.warning("⚠️ Nenhum dado processado. Verifique os ativos selecionados.")
