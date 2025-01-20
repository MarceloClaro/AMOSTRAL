import streamlit as st
import pandas as pd
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------------------------------------------------
# Funções de Cálculo de Tamanho Amostral
# ---------------------------------------------------
def obter_z(conf):
    if conf <= 80:
        return 1.28
    elif conf <= 85:
        return 1.44
    elif conf <= 90:
        return 1.64
    elif conf <= 95:
        return 1.96
    elif conf < 100:
        return 2.58
    else:
        return 2.58

def tamanho_amostral_proporcao(populacao, nivel_confianca, margem_erro, p=0.5):
    Z = obter_z(nivel_confianca)
    e = margem_erro / 100.0
    if e == 0:
        return None
    n0 = (Z**2 * p * (1 - p)) / (e**2)
    n_ajustado = n0 / (1 + (n0 - 1) / populacao)
    return math.ceil(n_ajustado)

def tamanho_amostral_media(populacao, nivel_confianca, margem_erro, desvio_padrao):
    Z = obter_z(nivel_confianca)
    if margem_erro <= 0:
        return None
    n0 = (Z * desvio_padrao / margem_erro)**2
    n_ajustado = n0 / (1 + (n0 - 1) / populacao)
    return math.ceil(n_ajustado)

# ---------------------------------------------------
# Funções para Intervalo de Confiança
# ---------------------------------------------------
def intervalo_confianca_proporcao(n, confianca, p_observado):
    Z = obter_z(confianca)
    erro_padrao = math.sqrt(p_observado * (1 - p_observado) / n)
    margem = Z * erro_padrao
    return (p_observado - margem, p_observado + margem)

def intervalo_confianca_media(n, confianca, media_amostral, desvio_padrao):
    Z = obter_z(confianca)
    erro_padrao = desvio_padrao / math.sqrt(n)
    margem = Z * erro_padrao
    return (media_amostral - margem, media_amostral + margem)

# ---------------------------------------------------
# Estatísticas Descritivas
# ---------------------------------------------------
def estatisticas_descritivas(data: pd.DataFrame):
    return data.describe()

# ---------------------------------------------------
# Testes de Normalidade
# ---------------------------------------------------
def teste_shapiro(data_series):
    return stats.shapiro(data_series.dropna())

def teste_ks(data_series):
    cleaned = data_series.dropna()
    mean = cleaned.mean()
    std = cleaned.std()
    return stats.kstest(cleaned, 'norm', args=(mean, std))

# ---------------------------------------------------
# Testes Não-Paramétricos
# ---------------------------------------------------
def teste_mannwhitney(data: pd.DataFrame, col_numerica: str, col_categ: str):
    grupos = data[col_categ].unique()
    if len(grupos) != 2:
        return None, None
    grupo1 = data[data[col_categ] == grupos[0]][col_numerica].dropna()
    grupo2 = data[data[col_categ] == grupos[1]][col_numerica].dropna()
    return stats.mannwhitneyu(grupo1, grupo2)

def teste_kruskal(data: pd.DataFrame, col_numerica: str, col_categ: str):
    grupos = [group[col_numerica].dropna() for name, group in data.groupby(col_categ)]
    return stats.kruskal(*grupos)

# ---------------------------------------------------
# Two-Way ANOVA
# ---------------------------------------------------
def anova_two_way(data: pd.DataFrame, col_numerica: str, cat1: str, cat2: str):
    formula = f"{col_numerica} ~ C({cat1}) + C({cat2}) + C({cat1}):C({cat2})"
    try:
        modelo = ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=2)
        return anova_table
    except Exception as e:
        st.error(f"Erro no Two-Way ANOVA: {e}")
        return None

# ---------------------------------------------------
# Regressões
# ---------------------------------------------------
def regressao_linear(data: pd.DataFrame, formula: str):
    try:
        modelo = ols(formula, data=data).fit()
        return modelo.summary().as_text()
    except Exception as e:
        return f"Erro na regressão linear: {e}"

def regressao_logistica(data: pd.DataFrame, formula: str):
    import statsmodels.formula.api as smf
    try:
        modelo = smf.logit(formula, data=data).fit(disp=False)
        return modelo.summary().as_text()
    except Exception as e:
        return f"Erro na regressão logística: {e}"

# ---------------------------------------------------
# Q-Estatística (Cochrane's Q)
# ---------------------------------------------------
def cochrane_q(effects, variances):
    w = 1.0 / np.array(variances)
    theta_fixed = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * (effects - theta_fixed)**2)
    df = len(effects) - 1
    p_val = 1 - stats.chi2.cdf(Q, df)
    return Q, p_val

# ---------------------------------------------------
# Q-Exponencial (Ajuste simplificado)
# ---------------------------------------------------
from scipy.optimize import curve_fit

def q_exponential_pdf(x, lam, q):
    # Implementação simplificada do PDF q-exponencial
    # Observação: esta função pode precisar de ajustes para garantir validade
    return (2 - q) * lam * np.power((1 - (1-q)*lam*x), 1/(1-q))

def fit_q_exponential(data):
    initial_guess = [0.1, 1.2]
    counts, bins = np.histogram(data, bins=30, density=True)
    xvals = 0.5*(bins[1:] + bins[:-1])
    yvals = counts
    popt, _ = curve_fit(q_exponential_pdf, xvals, yvals, p0=initial_guess, maxfev=10000)
    return popt  # lam, q

# ---------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------
def main():
    st.title("Ferramenta Avançada de Estatística e Cálculo Amostral")

    menu = st.sidebar.radio("Menu", [
        "Home",
        "Cálculo de Amostragem - Proporção",
        "Cálculo de Amostragem - Média",
        "Intervalo de Confiança - Proporção",
        "Intervalo de Confiança - Média",
        "Estatísticas Descritivas",
        "Testes de Normalidade",
        "Testes Não-Paramétricos",
        "Two-Way ANOVA",
        "Regressões",
        "Teste de Hipótese",
        "Testes de Correlação",
        "Q-Estatística",
        "Q-Exponencial"
    ])

    # --- Home ---
    if menu == "Home":
        st.subheader("Bem-vindo(a)!")
        st.markdown("""
        Utilize este aplicativo para realizar análises estatísticas avançadas:
        - Cálculo de tamanho amostral para proporção e média
        - Cálculo de intervalos de confiança
        - Estatísticas descritivas
        - Testes de normalidade e não-paramétricos
        - Two-Way ANOVA
        - Regressões lineares e logísticas
        - Teste de hipótese genérico (One-Sample t-test)
        - Testes de correlação (Pearson, Spearman, Kendall)
        - Q-Estatística (Cochrane's Q)
        - Q-Exponencial (ajuste de distribuição)
        """)

    # --- Cálculo de Amostragem - Proporção ---
    if menu == "Cálculo de Amostragem - Proporção":
        st.subheader("Cálculo de Tamanho Amostral para Proporção")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        margem_erro = st.slider("Margem de Erro (%)", 1, 50, 5, 1)
        p_est = st.number_input("Proporção estimada (0.0 a 1.0)", 0.0, 1.0, 0.5, 0.01)
        if st.button("Calcular"):
            resultado = tamanho_amostral_proporcao(populacao, nivel_confianca, margem_erro, p_est)
            if resultado:
                st.success(f"Tamanho amostral recomendado: {resultado}")
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados.")

    # --- Cálculo de Amostragem - Média ---
    if menu == "Cálculo de Amostragem - Média":
        st.subheader("Cálculo de Tamanho Amostral para Média")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        margem_erro_abs = st.number_input("Margem de Erro (valor absoluto)", min_value=0.001, value=5.0, step=0.1)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.001, value=10.0, step=0.1)
        if st.button("Calcular"):
            resultado = tamanho_amostral_media(populacao, nivel_confianca, margem_erro_abs, desvio_padrao)
            if resultado:
                st.success(f"Tamanho amostral recomendado: {resultado}")
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados.")

    # --- Intervalo de Confiança - Proporção ---
    if menu == "Intervalo de Confiança - Proporção":
        st.subheader("Cálculo de Intervalo de Confiança para Proporção")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=100, step=1)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        p_obs = st.number_input("Proporção Observada (0.0 a 1.0)", 0.0, 1.0, 0.5, 0.01)
        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_proporcao(n, confianca, p_obs)
            st.info(f"Intervalo de Confiança Aproximado: {ic[0]*100:.2f}% a {ic[1]*100:.2f}%")

    # --- Intervalo de Confiança - Média ---
    if menu == "Intervalo de Confiança - Média":
        st.subheader("Cálculo de Intervalo de Confiança para Média")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=50, step=1)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        media_amostral = st.number_input("Média Observada", value=50.0, step=0.1)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.001, value=10.0, step=0.1)
        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_media(n, confianca, media_amostral, desvio_padrao)
            st.info(f"Intervalo de Confiança Aproximado: {ic[0]:.2f} a {ic[1]:.2f}")

    # --- Estatísticas Descritivas ---
    if menu == "Estatísticas Descritivas":
        st.subheader("Estatísticas Descritivas")
        file = st.file_uploader("Faça upload de um arquivo CSV", type=["csv"], key="desc")
        if file:
            df = pd.read_csv(file)
            st.write("Dados:")
            st.dataframe(df.head())
            colunas_num = st.multiselect("Selecione colunas numéricas", df.columns.tolist(),
                                         default=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            if colunas_num:
                st.write(estatisticas_descritivas(df[colunas_num]))

    # --- Testes de Normalidade ---
    if menu == "Testes de Normalidade":
        st.subheader("Testes de Normalidade")
        file = st.file_uploader("Upload CSV para testes de normalidade", type=["csv"], key="normal")
        if file:
            df = pd.read_csv(file)
            st.write("Dados:")
            st.dataframe(df.head())
            coluna = st.selectbox("Selecione a coluna numérica para teste de normalidade", 
                                  [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            if st.button("Executar Shapiro-Wilk"):
                stat, p = teste_shapiro(df[coluna])
                st.write(f"Shapiro-Wilk: Estatística={stat:.4f}, p-valor={p:.4f}")
            if st.button("Executar Kolmogorov-Smirnov"):
                stat, p = teste_ks(df[coluna])
                st.write(f"K-S Test: Estatística={stat:.4f}, p-valor={p:.4f}")

    # --- Testes Não-Paramétricos ---
    if menu == "Testes Não-Paramétricos":
        st.subheader("Testes Não-Paramétricos")
        file = st.file_uploader("Upload CSV para testes não-paramétricos", type=["csv"], key="np")
        if file:
            df = pd.read_csv(file)
            st.write("Dados:")
            st.dataframe(df.head())
            col_num = st.selectbox("Coluna Numérica", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            col_cat = st.selectbox("Coluna Categórica", [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])])
            if st.button("Executar Mann-Whitney"):
                stat, p = teste_mannwhitney(df, col_num, col_cat)
                if stat is not None:
                    st.write(f"Mann-Whitney: Estatística={stat:.4f}, p-valor={p:.4f}")
                else:
                    st.error("Mann-Whitney requer exatamente 2 grupos na coluna categórica.")
            if st.button("Executar Kruskal-Wallis"):
                stat, p = teste_kruskal(df, col_num, col_cat)
                st.write(f"Kruskal-Wallis: Estatística={stat:.4f}, p-valor={p:.4f}")

    # --- Two-Way ANOVA ---
    if menu == "Two-Way ANOVA":
        st.subheader("Two-Way ANOVA")
        file = st.file_uploader("Upload CSV para Two-Way ANOVA", type=["csv"], key="anova2")
        if file:
            df = pd.read_csv(file)
            st.write("Dados:")
            st.dataframe(df.head())
            col_num = st.selectbox("Coluna Numérica", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            cat1 = st.selectbox("Fator 1", [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])])
            cat2 = st.selectbox("Fator 2", [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])])
            if st.button("Executar Two-Way ANOVA"):
                anova_table = anova_two_way(df, col_num, cat1, cat2)
                if anova_table is not None:
                    st.write(anova_table)

    # --- Regressões ---
    if menu == "Regressões":
        st.subheader("Regressões")
        file = st.file_uploader("Upload CSV para regressões", type=["csv"], key="reg")
        if file:
            df = pd.read_csv(file)
            st.write("Dados:")
            st.dataframe(df.head())
            st.markdown("Digite a fórmula para o modelo estatístico. Exemplo para regressão linear: `Y ~ X1 + X2`")
            formula = st.text_input("Fórmula", value="")
            tipo_regressao = st.selectbox("Tipo de Regressão", ["Linear", "Logística"])
            if st.button("Executar Regressão"):
                if not formula:
                    st.warning("Por favor, insira uma fórmula para o modelo.")
                else:
                    if tipo_regressao == "Linear":
                        resultado = regressao_linear(df, formula)
                    else:
                        resultado = regressao_logistica(df, formula)
                    st.text_area("Resultado da Regressão", resultado, height=300)

    # --- Teste de Hipótese ---
    if menu == "Teste de Hipótese":
        st.subheader("Teste de Hipótese (One-Sample t-test)")
        file = st.file_uploader("Upload CSV para teste de hipótese", type=["csv"], key="hipo")
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())
            col_num = st.selectbox("Selecione a coluna numérica para teste", 
                                   [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], key="hipo_col")
            media_hipotetica = st.number_input("Média hipotética (H0)", value=0.0, key="hipo_mean")
            if st.button("Executar One-Sample t-test"):
                data_series = df[col_num].dropna()
                t_stat, p_val = stats.ttest_1samp(data_series, popmean=media_hipotetica)
                st.write(f"Estatística t: {t_stat:.4f}, p-valor: {p_val:.4f}")
                if p_val < 0.05:
                    st.success("Rejeitamos H0 ao nível de 5%.")
                else:
                    st.info("Não rejeitamos H0 ao nível de 5%.")

    # --- Testes de Correlação ---
    if menu == "Testes de Correlação":
        st.subheader("Testes de Correlação (Pearson, Spearman, Kendall)")
        file = st.file_uploader("Upload CSV para correlação", type=["csv"], key="corr")
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(colunas_num) < 2:
                st.warning("O arquivo deve conter ao menos duas colunas numéricas.")
            else:
                col_x = st.selectbox("Selecione a primeira variável (X)", colunas_num, key="corr_x")
                col_y = st.selectbox("Selecione a segunda variável (Y)", colunas_num, key="corr_y")
                if st.button("Executar Pearson"):
                    corr, p_val = stats.pearsonr(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Pearson**: {corr:.4f}, p-valor={p_val:.4f}")
                if st.button("Executar Spearman"):
                    corr, p_val = stats.spearmanr(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Spearman**: {corr:.4f}, p-valor={p_val:.4f}")
                if st.button("Executar Kendall"):
                    corr, p_val = stats.kendalltau(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Kendall**: {corr:.4f}, p-valor={p_val:.4f}")

    # --- Q-Estatística ---
    if menu == "Q-Estatística":
        st.subheader("Cálculo de Q-Estatística (Cochrane’s Q para meta-análise)")
        file = st.file_uploader("Upload CSV com efeitos e variâncias", type=["csv"], key="qstat")
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())
            if st.button("Calcular Q"):
                try:
                    Q, p_val = cochrane_q(df["effect"], df["variance"])
                    st.write(f"**Q de Cochrane**: {Q:.4f}")
                    st.write(f"**p-valor de heterogeneidade**: {p_val:.4f}")
                except Exception as e:
                    st.error(f"Erro: {e}. Verifique as colunas 'effect' e 'variance' no CSV.")

    # --- Q-Exponencial ---
    if menu == "Q-Exponencial":
        st.subheader("Ajuste Q-Exponencial (Estatística de Tsallis)")
        file = st.file_uploader("Upload CSV com dados para ajuste", type=["csv"], key="qexp")
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())
            col_num = st.selectbox("Selecione a coluna numérica", 
                                   [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], key="qexp_col")
            if st.button("Ajustar Q-Exponencial"):
                data = df[col_num].dropna().values
                try:
                    lam_fit, q_fit = fit_q_exponential(data)
                    st.write(f"Parâmetros ajustados: λ = {lam_fit:.4f}, q = {q_fit:.4f}")
                except Exception as e:
                    st.error(f"Falha ao ajustar Q-Exponencial: {e}")

if __name__ == "__main__":
    main()
