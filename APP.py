import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import curve_fit

########################################
# FUNÇÕES AUXILIARES (Cálculos e Plots)
########################################
def obter_z(conf):
    """ Retorna valor de Z baseado no nível de confiança (conf). """
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
    """Cálculo de amostra para proporção."""
    Z = obter_z(nivel_confianca)
    e = margem_erro / 100.0
    if e == 0:
        return None
    n0 = (Z**2 * p * (1 - p)) / (e**2)
    n_ajustado = n0 / (1 + (n0 - 1) / populacao)
    return math.ceil(n_ajustado)

def tamanho_amostral_media(populacao, nivel_confianca, margem_erro, desvio_padrao):
    """Cálculo de amostra para média."""
    Z = obter_z(nivel_confianca)
    if margem_erro <= 0:
        return None
    n0 = (Z * desvio_padrao / margem_erro)**2
    n_ajustado = n0 / (1 + (n0 - 1) / populacao)
    return math.ceil(n_ajustado)

def intervalo_confianca_proporcao(n, confianca, p_observado):
    """Retorna intervalo de confiança (IC) para proporção."""
    Z = obter_z(confianca)
    erro_padrao = math.sqrt(p_observado * (1 - p_observado) / n)
    margem = Z * erro_padrao
    return (p_observado - margem, p_observado + margem)

def intervalo_confianca_media(n, confianca, media_amostral, desvio_padrao):
    """Retorna intervalo de confiança (IC) para média."""
    Z = obter_z(confianca)
    erro_padrao = desvio_padrao / math.sqrt(n)
    margem = Z * erro_padrao
    return (media_amostral - margem, media_amostral + margem)

def estatisticas_descritivas(data: pd.DataFrame):
    """Retorna as estatísticas descritivas (describe)."""
    return data.describe()

def teste_shapiro(data_series):
    """Teste Shapiro-Wilk."""
    return stats.shapiro(data_series.dropna())

def teste_ks(data_series):
    """Teste Kolmogorov-Smirnov para normalidade."""
    data_clean = data_series.dropna()
    mean = data_clean.mean()
    std = data_clean.std()
    return stats.kstest(data_clean, 'norm', args=(mean, std))

def teste_mannwhitney(data: pd.DataFrame, col_numerica: str, col_categ: str):
    """Teste Mann-Whitney."""
    grupos = data[col_categ].unique()
    if len(grupos) != 2:
        return None, None
    grupo1 = data[data[col_categ] == grupos[0]][col_numerica].dropna()
    grupo2 = data[data[col_categ] == grupos[1]][col_numerica].dropna()
    return stats.mannwhitneyu(grupo1, grupo2)

def teste_kruskal(data: pd.DataFrame, col_numerica: str, col_categ: str):
    """Teste Kruskal-Wallis."""
    grupos = [group[col_numerica].dropna() for name, group in data.groupby(col_categ)]
    return stats.kruskal(*grupos)

def anova_two_way(data: pd.DataFrame, col_numerica: str, cat1: str, cat2: str):
    """Two-Way ANOVA."""
    data_clean = data[[col_numerica, cat1, cat2]].dropna()
    if data_clean.empty:
        st.error("Todos os dados são NaN ou Inf. Verifique seu dataset.")
        return None
    try:
        formula = f"{col_numerica} ~ C({cat1}) + C({cat2}) + C({cat1}):C({cat2})"
        modelo = ols(formula, data=data_clean).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=2)
        return anova_table
    except Exception as e:
        st.error(f"Erro no Two-Way ANOVA: {e}")
        return None

def regressao_linear(data: pd.DataFrame, formula: str):
    """Regressão Linear."""
    data_clean = data.dropna()
    if data_clean.empty:
        return "Sem dados após remoção de NaN/Inf"
    try:
        modelo = ols(formula, data=data_clean).fit()
        return modelo.summary().as_text()
    except Exception as e:
        return f"Erro na regressão linear: {e}"

def regressao_logistica(data: pd.DataFrame, formula: str):
    """Regressão Logística."""
    import statsmodels.formula.api as smf
    data_clean = data.dropna()
    if data_clean.empty:
        return "Sem dados após remoção de NaN/Inf"
    try:
        modelo = smf.logit(formula, data=data_clean).fit(disp=False)
        return modelo.summary().as_text()
    except Exception as e:
        return f"Erro na regressão logística: {e}"

def cochrane_q(effects, variances):
    """Cálculo da Q-Estatística de Cochrane."""
    w = 1.0 / np.array(variances)
    theta_fixed = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * (effects - theta_fixed)**2)
    df = len(effects) - 1
    p_val = 1 - stats.chi2.cdf(Q, df)
    return Q, p_val

def q_exponential_pdf(x, lam, q):
    """Função PDF q-exponencial (Tsallis)."""
    return (2 - q) * lam * np.power((1 - (1-q)*lam*x), 1/(1-q))

def fit_q_exponential(data):
    """Ajuste q-exponencial nos dados."""
    data_clean = data[~np.isnan(data) & ~np.isinf(data)]
    if len(data_clean) == 0:
        raise ValueError("Dados inválidos: apenas Inf/NaN encontrados.")
    counts, bins = np.histogram(data_clean, bins=30, density=True)
    xvals = 0.5*(bins[1:] + bins[:-1])
    yvals = counts
    popt, _ = curve_fit(q_exponential_pdf, xvals, yvals, p0=[0.1, 1.2], maxfev=10000)
    return popt  # lam, q

########################################
# CRIAÇÃO DO DATASET DE EXEMPLO
########################################
def create_well_dataset():
    data = {
        "Well_ID": ["Well_001", "Well_002", "Well_003", "Well_004", "Well_005",
                    "Well_006", "Well_007", "Well_008", "Well_009", "Well_010"],
        "Flow_m3_per_h": [120, 95, 150, 80, 110, 130, 105, 90, 115, 100],
        "Salinity_ppm": [350, 420, 290, 500, 375, 410, 330, 460, 360, 395],
        "pH": [7.2, 6.8, 7.4, 7.0, 7.1, 6.9, 7.3, 6.7, 7.0, 7.2],
        "Calcium_mg_per_L": [150, 180, 130, 160, 155, 170, 140, 165, 150, 158],
        "Magnesium_mg_per_L": [75, 65, 80, 70, 78, 82, 76, 69, 80, 77],
        "Sulfate_mg_per_L": [80, 100, 70, 90, 85, 95, 75, 88, 82, 89],
        "Chloride_mg_per_L": [120, 140, 110, 130, 125, 135, 115, 128, 122, 119],
        "Geological_Formation": ["Granite", "Shale", "Limestone", "Sandstone", "Granite",
                                 "Shale", "Limestone", "Sandstone", "Granite", "Shale"],
        "Climate_Type": ["Temperate", "Arid", "Subtropical", "Continental", "Temperate",
                         "Arid", "Subtropical", "Continental", "Temperate", "Arid"],
        "Latitude": [40.7128, 34.0522, 29.7604, 41.8781, 39.9526,
                     34.0522, 29.7604, 41.8781, 40.7128, 34.0522],
        "Longitude": [-74.0060, -118.2437, -95.3698, -87.6298, -75.1652,
                      -118.2437, -95.3698, -87.6298, -74.0060, -118.2437],
        "Depth_m": [250, 300, 210, 275, 240, 320, 230, 290, 260, 310]
    }
    return pd.DataFrame(data)

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

########################################
# APP (MAIN)
########################################
def main():
    st.title("Ferramenta de Análises Estatísticas com Graficos")

    # Gera dataset fictício
    df_wells = create_well_dataset()

    menu = st.sidebar.radio("Menu", [
        "Dataset de Poços Artesianos",
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

    if menu == "Dataset de Poços Artesianos":
        st.subheader("Dataset")
        st.write("Pré-visualização do dataset de poços artesianos:")
        st.dataframe(df_wells.head())

        csv_data = convert_df_to_csv(df_wells)
        st.download_button("Baixar CSV", data=csv_data, file_name="pocos_artesianos.csv")

        st.markdown("**Interpretação**: ...")

    elif menu == "Cálculo de Amostragem - Proporção":
        st.subheader("Cálculo de Amostra (Proporção)")
        populacao = st.number_input("População (N)", min_value=1, value=1000)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95)
        margem_erro = st.slider("Margem de Erro (%)", 1, 50, 5)
        p_est = st.number_input("Proporção Estimada", 0.0, 1.0, 0.5)

        if st.button("Calcular"):
            n_calc = tamanho_amostral_proporcao(populacao, nivel_confianca, margem_erro, p_est)
            if n_calc:
                st.success(f"Tamanho de amostra recomendado: {n_calc}")
                st.markdown("**Interpretação**: ...")
            else:
                st.error("Erro no cálculo, verifique parâmetros.")

    elif menu == "Cálculo de Amostragem - Média":
        st.subheader("Cálculo de Amostra (Média)")
        populacao = st.number_input("População (N)", min_value=1, value=1000)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95)
        margem_erro = st.number_input("Margem de Erro (absoluto)", 0.1, 999.0, 5.0)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", 0.1, 999.0, 10.0)
        if st.button("Calcular"):
            n_calc = tamanho_amostral_media(populacao, nivel_confianca, margem_erro, desvio_padrao)
            if n_calc:
                st.success(f"Tamanho de amostra recomendado: {n_calc}")
                st.markdown("**Interpretação**: ...")

    elif menu == "Intervalo de Confiança - Proporção":
        st.subheader("IC - Proporção")
        n = st.number_input("n (tamanho amostra)", min_value=1, value=100)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95)
        p_obs = st.number_input("Proporção observada", 0.0, 1.0, 0.5)
        if st.button("Calcular IC"):
            ic = intervalo_confianca_proporcao(n, confianca, p_obs)
            st.info(f"Intervalo: {ic[0]*100:.2f}% até {ic[1]*100:.2f}%")

    elif menu == "Intervalo de Confiança - Média":
        st.subheader("IC - Média")
        n = st.number_input("n (tamanho amostra)", min_value=1, value=50)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95)
        media_obs = st.number_input("Média observada", 0.0, 9999.0, 50.0)
        desvio_padrao = st.number_input("Desvio Padrão (σ)", 0.1, 9999.0, 10.0)
        if st.button("Calcular IC"):
            ic = intervalo_confianca_media(n, confianca, media_obs, desvio_padrao)
            st.info(f"Intervalo de Confiança: {ic[0]:.2f} a {ic[1]:.2f}")

    elif menu == "Estatísticas Descritivas":
        st.subheader("Estatísticas Descritivas")
        file = st.file_uploader("Upload CSV", type=["csv"], key="desc")
        if file:
            df = pd.read_csv(file)
            cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            choose_cols = st.multiselect("Colunas numéricas", cols_num, default=cols_num)

            if choose_cols:
                st.dataframe(df[choose_cols].describe())

                # Exemplo de gráfico (histograma)
                if st.checkbox("Exibir histogramas"):
                    for col in choose_cols:
                        data_series = df[col].dropna()
                        if len(data_series) > 0:
                            fig, ax = plt.subplots(figsize=(6,4))
                            ax.hist(data_series, bins=15, alpha=0.7, edgecolor='black')
                            ax.set_title(f"Histograma de {col}")
                            st.pyplot(fig)

    elif menu == "Testes de Normalidade":
        st.subheader("Testes de Normalidade")
        file = st.file_uploader("Upload CSV", type=["csv"], key="normal")
        if file:
            df = pd.read_csv(file)
            cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col = st.selectbox("Coluna numérica", cols_num)

            if st.button("Shapiro-Wilk"):
                stat, pval = teste_shapiro(df[col])
                st.write(f"Estatística={stat:.4f}, p-valor={pval:.4f}")

            if st.button("Kolmogorov-Smirnov"):
                stat, pval = teste_ks(df[col])
                st.write(f"Estatística={stat:.4f}, p-valor={pval:.4f}")

            # Exemplo de hist e QQ-Plot
            if st.checkbox("Exibir Hist e QQPlot"):
                data_series = df[col].dropna()

                fig, ax = plt.subplots()
                ax.hist(data_series, bins=15, edgecolor='black', alpha=0.6)
                ax.set_title(f"Histograma de {col}")
                st.pyplot(fig)

                # QQ Plot
                fig2 = sm.qqplot(data_series, line='s')
                plt.title(f"QQPlot de {col}")
                st.pyplot(fig2)

    elif menu == "Testes Não-Paramétricos":
        st.subheader("Testes Não-Paramétricos")
        file = st.file_uploader("CSV", type=["csv"], key="np")
        if file:
            df = pd.read_csv(file)
            cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            cols_cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

            if cols_num and cols_cat:
                col_num = st.selectbox("Coluna Numérica", cols_num)
                col_cat = st.selectbox("Coluna Categórica", cols_cat)

                if st.button("Mann-Whitney"):
                    stat, pval = teste_mannwhitney(df, col_num, col_cat)
                    st.write(f"Mann-Whitney: Estat={stat:.4f}, p={pval:.4f}")

                if st.button("Kruskal-Wallis"):
                    stat, pval = teste_kruskal(df, col_num, col_cat)
                    st.write(f"Kruskal-Wallis: Estat={stat:.4f}, p={pval:.4f}")

    elif menu == "Two-Way ANOVA":
        st.subheader("Two-Way ANOVA")
        file = st.file_uploader("CSV", type=["csv"], key="anova2")
        if file:
            df = pd.read_csv(file)
            cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            cols_cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

            if len(cols_cat) >= 2 and cols_num:
                col_num = st.selectbox("Var. Numérica", cols_num)
                cat1 = st.selectbox("Fator1 (cat)", cols_cat)
                cat2 = st.selectbox("Fator2 (cat)", cols_cat)

                if st.button("Executar ANOVA"):
                    res = anova_two_way(df, col_num, cat1, cat2)
                    if res is not None:
                        st.dataframe(res)

    elif menu == "Regressões":
        st.subheader("Regressões")
        file = st.file_uploader("CSV", type=["csv"], key="reg")
        if file:
            df = pd.read_csv(file)
            formula = st.text_input("Fórmula Ex: Y ~ X + C(Grupo)", "")
            tipo = st.selectbox("Tipo", ["Linear", "Logística"])
            if st.button("Rodar Regressão"):
                if tipo == "Linear":
                    out = regressao_linear(df, formula)
                    st.text_area("Saída da Regressão Linear", out, height=300)
                else:
                    out = regressao_logistica(df, formula)
                    st.text_area("Saída da Regressão Logística", out, height=300)

    elif menu == "Teste de Hipótese":
        st.subheader("Teste de Hipótese (One-Sample t-test)")
        file = st.file_uploader("CSV", type=["csv"], key="hipo")
        if file:
            df = pd.read_csv(file)
            cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col = st.selectbox("Col. Numérica", cols_num)
            media_hipot = st.number_input("Média hipotética", 0.0, 9999.0, 0.0)
            if st.button("Executar t-test"):
                data_series = df[col].dropna()
                stat, pval = stats.ttest_1samp(data_series, popmean=media_hipot)
                st.write(f"T={stat:.4f}, p={pval:.4f}")

    elif menu == "Testes de Correlação":
        st.subheader("Testes de Correlação")
        file = st.file_uploader("CSV", type=["csv"], key="corr")
        if file:
            df = pd.read_csv(file)
            cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(cols_num) >= 2:
                x_var = st.selectbox("X", cols_num)
                y_var = st.selectbox("Y", cols_num)

                if st.button("Pearson"):
                    corr, pval = stats.pearsonr(df[x_var].dropna(), df[y_var].dropna())
                    st.write(f"Pearson: r={corr:.4f}, p={pval:.4f}")

                if st.button("Spearman"):
                    corr, pval = stats.spearmanr(df[x_var].dropna(), df[y_var].dropna())
                    st.write(f"Spearman: r={corr:.4f}, p={pval:.4f}")

                if st.button("Kendall"):
                    corr, pval = stats.kendalltau(df[x_var].dropna(), df[y_var].dropna())
                    st.write(f"Kendall: tau={corr:.4f}, p={pval:.4f}")

    elif menu == "Q-Estatística":
        st.subheader("Q-Estatística (Cochrane)")
        file = st.file_uploader("CSV", type=["csv"], key="qstat")
        if file:
            df = pd.read_csv(file)
            if st.button("Calcular Q"):
                try:
                    effects = df["effect"].dropna()
                    variances = df["variance"].dropna()
                    Q, p_val = cochrane_q(effects, variances)
                    st.write(f"Cochrane Q={Q:.4f}, p={p_val:.4f}")
                except KeyError:
                    st.error("Colunas 'effect' e 'variance' não encontradas.")

    elif menu == "Q-Exponencial":
        st.subheader("Q-Exponencial (Tsallis)")
        file = st.file_uploader("CSV", type=["csv"], key="qexp")
        if file:
            df = pd.read_csv(file)
            cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col = st.selectbox("Col. Numérica", cols_num)
            if st.button("Ajustar q-exponencial"):
                data_series = df[col].dropna()
                lam, q = fit_q_exponential(data_series)
                st.write(f"Parâmetros: λ={lam:.4f}, q={q:.4f}")

                # Gráfico
                counts, bins = np.histogram(data_series, bins=30, density=True)
                xvals = 0.5*(bins[1:]+bins[:-1])
                plt.figure(figsize=(6,4))
                plt.hist(data_series, bins=30, density=True, alpha=0.6, edgecolor='black', label="Dados")
                x_smooth = np.linspace(xvals.min(), xvals.max(), 200)
                y_smooth = q_exponential_pdf(x_smooth, lam, q)
                plt.plot(x_smooth, y_smooth, 'r-', label="Curva q-exponencial")
                plt.title("Ajuste q-exponencial")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()

if __name__ == "__main__":
    main()
