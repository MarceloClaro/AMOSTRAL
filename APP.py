import streamlit as st
import pandas as pd
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ===================================================
# 1) Dataset de Poços Artesianos (código e funções)
# ===================================================
@st.cache_data
def convert_df_to_csv(dataframe: pd.DataFrame):
    """
    Converte um DataFrame para CSV em bytes (UTF-8), 
    permitindo o uso no st.download_button.
    """
    return dataframe.to_csv(index=False).encode('utf-8')

def create_well_dataset():
    """
    Retorna um DataFrame fictício com dados de poços artesianos.
    """
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

# ===================================================
# 2) Funções de Cálculo de Tamanho Amostral
# ===================================================
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

# ===================================================
# 3) Funções para Intervalo de Confiança
# ===================================================
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

# ===================================================
# 4) Estatísticas Descritivas
# ===================================================
def estatisticas_descritivas(data: pd.DataFrame):
    return data.describe()

# ===================================================
# 5) Testes de Normalidade
# ===================================================
def teste_shapiro(data_series):
    return stats.shapiro(data_series.dropna())

def teste_ks(data_series):
    cleaned = data_series.dropna()
    mean = cleaned.mean()
    std = cleaned.std()
    return stats.kstest(cleaned, 'norm', args=(mean, std))

# ===================================================
# 6) Testes Não-Paramétricos
# ===================================================
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

# ===================================================
# 7) Two-Way ANOVA
# ===================================================
def anova_two_way(data: pd.DataFrame, col_numerica: str, cat1: str, cat2: str):
    """
    Realiza Two-Way ANOVA considerando col_numerica como variável dependente
    e cat1, cat2 como fatores (categóricos).
    
    Retorna a tabela ANOVA se não houver problemas com Inf/NaN.
    Caso contrário, avisa o usuário sobre dados ausentes ou inválidos.
    """
    # Remove linhas com NaN ou Inf
    data_clean = data[[col_numerica, cat1, cat2]].replace([np.inf, -np.inf], np.nan).dropna()
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

# ===================================================
# 8) Regressões
# ===================================================
def regressao_linear(data: pd.DataFrame, formula: str):
    """
    Regressão linear usando statsmodels. 
    Remove valores Inf/NaN antes de ajustar o modelo.
    """
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data_clean.empty:
        return "Erro: dados inválidos (NaN/Inf). Verifique limpeza dos dados."
    try:
        modelo = ols(formula, data=data_clean).fit()
        return modelo.summary().as_text()
    except Exception as e:
        return f"Erro na regressão linear: {e}"

def regressao_logistica(data: pd.DataFrame, formula: str):
    """
    Regressão logística usando statsmodels. 
    Remove valores Inf/NaN antes de ajustar o modelo.
    """
    import statsmodels.formula.api as smf
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data_clean.empty:
        return "Erro: dados inválidos (NaN/Inf). Verifique limpeza dos dados."
    try:
        modelo = smf.logit(formula, data=data_clean).fit(disp=False)
        return modelo.summary().as_text()
    except Exception as e:
        return f"Erro na regressão logística: {e}"

# ===================================================
# 9) Q-Estatística (Cochrane's Q)
# ===================================================
def cochrane_q(effects, variances):
    w = 1.0 / np.array(variances)
    theta_fixed = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * (effects - theta_fixed)**2)
    df = len(effects) - 1
    p_val = 1 - stats.chi2.cdf(Q, df)
    return Q, p_val

# ===================================================
# 10) Q-Exponencial (Ajuste simplificado)
# ===================================================
def q_exponential_pdf(x, lam, q):
    return (2 - q) * lam * np.power((1 - (1-q)*lam*x), 1/(1-q))

def fit_q_exponential(data):
    """
    Ajusta distribuição q-exponencial usando curve_fit, 
    removendo Inf/NaN antes de estimar parâmetros.
    """
    data_clean = data[~np.isnan(data) & ~np.isinf(data)]
    if len(data_clean) == 0:
        raise ValueError("Dados inválidos: somente valores Inf/NaN encontrados.")
    counts, bins = np.histogram(data_clean, bins=30, density=True)
    xvals = 0.5*(bins[1:] + bins[:-1])
    yvals = counts
    initial_guess = [0.1, 1.2]
    popt, _ = curve_fit(q_exponential_pdf, xvals, yvals, p0=initial_guess, maxfev=10000)
    return popt  # lam, q

# ===================================================
# 11) INTERFACE STREAMLIT (main)
# ===================================================
def main():
    st.title("Ferramenta Avançada de Estatística e Cálculo Amostral")

    # Criação do dataset de poços artesianos para exibição e download
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

    # =========================================================
    # SEÇÃO 1: Dataset de Poços Artesianos
    # =========================================================
    if menu == "Dataset de Poços Artesianos":
        st.subheader("Dataset de Poços Artesianos")
        st.write("Este é um dataset fictício de poços artesianos com informações sobre vazão, salinidade, composição química, clima local e profundidade.")

        # Exibir o DataFrame
        st.write("Pré-visualização dos dados:")
        st.dataframe(df_wells)

        # Botão para download do arquivo CSV
        csv_bytes = convert_df_to_csv(df_wells)
        st.download_button(
            label="Baixar Dataset como CSV",
            data=csv_bytes,
            file_name="pocos_artesianos.csv",
            mime="text/csv",
        )

        st.markdown("""
        **Interpretação**:
        - Este dataset possui colunas numéricas (Flow, Salinity, pH, etc.) e colunas categóricas (Geological_Formation, Climate_Type).
        - Caso seu dataset real tenha valores ausentes (NaN) ou infinitos (Inf), considere limpar ou tratar esses dados antes das análises.
        """)

    # ---------------------------------------------------------
    # SEÇÃO 2: Cálculo de Amostragem - Proporção
    # ---------------------------------------------------------
    elif menu == "Cálculo de Amostragem - Proporção":
        st.subheader("Cálculo de Tamanho Amostral para Proporção")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        margem_erro = st.slider("Margem de Erro (%)", 1, 50, 5, 1)
        p_est = st.number_input("Proporção estimada (0.0 a 1.0)", 0.0, 1.0, 0.5, 0.01)

        if st.button("Calcular"):
            resultado = tamanho_amostral_proporcao(populacao, nivel_confianca, margem_erro, p_est)
            if resultado:
                st.success(f"Tamanho amostral recomendado: {resultado}")
                st.markdown(
                    f"**Interpretação**: Em uma população de {populacao} indivíduos, com nível de confiança de {nivel_confianca}% "
                    f"e margem de erro de {margem_erro}%, assumindo que a proporção verdadeira seja de aproximadamente {p_est}, "
                    f"são necessários {resultado} respondentes para obter resultados com a precisão desejada."
                )
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados (margem de erro não pode ser 0%).")

    # ---------------------------------------------------------
    # SEÇÃO 3: Cálculo de Amostragem - Média
    # ---------------------------------------------------------
    elif menu == "Cálculo de Amostragem - Média":
        st.subheader("Cálculo de Tamanho Amostral para Média")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        margem_erro_abs = st.number_input("Margem de Erro (valor absoluto)", min_value=0.001, value=5.0, step=0.1)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.001, value=10.0, step=0.1)

        if st.button("Calcular"):
            resultado = tamanho_amostral_media(populacao, nivel_confianca, margem_erro_abs, desvio_padrao)
            if resultado:
                st.success(f"Tamanho amostral recomendado: {resultado}")
                st.markdown(
                    f"**Interpretação**: Para uma população de {populacao} indivíduos, nível de confiança de {nivel_confianca}%, "
                    f"margem de erro de ±{margem_erro_abs} e desvio-padrão de {desvio_padrao}, "
                    f"uma amostra de {resultado} elementos ajuda a estimar a média com a precisão desejada."
                )
            else:
                st.error("Erro no cálculo. Verifique se a margem de erro não é zero ou negativa.")

    # ---------------------------------------------------------
    # SEÇÃO 4: Intervalo de Confiança - Proporção
    # ---------------------------------------------------------
    elif menu == "Intervalo de Confiança - Proporção":
        st.subheader("Cálculo de Intervalo de Confiança para Proporção")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=100, step=1)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        p_obs = st.number_input("Proporção Observada (0.0 a 1.0)", 0.0, 1.0, 0.5, 0.01)

        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_proporcao(n, confianca, p_obs)
            st.info(f"Intervalo de Confiança Aproximado: {ic[0]*100:.2f}% a {ic[1]*100:.2f}%")
            st.markdown(
                f"**Interpretação**: Se a proporção amostral é {p_obs*100:.2f}%, com {confianca}% de confiança, "
                f"o valor real da proporção na população está entre {ic[0]*100:.2f}% e {ic[1]*100:.2f}%."
            )

    # ---------------------------------------------------------
    # SEÇÃO 5: Intervalo de Confiança - Média
    # ---------------------------------------------------------
    elif menu == "Intervalo de Confiança - Média":
        st.subheader("Cálculo de Intervalo de Confiança para Média")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=50, step=1)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        media_amostral = st.number_input("Média Observada", value=50.0, step=0.1)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.001, value=10.0, step=0.1)

        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_media(n, confianca, media_amostral, desvio_padrao)
            st.info(f"Intervalo de Confiança Aproximado: {ic[0]:.2f} a {ic[1]:.2f}")
            st.markdown(
                f"**Interpretação**: Dada uma amostra de {n} itens, média {media_amostral} e desvio-padrão {desvio_padrao}, "
                f"com {confianca}% de confiança, a média populacional deve situar-se entre {ic[0]:.2f} e {ic[1]:.2f}."
            )

    # ---------------------------------------------------------
    # SEÇÃO 6: Estatísticas Descritivas
    # ---------------------------------------------------------
    elif menu == "Estatísticas Descritivas":
        st.subheader("Estatísticas Descritivas")
        st.markdown("""
            **Dica**: Se houver valores ausentes ou infinitos no seu dataset, 
            considere removê-los ou tratá-los antes de gerar estatísticas confiáveis.
        """)
        file = st.file_uploader("Faça upload de um arquivo CSV", type=["csv"], key="desc")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            colunas_num = st.multiselect(
                "Selecione colunas numéricas (ex.: valores contínuos ou inteiros)",
                df.columns.tolist(),
                default=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            )
            if colunas_num:
                desc = estatisticas_descritivas(df[colunas_num].replace([np.inf, -np.inf], np.nan).dropna())
                st.write(desc)
                st.markdown(
                    "**Interpretação**: As métricas incluem média, desvio padrão, valor mínimo, valor máximo e quartis. "
                    "Verifique se a quantidade de linhas após remover NaN/Inf é suficiente para uma análise robusta."
                )

    # ---------------------------------------------------------
    # SEÇÃO 7: Testes de Normalidade
    # ---------------------------------------------------------
    elif menu == "Testes de Normalidade":
        st.subheader("Testes de Normalidade")
        file = st.file_uploader("Upload CSV para testes de normalidade", type=["csv"], key="normal")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())

            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            coluna = st.selectbox("Selecione a coluna numérica para teste de normalidade", colunas_num)

            if st.button("Executar Shapiro-Wilk"):
                data_series = df[coluna].dropna()
                if data_series.empty:
                    st.warning("Nenhum dado válido após remover NaN/Inf.")
                else:
                    stat, p = teste_shapiro(data_series)
                    st.write(f"Shapiro-Wilk: Estatística={stat:.4f}, p-valor={p:.4f}")
                    if p < 0.05:
                        st.warning("Resultado sugere que a distribuição não é normal ao nível de 5%.")
                    else:
                        st.info("Não há evidência para rejeitar a normalidade ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: Um p-valor < 0.05 indica que seus dados não seguem uma distribuição normal."
                    )

            if st.button("Executar Kolmogorov-Smirnov"):
                data_series = df[coluna].dropna()
                if data_series.empty:
                    st.warning("Nenhum dado válido após remover NaN/Inf.")
                else:
                    stat, p = teste_ks(data_series)
                    st.write(f"K-S Test: Estatística={stat:.4f}, p-valor={p:.4f}")
                    if p < 0.05:
                        st.warning("Resultado sugere que a distribuição não é normal ao nível de 5%.")
                    else:
                        st.info("Não há evidência para rejeitar a normalidade ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: Um p-valor < 0.05 indica desvio significativo de uma distribuição normal."
                    )

    # ---------------------------------------------------------
    # SEÇÃO 8: Testes Não-Paramétricos
    # ---------------------------------------------------------
    elif menu == "Testes Não-Paramétricos":
        st.subheader("Testes Não-Paramétricos")
        file = st.file_uploader("Upload CSV para testes não-paramétricos", type=["csv"], key="np")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            st.write("Exemplo de dados (após remoção de Inf):")
            st.dataframe(df.head())

            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            colunas_cat = [c for c in df.columns if (df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c]))]

            col_num = st.selectbox("Coluna Numérica (valor contínuo)", colunas_num)
            col_cat = st.selectbox("Coluna Categórica (grupos)", colunas_cat)

            if st.button("Executar Mann-Whitney"):
                data_f = df[[col_num, col_cat]].dropna()
                stat, pval = teste_mannwhitney(df, col_num, col_cat)
                if stat is not None and pval is not None:
                    st.write(f"Mann-Whitney: Estat={stat:.4f}, p={pval:.4f}")
                    if pval < 0.05:
                        st.success("Diferença significativa entre os dois grupos ao nível de 5%.")
                    else:
                        st.info("Não há diferença significativa entre os grupos ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: O teste Mann-Whitney avalia diferenças na distribuição/mediana entre 2 grupos."
                    )
                else:
                    st.error("A coluna categórica selecionada não possui exatamente 2 grupos. Selecione outra.")

            if st.button("Executar Kruskal-Wallis"):
                data_f = df[[col_num, col_cat]].dropna()
                stat, pval = teste_kruskal(df, col_num, col_cat)
                st.write(f"Kruskal-Wallis: Estatística={stat:.4f}, p-valor={pval:.4f}")
                if pval < 0.05:
                    st.success("Diferença significativa entre os grupos ao nível de 5%.")
                else:
                    st.info("Não há diferença significativa entre os grupos ao nível de 5%.")
                st.markdown(
                    "**Interpretação**: O teste Kruskal-Wallis compara 3 ou mais grupos sem pressupor normalidade."
                )

    # ---------------------------------------------------------
    # SEÇÃO 9: Two-Way ANOVA
    # ---------------------------------------------------------
    elif menu == "Two-Way ANOVA":
        st.subheader("Two-Way ANOVA")
        st.markdown("""
            **Orientação**:
            - Selecione uma coluna numérica como variável dependente.
            - Selecione duas colunas categóricas (fatores), cada uma representando grupos/texto.
            - Caso apareça erro de Inf/NaN, verifique se os dados possuem valores ausentes ou infinitos 
              e considere limpá-los antes do teste.
        """)
        file = st.file_uploader("Upload CSV para Two-Way ANOVA", type=["csv"], key="anova2")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            st.write("Exemplo de dados (após remoção de Inf):")
            st.dataframe(df.head())

            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            colunas_cat = [c for c in df.columns if (df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c]))]

            col_num = st.selectbox("Coluna Numérica (dependente)", colunas_num)
            cat1 = st.selectbox("Fator 1 (categórico)", colunas_cat)
            cat2 = st.selectbox("Fator 2 (categórico)", colunas_cat)

            if st.button("Executar Two-Way ANOVA"):
                anova_table = anova_two_way(df, col_num, cat1, cat2)
                if anova_table is not None:
                    st.write(anova_table)
                    st.markdown(
                        "**Interpretação**: A Tabela ANOVA mostra se cada fator (cat1, cat2) e a interação deles "
                        "afetam significativamente a variável numérica. p-valor < 0.05 indica efeito significativo."
                    )
