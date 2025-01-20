import streamlit as st
import pandas as pd
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import curve_fit

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
    formula = f"{col_numerica} ~ C({cat1}) + C({cat2}) + C({cat1}):C({cat2})"
    try:
        modelo = ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=2)
        return anova_table
    except Exception as e:
        st.error(f"Erro no Two-Way ANOVA: {e}")
        return None

# ===================================================
# 8) Regressões
# ===================================================
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
    initial_guess = [0.1, 1.2]
    counts, bins = np.histogram(data, bins=30, density=True)
    xvals = 0.5*(bins[1:] + bins[:-1])
    yvals = counts
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
        - O dataset inclui colunas de **Flow**, **Salinity**, **pH**, íons (Calcium, Magnesium, Sulfate, Chloride),
          bem como **informações geológicas** (Geological_Formation), **climáticas** (Climate_Type) e **geográficas** (Latitude, Longitude).
        - Isso pode servir como teste para análises exploratórias, regressões ou qualquer método estatístico desejado.
        """)

    # =========================================================
    # SEÇÃO 2: Cálculo de Amostragem - Proporção
    # =========================================================
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
                    f"**Interpretação**: Para uma população de {populacao} indivíduos, com nível de confiança de {nivel_confianca}% "
                    f"e margem de erro de {margem_erro}%, assumindo proporção verdadeira por volta de {p_est}, "
                    f"o tamanho de amostra recomendado é {resultado} para alcançar a precisão desejada."
                )
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados.")

    # =========================================================
    # SEÇÃO 3: Cálculo de Amostragem - Média
    # =========================================================
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
                    f"uma amostra de {resultado} elementos é indicada para estimar a média com a precisão desejada."
                )
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados.")

    # =========================================================
    # SEÇÃO 4: Intervalo de Confiança - Proporção
    # =========================================================
    elif menu == "Intervalo de Confiança - Proporção":
        st.subheader("Cálculo de Intervalo de Confiança para Proporção")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=100, step=1)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        p_obs = st.number_input("Proporção Observada (0.0 a 1.0)", 0.0, 1.0, 0.5, 0.01)

        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_proporcao(n, confianca, p_obs)
            st.info(f"Intervalo de Confiança Aproximado: {ic[0]*100:.2f}% a {ic[1]*100:.2f}%")
            st.markdown(
                f"**Interpretação**: Com {confianca}% de confiança, se a proporção amostral for {p_obs*100:.2f}%, "
                f"o valor real da proporção populacional deve estar entre {ic[0]*100:.2f}% e {ic[1]*100:.2f}%."
            )

    # =========================================================
    # SEÇÃO 5: Intervalo de Confiança - Média
    # =========================================================
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
                f"**Interpretação**: Para uma amostra de {n} itens, média de {media_amostral}, "
                f"desvio-padrão {desvio_padrao} e {confianca}% de confiança, "
                f"o intervalo ({ic[0]:.2f}, {ic[1]:.2f}) abrange o valor provável da média populacional."
            )

    # =========================================================
    # SEÇÃO 6: Estatísticas Descritivas
    # =========================================================
    elif menu == "Estatísticas Descritivas":
        st.subheader("Estatísticas Descritivas")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um arquivo CSV.
            2. Selecione as colunas numéricas que deseja analisar (colunas que contêm valores decimais ou inteiros).
            3. As colunas com valores de texto ou categorias não devem ser selecionadas para estatísticas descritivas.
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
                desc = estatisticas_descritivas(df[colunas_num])
                st.write(desc)
                st.markdown(
                    "**Interpretação**: As métricas incluem média, desvio padrão, valor mínimo, valor máximo e quartis. "
                    "Elas descrevem a tendência central e a dispersão dos dados."
                )

    # =========================================================
    # SEÇÃO 7: Testes de Normalidade
    # =========================================================
    elif menu == "Testes de Normalidade":
        st.subheader("Testes de Normalidade")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um CSV que contenha ao menos uma coluna numérica (ex.: colunas de valores contínuos).
            2. Selecione a coluna numérica para verificar se os dados seguem distribuição normal.
        """)
        file = st.file_uploader("Upload CSV para testes de normalidade", type=["csv"], key="normal")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            # Filtra apenas colunas numéricas
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            coluna = st.selectbox("Selecione a coluna numérica para teste de normalidade", colunas_num)

            if st.button("Executar Shapiro-Wilk"):
                stat, p = teste_shapiro(df[coluna])
                st.write(f"Shapiro-Wilk: Estatística={stat:.4f}, p-valor={p:.4f}")
                if p < 0.05:
                    st.warning("Resultado sugere que a distribuição não é normal ao nível de 5%.")
                else:
                    st.info("Não há evidência para rejeitar a normalidade ao nível de 5%.")
                st.markdown(
                    "**Interpretação**: p-valor < 0.05 indica que a amostra provavelmente não segue distribuição normal."
                )

            if st.button("Executar Kolmogorov-Smirnov"):
                stat, p = teste_ks(df[coluna])
                st.write(f"K-S Test: Estatística={stat:.4f}, p-valor={p:.4f}")
                if p < 0.05:
                    st.warning("Resultado sugere que a distribuição não é normal ao nível de 5%.")
                else:
                    st.info("Não há evidência para rejeitar a normalidade ao nível de 5%.")
                st.markdown(
                    "**Interpretação**: p-valor < 0.05 sugere que a amostra se desvia de uma distribuição normal."
                )

    # =========================================================
    # SEÇÃO 8: Testes Não-Paramétricos
    # =========================================================
    elif menu == "Testes Não-Paramétricos":
        st.subheader("Testes Não-Paramétricos")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um CSV.
            2. Identifique qual a **coluna numérica** (ex.: valores contínuos).
            3. Escolha a **coluna categórica** (ex.: grupos representados por texto ou 'category').
            - Para Mann-Whitney: a coluna categórica deve conter **exatamente 2 grupos**.
            - Para Kruskal-Wallis: pode conter 3 ou mais grupos.
        """)
        file = st.file_uploader("Upload CSV para testes não-paramétricos", type=["csv"], key="np")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())

            # Identifica possíveis colunas
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            colunas_cat = [c for c in df.columns if (df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c]))]

            col_num = st.selectbox("Coluna Numérica (valor contínuo)", colunas_num)
            col_cat = st.selectbox("Coluna Categórica (ex.: grupos, texto ou category)", colunas_cat)

            if st.button("Executar Mann-Whitney"):
                stat, p = teste_mannwhitney(df, col_num, col_cat)
                if stat is not None:
                    st.write(f"Mann-Whitney: Estatística={stat:.4f}, p-valor={p:.4f}")
                    if p < 0.05:
                        st.success("Diferença significativa entre os dois grupos ao nível de 5%.")
                    else:
                        st.info("Não há diferença significativa entre os grupos ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: O teste Mann-Whitney avalia se duas amostras independentes "
                        "têm distribuições (medianas) diferentes."
                    )
                else:
                    st.error("Mann-Whitney requer exatamente 2 grupos na coluna categórica.")

            if st.button("Executar Kruskal-Wallis"):
                stat, p = teste_kruskal(df, col_num, col_cat)
                st.write(f"Kruskal-Wallis: Estatística={stat:.4f}, p-valor={p:.4f}")
                if p < 0.05:
                    st.success("Diferença significativa entre os grupos ao nível de 5%.")
                else:
                    st.info("Não há diferença significativa entre os grupos ao nível de 5%.")
                st.markdown(
                    "**Interpretação**: O teste Kruskal-Wallis compara três ou mais grupos sem pressupor normalidade. "
                    "p-valor < 0.05 indica que ao menos um grupo difere dos demais."
                )

    # =========================================================
    # SEÇÃO 9: Two-Way ANOVA
    # =========================================================
    elif menu == "Two-Way ANOVA":
        st.subheader("Two-Way ANOVA")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um CSV.
            2. Selecione a **coluna numérica** (valor contínuo) como variável dependente.
            3. Escolha duas **colunas categóricas** (ex.: fator 1 e fator 2, que representem grupos, como texto ou category).
        """)
        file = st.file_uploader("Upload CSV para Two-Way ANOVA", type=["csv"], key="anova2")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
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
                        "**Interpretação**: Cada linha representa o efeito de um fator ou da interação entre fatores. "
                        "Verifique os p-valores para saber se há efeitos significativos na variável numérica."
                    )

    # =========================================================
    # SEÇÃO 10: Regressões
    # =========================================================
    elif menu == "Regressões":
        st.subheader("Regressões")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um CSV.
            2. Observe quais colunas são numéricas (variáveis quantitativas) e categóricas (grupos, texto).
            3. Monte a fórmula no estilo `VariavelDependente ~ VariavelIndependente1 + VariavelIndependente2`.
               - Para variáveis categóricas, normalmente o modelo as interpreta como fatores automaticamente (ex.: `C(Coluna)` se precisar explicitar).
            4. Selecione o tipo de regressão (Linear ou Logística).
        """)
        file = st.file_uploader("Upload CSV para regressões", type=["csv"], key="reg")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())

            st.markdown("Informe a fórmula. Ex.: `VariavelDependente ~ VariavelIndependente1 + VariavelIndependente2`")
            formula = st.text_input("Fórmula", value="")
            tipo_regressao = st.selectbox("Tipo de Regressão", ["Linear", "Logística"])

            if st.button("Executar Regressão"):
                if not formula:
                    st.warning("Insira uma fórmula para o modelo.")
                else:
                    if tipo_regressao == "Linear":
                        resultado = regressao_linear(df, formula)
                    else:
                        resultado = regressao_logistica(df, formula)
                    st.text_area("Resultado da Regressão", resultado, height=300)
                    st.markdown(
                        "**Interpretação**: Observe coeficientes, p-valores e estatísticas de ajuste. "
                        "Na regressão linear, o R² indica quanto o modelo explica da variação da variável dependente. "
                        "Na logística, verifique os odds-ratios (exp(coef)) e seus intervalos de confiança."
                    )

    # =========================================================
    # SEÇÃO 11: Teste de Hipótese (One-Sample t-test)
    # =========================================================
    elif menu == "Teste de Hipótese":
        st.subheader("Teste de Hipótese (One-Sample t-test)")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um CSV.
            2. Selecione a **coluna numérica** (ex.: valores contínuos).
            3. Defina uma média hipotética (H0) para comparar.
        """)
        file = st.file_uploader("Upload CSV para teste de hipótese", type=["csv"], key="hipo")
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())

            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col_num = st.selectbox("Selecione a coluna numérica para teste", colunas_num, key="hipo_col")
            media_hipotetica = st.number_input("Média hipotética (H0)", value=0.0, key="hipo_mean")

            if st.button("Executar One-Sample t-test"):
                data_series = df[col_num].dropna()
                t_stat, p_val = stats.ttest_1samp(data_series, popmean=media_hipotetica)
                st.write(f"Estatística t: {t_stat:.4f}, p-valor: {p_val:.4f}")
                if p_val < 0.05:
                    st.success("Rejeitamos H0 ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: O p-valor < 0.05 sugere que a média amostral difere significativamente "
                        "da média hipotética definida em H0."
                    )
                else:
                    st.info("Não rejeitamos H0 ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: Não há evidências para afirmar que a média amostral seja diferente "
                        "da média hipotética (H0)."
                    )

    # =========================================================
    # SEÇÃO 12: Testes de Correlação
    # =========================================================
    elif menu == "Testes de Correlação":
        st.subheader("Testes de Correlação (Pearson, Spearman, Kendall)")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um CSV que contenha ao menos duas colunas numéricas.
            2. Selecione a primeira e a segunda coluna numérica para avaliar a correlação.
            3. Escolha o teste desejado (Pearson, Spearman ou Kendall).
        """)
        file = st.file_uploader("Upload CSV para correlação", type=["csv"], key="corr")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(colunas_num) < 2:
                st.warning("O arquivo deve conter ao menos duas colunas numéricas para correlação.")
            else:
                col_x = st.selectbox("Selecione a primeira variável (X)", colunas_num, key="corr_x")
                col_y = st.selectbox("Selecione a segunda variável (Y)", colunas_num, key="corr_y")

                if st.button("Executar Pearson"):
                    corr, p_val = stats.pearsonr(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Pearson**: {corr:.4f}, p-valor={p_val:.4f}")
                    st.markdown(
                        "**Interpretação (Pearson)**: Correlação linear. p-valor < 0.05 indica correlação linear significativa."
                    )

                if st.button("Executar Spearman"):
                    corr, p_val = stats.spearmanr(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Spearman**: {corr:.4f}, p-valor={p_val:.4f}")
                    st.markdown(
                        "**Interpretação (Spearman)**: Correlação baseada em ranques (monotônica). p-valor < 0.05 indica correlação significativa."
                    )

                if st.button("Executar Kendall"):
                    corr, p_val = stats.kendalltau(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Kendall**: {corr:.4f}, p-valor={p_val:.4f}")
                    st.markdown(
                        "**Interpretação (Kendall)**: Também ranque, mas abordagem diferente de Spearman. p-valor < 0.05 indica correlação significativa."
                    )

    # =========================================================
    # SEÇÃO 13: Q-Estatística (Cochrane's Q)
    # =========================================================
    elif menu == "Q-Estatística":
        st.subheader("Cálculo de Q-Estatística (Cochrane’s Q para meta-análise)")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um CSV contendo as colunas 'effect' (efeito estimado em cada estudo) e 'variance' (variância desses efeitos).
            2. A Q-Estatística (Cochrane's Q) serve para verificar se há heterogeneidade significativa entre estudos em meta-análise.
        """)
        file = st.file_uploader("Upload CSV com efeitos e variâncias", type=["csv"], key="qstat")
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())
            if st.button("Calcular Q"):
                try:
                    Q, p_val = cochrane_q(df["effect"], df["variance"])
                    st.write(f"**Q de Cochrane**: {Q:.4f}")
                    st.write(f"**p-valor de heterogeneidade**: {p_val:.4f}")
                    if p_val < 0.05:
                        st.warning("Há heterogeneidade significativa entre os estudos.")
                    else:
                        st.info("Não há evidências de heterogeneidade significativa.")
                    st.markdown(
                        "**Interpretação**: p-valor < 0.05 indica que os estudos não são homogêneos, ou seja, "
                        "há diferença entre eles que pode impactar a meta-análise."
                    )
                except Exception as e:
                    st.error(f"Erro: {e}. Verifique se as colunas 'effect' e 'variance' existem no CSV.")

    # =========================================================
    # SEÇÃO 14: Q-Exponencial
    # =========================================================
    elif menu == "Q-Exponencial":
        st.subheader("Ajuste Q-Exponencial (Estatística de Tsallis)")
        st.markdown("""
            **Instruções**:
            1. Faça upload de um CSV contendo ao menos uma coluna numérica (ex.: valores contínuos).
            2. Selecione a coluna que deseja ajustar a uma distribuição q-exponencial.
            3. O método tentará ajustar os parâmetros da distribuição com base nos dados.
        """)
        file = st.file_uploader("Upload CSV com dados para ajuste", type=["csv"], key="qexp")
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col_num = st.selectbox("Selecione a coluna numérica", colunas_num, key="qexp_col")

            if st.button("Ajustar Q-Exponencial"):
                data = df[col_num].dropna().values
                try:
                    lam_fit, q_fit = fit_q_exponential(data)
                    st.write(f"**Parâmetros ajustados**: λ = {lam_fit:.4f}, q = {q_fit:.4f}")
                    st.markdown(
                        "**Interpretação**: A distribuição q-exponencial é uma generalização da exponencial. "
                        "Valores de q próximos de 1 indicam comportamento semelhante à exponencial simples; "
                        "valores de q diferentes de 1 sugerem sistemas não-extensivos (Tsallis)."
                    )
                except Exception as e:
                    st.error(f"Falha ao ajustar Q-Exponencial: {e}")

if __name__ == "__main__":
    main()
