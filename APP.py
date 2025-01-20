import streamlit as st
import pandas as pd
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

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
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data_clean.empty:
        return "Erro: dados inválidos (NaN/Inf). Verifique limpeza dos dados."
    try:
        modelo = ols(formula, data=data_clean).fit()
        return modelo.summary().as_text()
    except Exception as e:
        return f"Erro na regressão linear: {e}"

def regressao_logistica(data: pd.DataFrame, formula: str):
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
        st.write("Este é um dataset fictício de poços artesianos...")

        st.write("Pré-visualização dos dados:")
        st.dataframe(df_wells)

        csv_bytes = convert_df_to_csv(df_wells)
        st.download_button(
            label="Baixar Dataset como CSV",
            data=csv_bytes,
            file_name="pocos_artesianos.csv",
            mime="text/csv",
        )

        st.markdown("""
        **Interpretação**:
        - Dataset contendo colunas numéricas e categóricas.
        - Trate valores ausentes (NaN) ou Inf caso existam no seu dataset real.
        """)

    # =========================================================
    # GRÁFICO DE FUNÇÃO AUXILIAR
    # =========================================================
    def plot_histogram(data_series, titulo="Histograma", x_label="Valores", bins=10):
        """
        Exibe um histograma com título, rótulo de eixo X e 
        bins personalizáveis.
        """
        plt.figure(figsize=(6,4))
        plt.hist(data_series, bins=bins, alpha=0.7, color="cadetblue", edgecolor="black")
        plt.title(titulo)
        plt.xlabel(x_label)
        plt.ylabel("Frequência")
        st.pyplot(plt.gcf())
        plt.close()

    def plot_boxplot(data: pd.DataFrame, col_numerica: str, col_categ: str, titulo: str):
        """
        Exibe um boxplot de col_numerica separado por grupos col_categ.
        """
        plt.figure(figsize=(6,4))
        sns.boxplot(x=col_categ, y=col_numerica, data=data, palette="Set2")
        plt.title(titulo)
        plt.xlabel(col_categ)
        plt.ylabel(col_numerica)
        st.pyplot(plt.gcf())
        plt.close()

    def plot_scatter(x, y, x_label="X", y_label="Y", titulo="Gráfico de Dispersão"):
        """
        Exibe um scatter plot de x vs y.
        """
        plt.figure(figsize=(6,4))
        plt.scatter(x, y, alpha=0.7, color="teal", edgecolors="black")
        plt.title(titulo)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        st.pyplot(plt.gcf())
        plt.close()

    # =========================================================
    # SEÇÃO 2: Cálculo de Amostragem - Proporção
    # =========================================================
    elif menu == "Cálculo de Amostragem - Proporção":
        st.subheader("Cálculo de Tamanho Amostral para Proporção")
        # ... (cálculo permanece)
        # Não é típico gerar gráfico aqui, pois é um cálculo pontual.
        # Mas, poderíamos mostrar um gauge ou algo. Não implementado por padrão.

        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        margem_erro = st.slider("Margem de Erro (%)", 1, 50, 5, 1)
        p_est = st.number_input("Proporção estimada (0.0 a 1.0)", 0.0, 1.0, 0.5, 0.01)

        if st.button("Calcular"):
            resultado = tamanho_amostral_proporcao(populacao, nivel_confianca, margem_erro, p_est)
            if resultado:
                st.success(f"Tamanho amostral recomendado: {resultado}")
                st.markdown(
                    f"**Interpretação**: Em uma população de {populacao} indivíduos..."
                )
            else:
                st.error("Erro no cálculo.")

    # =========================================================
    # SEÇÃO 3: Cálculo de Amostragem - Média
    # =========================================================
    elif menu == "Cálculo de Amostragem - Média":
        st.subheader("Cálculo de Tamanho Amostral para Média")
        # (idem, sem gráfico por padrão)

        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        margem_erro_abs = st.number_input("Margem de Erro (valor absoluto)", min_value=0.001, value=5.0, step=0.1)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.001, value=10.0, step=0.1)

        if st.button("Calcular"):
            resultado = tamanho_amostral_media(populacao, nivel_confianca, margem_erro_abs, desvio_padrao)
            if resultado:
                st.success(f"Tamanho amostral recomendado: {resultado}")
                st.markdown(
                    f"**Interpretação**: ... "
                )
            else:
                st.error("Erro no cálculo.")

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
            st.info(f"IC Aproximado: {ic[0]*100:.2f}% a {ic[1]*100:.2f}%")
            st.markdown("**Interpretação**: ...")

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
            st.info(f"Intervalo de Confiança: {ic[0]:.2f} a {ic[1]:.2f}")
            st.markdown("**Interpretação**: ...")

    # =========================================================
    # SEÇÃO 6: Estatísticas Descritivas
    # =========================================================
    elif menu == "Estatísticas Descritivas":
        st.subheader("Estatísticas Descritivas")
        file = st.file_uploader("Upload de CSV", type=["csv"], key="desc")

        if file:
            df = pd.read_csv(file)
            df = df.replace([np.inf, -np.inf], np.nan)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            colunas_num = st.multiselect("Colunas numéricas", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])

            if colunas_num:
                desc = estatisticas_descritivas(df[colunas_num].dropna())
                st.write(desc)

                # Opção para plotar histogramas
                if st.checkbox("Exibir Histogramas das colunas selecionadas?"):
                    for c in colunas_num:
                        data_series = df[c].dropna()
                        if len(data_series) > 0:
                            plot_histogram(data_series, titulo=f"Histograma de {c}", x_label=c, bins=10)

    # =========================================================
    # SEÇÃO 7: Testes de Normalidade
    # =========================================================
    elif menu == "Testes de Normalidade":
        st.subheader("Testes de Normalidade")
        file = st.file_uploader("Upload CSV", type=["csv"], key="normal")

        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

            if colunas_num:
                coluna = st.selectbox("Selecione a coluna numérica", colunas_num)
                data_series = df[coluna].dropna()

                if len(data_series) == 0:
                    st.warning("Coluna sem dados numéricos válidos após remoção de NaN/Inf.")
                else:
                    if st.button("Executar Shapiro-Wilk"):
                        stat, p = teste_shapiro(data_series)
                        st.write(f"Shapiro-Wilk: Estatística={stat:.4f}, p-valor={p:.4f}")

                    if st.button("Executar Kolmogorov-Smirnov"):
                        stat, p = teste_ks(data_series)
                        st.write(f"K-S: Estatística={stat:.4f}, p-valor={p:.4f}")

                    # Plotar histograma e qqplot
                    if st.checkbox("Exibir histograma e QQ-Plot"):
                        plot_histogram(data_series, titulo=f"Histograma de {coluna}", x_label=coluna)
                        
                        # QQ-plot usando statsmodels
                        fig = plt.figure(figsize=(6,4))
                        sm.qqplot(data_series, line='s', alpha=0.7)
                        plt.title(f"QQ-Plot de {coluna}")
                        st.pyplot(fig)
                        plt.close(fig)

    # =========================================================
    # SEÇÃO 8: Testes Não-Paramétricos
    # =========================================================
    elif menu == "Testes Não-Paramétricos":
        st.subheader("Testes Não-Paramétricos")
        file = st.file_uploader("Upload CSV", type=["csv"], key="np")

        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            colunas_cat = [c for c in df.columns if (df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c]))]

            if colunas_num and colunas_cat:
                col_num = st.selectbox("Coluna Numérica", colunas_num)
                col_cat = st.selectbox("Coluna Categórica", colunas_cat)

                if st.button("Executar Mann-Whitney"):
                    data_f = df[[col_num, col_cat]].dropna()
                    stat, p = teste_mannwhitney(data_f, col_num, col_cat)
                    if stat is not None:
                        st.write(f"Mann-Whitney U: Estatística={stat:.4f}, p-valor={p:.4f}")
                        if st.checkbox("Exibir Boxplot"):
                            plot_boxplot(data_f, col_num, col_cat, titulo=f"Mann-Whitney: {col_num} vs {col_cat}")

                if st.button("Executar Kruskal-Wallis"):
                    data_f = df[[col_num, col_cat]].dropna()
                    stat, p = teste_kruskal(data_f, col_num, col_cat)
                    st.write(f"Kruskal-Wallis: Estatística={stat:.4f}, p-valor={p:.4f}")
                    if st.checkbox("Exibir Boxplot"):
                        plot_boxplot(data_f, col_num, col_cat, titulo=f"Kruskal-Wallis: {col_num} vs {col_cat}")

    # =========================================================
    # SEÇÃO 9: Two-Way ANOVA
    # =========================================================
    elif menu == "Two-Way ANOVA":
        st.subheader("Two-Way ANOVA")
        file = st.file_uploader("Upload CSV", type=["csv"], key="anova2")

        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            colunas_cat = [c for c in df.columns if (df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c]))]

            if colunas_num and len(colunas_cat) >= 2:
                col_num = st.selectbox("Coluna Numérica (dependente)", colunas_num)
                cat1 = st.selectbox("Fator 1 (categórico)", colunas_cat)
                cat2 = st.selectbox("Fator 2 (categórico)", colunas_cat)

                if st.button("Executar Two-Way ANOVA"):
                    anova_table = anova_two_way(df, col_num, cat1, cat2)
                    if anova_table is not None:
                        st.write(anova_table)

                        if st.checkbox("Exibir Boxplot (cat1)"):
                            data_f = df[[col_num, cat1]].dropna()
                            plot_boxplot(data_f, col_num, cat1, titulo=f"Boxplot de {col_num} por {cat1}")

                        if st.checkbox("Exibir Boxplot (cat2)"):
                            data_f = df[[col_num, cat2]].dropna()
                            plot_boxplot(data_f, col_num, cat2, titulo=f"Boxplot de {col_num} por {cat2}")

    # =========================================================
    # SEÇÃO 10: Regressões
    # =========================================================
    elif menu == "Regressões":
        st.subheader("Regressões")
        file = st.file_uploader("Upload CSV", type=["csv"], key="reg")

        if file:
            df = pd.read_csv(file)
            formula = st.text_input("Fórmula (ex.: 'Y ~ X + C(Grupo)')", value="")
            tipo_regressao = st.selectbox("Tipo de Regressão", ["Linear", "Logística"])

            if st.button("Executar Regressão"):
                if not formula:
                    st.warning("Insira uma fórmula válida.")
                else:
                    if tipo_regressao == "Linear":
                        resultado = regressao_linear(df, formula)
                    else:
                        resultado = regressao_logistica(df, formula)

                    st.text_area("Resultado da Regressão", resultado, height=300)

            # Opcional: Se quisermos exibir um scatter plot para X e Y se houver apenas 1 preditora numérica
            if st.checkbox("Exibir gráfico de Dispersão (somente se houver 1 var. numérica no lado direito da fórmula)"):
                try:
                    # Exemplo básico: extrair 'Y' e 'X' se a fórmula for do tipo "Y ~ X"
                    lhs, rhs = formula.split("~")
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    # Se tiver mais de uma var. no RHS, dificultaria, mas este é apenas exemplo
                    if "+" not in rhs and "C(" not in rhs:
                        x_vals = df[rhs].dropna()
                        y_vals = df[lhs].dropna()
                        plot_scatter(x_vals, y_vals, x_label=rhs, y_label=lhs, titulo="Dispersão para Regressão")
                except Exception as e:
                    st.warning(f"Não foi possível criar gráfico: {e}")

    # =========================================================
    # SEÇÃO 11: Teste de Hipótese (One-Sample t-test)
    # =========================================================
    elif menu == "Teste de Hipótese":
        st.subheader("Teste de Hipótese (One-Sample t-test)")
        file = st.file_uploader("Upload CSV", type=["csv"], key="hipo")

        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col_num = st.selectbox("Coluna Numérica para teste", colunas_num)
            media_hipotetica = st.number_input("Média hipotética (H0)", value=0.0)

            if st.button("Executar One-Sample t-test"):
                data_series = df[col_num].dropna()
                t_stat, p_val = stats.ttest_1samp(data_series, popmean=media_hipotetica)
                st.write(f"Estatística t: {t_stat:.4f}, p-valor: {p_val:.4f}")

                if p_val < 0.05:
                    st.success("Rejeitamos H0 ao nível de 5%.")
                else:
                    st.info("Não rejeitamos H0 ao nível de 5%.")

                if st.checkbox("Exibir Histograma"):
                    plot_histogram(data_series, titulo=f"Histograma da coluna {col_num}", x_label=col_num)

    # =========================================================
    # SEÇÃO 12: Testes de Correlação
    # =========================================================
    elif menu == "Testes de Correlação":
        st.subheader("Testes de Correlação (Pearson, Spearman, Kendall)")
        file = st.file_uploader("Upload CSV", type=["csv"], key="corr")

        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(colunas_num) < 2:
                st.warning("É necessário ao menos duas colunas numéricas.")
            else:
                col_x = st.selectbox("Var. X", colunas_num)
                col_y = st.selectbox("Var. Y", colunas_num)

                if st.button("Executar Pearson"):
                    xvals = df[col_x].dropna()
                    yvals = df[col_y].dropna()
                    corr, p_val = stats.pearsonr(xvals, yvals)
                    st.write(f"**Pearson**: r={corr:.4f}, p-valor={p_val:.4f}")

                if st.button("Executar Spearman"):
                    xvals = df[col_x].dropna()
                    yvals = df[col_y].dropna()
                    corr, p_val = stats.spearmanr(xvals, yvals)
                    st.write(f"**Spearman**: r={corr:.4f}, p-valor={p_val:.4f}")

                if st.button("Executar Kendall"):
                    xvals = df[col_x].dropna()
                    yvals = df[col_y].dropna()
                    corr, p_val = stats.kendalltau(xvals, yvals)
                    st.write(f"**Kendall**: tau={corr:.4f}, p-valor={p_val:.4f}")

                if st.checkbox("Exibir Gráfico de Dispersão"):
                    xvals = df[col_x].dropna()
                    yvals = df[col_y].dropna()
                    plot_scatter(xvals, yvals, x_label=col_x, y_label=col_y,
                                 titulo=f"Dispersão entre {col_x} e {col_y}")

    # =========================================================
    # SEÇÃO 13: Q-Estatística (Cochrane's Q)
    # =========================================================
    elif menu == "Q-Estatística":
        st.subheader("Cálculo de Q-Estatística (Cochrane’s Q)")
        file = st.file_uploader("Upload CSV", type=["csv"], key="qstat")

        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            if st.button("Calcular Q"):
                try:
                    effects = df["effect"].dropna()
                    variances = df["variance"].dropna()
                    Q, p_val = cochrane_q(effects, variances)
                    st.write(f"Q de Cochrane: {Q:.4f}, p-valor={p_val:.4f}")
                except KeyError:
                    st.error("Colunas 'effect' e 'variance' não encontradas.")
                except Exception as e:
                    st.error(f"Erro: {e}.")

    # =========================================================
    # SEÇÃO 14: Q-Exponencial
    # =========================================================
    elif menu == "Q-Exponencial":
        st.subheader("Ajuste Q-Exponencial (Tsallis)")
        file = st.file_uploader("Upload CSV", type=["csv"], key="qexp")

        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

            if colunas_num:
                col_num = st.selectbox("Coluna Numérica", colunas_num)
                if st.button("Ajustar Q-Exponencial"):
                    data_clean = df[col_num].dropna().values
                    if len(data_clean) == 0:
                        st.error("Sem dados válidos após remover NaN/Inf.")
                    else:
                        try:
                            lam_fit, q_fit = fit_q_exponential(data_clean)
                            st.write(f"Parâmetros ajustados: λ={lam_fit:.4f}, q={q_fit:.4f}")

                            # Plotar histograma + curva ajustada
                            counts, bins = np.histogram(data_clean, bins=30, density=True)
                            xvals = 0.5*(bins[1:]+bins[:-1])
                            plt.figure(figsize=(6,4))
                            plt.hist(data_clean, bins=30, density=True, alpha=0.5, color="grey", edgecolor="black", label="Dados")
                            # Gerar curva
                            x_smooth = np.linspace(xvals.min(), xvals.max(), 200)
                            y_smooth = q_exponential_pdf(x_smooth, lam_fit, q_fit)
                            plt.plot(x_smooth, y_smooth, 'r-', label="Curva q-exponencial ajustada")
                            plt.title("Ajuste Q-Exponencial")
                            plt.xlabel(col_num)
                            plt.ylabel("Densidade")
                            plt.legend()
                            st.pyplot(plt.gcf())
                            plt.close()

                        except Exception as e:
                            st.error(f"Falha no ajuste: {e}")

if __name__ == "__main__":
    main()
