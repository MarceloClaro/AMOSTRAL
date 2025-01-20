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
    """
    Obtém o valor de Z com base no nível de confiança. 
    O valor Z é usado para calcular margens de erro nos cálculos amostrais.
    """
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
    """
    Calcula o tamanho da amostra necessária para estimar uma proporção, dado um nível de confiança,
    margem de erro e proporção estimada.
    """
    Z = obter_z(nivel_confianca)
    e = margem_erro / 100.0
    if e == 0:
        return None
    n0 = (Z**2 * p * (1 - p)) / (e**2)
    n_ajustado = n0 / (1 + (n0 - 1) / populacao)
    return math.ceil(n_ajustado)

def tamanho_amostral_media(populacao, nivel_confianca, margem_erro, desvio_padrao):
    """
    Calcula o tamanho da amostra necessário para estimar uma média, dado um nível de confiança,
    margem de erro e desvio padrão.
    """
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
    """
    Calcula o intervalo de confiança para uma proporção.
    """
    Z = obter_z(confianca)
    erro_padrao = math.sqrt(p_observado * (1 - p_observado) / n)
    margem = Z * erro_padrao
    return (p_observado - margem, p_observado + margem)

def intervalo_confianca_media(n, confianca, media_amostral, desvio_padrao):
    """
    Calcula o intervalo de confiança para uma média.
    """
    Z = obter_z(confianca)
    erro_padrao = desvio_padrao / math.sqrt(n)
    margem = Z * erro_padrao
    return (media_amostral - margem, media_amostral + margem)

# ===================================================
# 4) Estatísticas Descritivas
# ===================================================
def estatisticas_descritivas(data: pd.DataFrame):
    """
    Gera as estatísticas descritivas como média, mediana, quartis, etc.
    """
    return data.describe()

# ===================================================
# 5) Testes de Normalidade
# ===================================================
def teste_shapiro(data_series):
    """
    Realiza o teste de normalidade de Shapiro-Wilk para uma série de dados.
    """
    return stats.shapiro(data_series.dropna())

def teste_ks(data_series):
    """
    Realiza o teste de Kolmogorov-Smirnov para normalidade.
    """
    cleaned = data_series.dropna()
    mean = cleaned.mean()
    std = cleaned.std()
    return stats.kstest(cleaned, 'norm', args=(mean, std))

# ===================================================
# 6) Testes Não-Paramétricos
# ===================================================
def teste_mannwhitney(data: pd.DataFrame, col_numerica: str, col_categ: str):
    """
    Realiza o teste de Mann-Whitney para comparação de duas amostras independentes.
    """
    grupos = data[col_categ].unique()
    if len(grupos) != 2:
        return None, None
    grupo1 = data[data[col_categ] == grupos[0]][col_numerica].dropna()
    grupo2 = data[data[col_categ] == grupos[1]][col_numerica].dropna()
    return stats.mannwhitneyu(grupo1, grupo2)

def teste_kruskal(data: pd.DataFrame, col_numerica: str, col_categ: str):
    """
    Realiza o teste de Kruskal-Wallis para comparação de múltiplos grupos independentes.
    """
    grupos = [group[col_numerica].dropna() for name, group in data.groupby(col_categ)]
    return stats.kruskal(*grupos)

# ===================================================
# 7) Two-Way ANOVA
# ===================================================
def anova_two_way(data: pd.DataFrame, col_numerica: str, cat1: str, cat2: str):
    """
    Realiza o Two-Way ANOVA considerando duas variáveis categóricas.
    """
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
    Realiza a regressão linear com base em uma fórmula fornecida.
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
    Realiza a regressão logística com base em uma fórmula fornecida.
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
    """
    Calcula a Q-Estatística de Cochrane usada em meta-análises.
    """
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
    """
    Função de distribuição exponencial generalizada (Q-Exponencial).
    """
    return (2 - q) * lam * np.power((1 - (1-q)*lam*x), 1/(1-q))

def fit_q_exponential(data):
    """
    Ajusta a distribuição Q-Exponencial aos dados.
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
        st.write("Este dataset fictício contém informações sobre vazão, salinidade, pH, composição química e profundidade de poços artesianos.")

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
        - Este dataset contém colunas numéricas (como vazão e salinidade) e categóricas (como tipo geológico e clima).
        - Utilize esses dados para realizar análises estatísticas, como regressões ou cálculos de intervalo de confiança.
        """)

    # Outras seções seguem o mesmo modelo com base no menu
    # Abaixo mostramos a primeira seção do cálculo de amostragem para proporção

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
                    f"**Interpretação**: Em uma população de {populacao} indivíduos, com nível de confiança de {nivel_confianca}% "
                    f"e margem de erro de {margem_erro}%, assumindo que a proporção verdadeira seja de aproximadamente {p_est}, "
                    f"são necessários {resultado} respondentes para obter resultados com a precisão desejada."
                )
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados (margem de erro não pode ser 0%).")

if __name__ == "__main__":
    main()
