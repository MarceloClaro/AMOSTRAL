import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Tenta importar dcor; se não for possível, desativa a funcionalidade
try:
    import dcor
    dcor_installed = True
except ImportError:
    dcor_installed = False

# ============================
# FUNÇÕES AUXILIARES E DATASETS
# ============================

def create_well_dataset():
    """Dataset de Poços Artesianos (exemplo)"""
    data = {
        "Well_ID": [f"Well_{i:03d}" for i in range(1, 11)],
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
        "Depth_m": [250, 300, 210, 275, 240, 320, 230, 290, 260, 310]
    }
    return pd.DataFrame(data)

def create_synthetic_data(n_rows=100):
    """Cria um dataset sintético com dados numéricos inteiros e categóricos."""
    np.random.seed(42)
    df = pd.DataFrame({
        "ID": range(1, n_rows + 1),
        "Categoria": np.random.choice(["A", "B", "C"], size=n_rows),
        "Valor_Int": np.random.randint(0, 100, size=n_rows),
        "Valor_Real": np.random.rand(n_rows) * 100,
        "Grupo": np.random.choice(["X", "Y"], size=n_rows)
    })
    return df

def convert_df_to_csv(dataframe: pd.DataFrame):
    return dataframe.to_csv(index=False).encode('utf-8')

# ============================
# FUNÇÕES DE ESTATÍSTICA DESCRITIVA
# ============================
def estatisticas_descritivas(df: pd.DataFrame):
    return df.describe()

# ============================
# FUNÇÕES DE INTERVALO DE CONFIANÇA
# ============================
def intervalo_confianca_proporcao(n, confianca, p_obs):
    Z = stats.norm.ppf(1 - (1 - confianca/100)/2)
    erro_padrao = np.sqrt(p_obs * (1 - p_obs) / n)
    margem = Z * erro_padrao
    return (p_obs - margem, p_obs + margem)

def intervalo_confianca_media(n, confianca, media, desvio):
    Z = stats.norm.ppf(1 - (1 - confianca/100)/2)
    erro_padrao = desvio / np.sqrt(n)
    margem = Z * erro_padrao
    return (media - margem, media + margem)

# ============================
# FUNÇÕES DE TESTES DE NORMALIDADE
# ============================
def teste_shapiro(data_series):
    return stats.shapiro(data_series.dropna())

# ============================
# FUNÇÕES DE TESTES NÃO-PARAMÉTRICOS
# ============================
def teste_mannwhitney(data: pd.DataFrame, col_numerica, col_categ):
    grupos = data[col_categ].unique()
    if len(grupos) != 2:
        return None, None
    grupo1 = data[data[col_categ] == grupos[0]][col_numerica].dropna()
    grupo2 = data[data[col_categ] == grupos[1]][col_numerica].dropna()
    return stats.mannwhitneyu(grupo1, grupo2)

# ============================
# FUNÇÕES DE TWO-WAY ANOVA
# ============================
def anova_two_way(data: pd.DataFrame, dep_var, fator1, fator2):
    formula = f"{dep_var} ~ C({fator1}) + C({fator2}) + C({fator1}):C({fator2})"
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

# ============================
# FUNÇÕES DE REGRESSÃO
# ============================
def regressao_linear(data: pd.DataFrame, formula: str):
    model = ols(formula, data=data).fit()
    return model.summary().as_text()

# ============================
# FUNÇÕES DE Q-ESTATÍSTICA E Q-EXPONENCIAL
# ============================
def cochrane_q(effects, variances):
    w = 1.0 / np.array(variances)
    theta_fixed = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * (effects - theta_fixed)**2)
    df = len(effects) - 1
    p_val = 1 - stats.chi2.cdf(Q, df)
    return Q, p_val

def q_exponential_pdf(x, lam, q):
    return (2 - q) * lam * np.power((1 - (1-q)*lam*x), 1/(1-q))

def fit_q_exponential(data):
    data_clean = data[~np.isnan(data) & ~np.isinf(data)]
    counts, bins = np.histogram(data_clean, bins=30, density=True)
    xvals = 0.5*(bins[1:] + bins[:-1])
    yvals = counts
    initial_guess = [0.1, 1.2]
    popt, _ =  stats.curve_fit(q_exponential_pdf, xvals, yvals, p0=initial_guess, maxfev=10000)
    return popt  # lam, q

# ============================
# FUNÇÕES DE CORRELAÇÃO (incluindo as inovações)
# ============================
def correlacao_pearson(x, y):
    corr, pval = stats.pearsonr(x, y)
    return corr, pval

def correlacao_spearman(x, y):
    corr, pval = stats.spearmanr(x, y)
    return corr, pval

def correlacao_kendall(x, y):
    corr, pval = stats.kendalltau(x, y)
    return corr, pval

def correlacao_distancia(x, y):
    if not dcor_installed:
        st.error("O módulo 'dcor' não está instalado. Instale-o para utilizar esta funcionalidade.")
        return None
    return dcor.distance_correlation(x, y)

def correlacao_parcial(x, y, control):
    model_x = ols("x ~ control", data=pd.DataFrame({'x': x, 'control': control})).fit()
    res_x = model_x.resid
    model_y = ols("y ~ control", data=pd.DataFrame({'y': y, 'control': control})).fit()
    res_y = model_y.resid
    r, pval = stats.pearsonr(res_x, res_y)
    return r, pval

# ============================
# FUNÇÕES DE CLUSTERING
# ============================
def clustering_section():
    st.subheader("Técnicas de Clustering")
    st.markdown("""
    Nesta seção, você pode aplicar diferentes técnicas de clustering para segmentar seus dados.
    Escolha o algoritmo e os parâmetros para visualizar a segmentação.
    """)
    file = st.file_uploader("Envie um arquivo CSV para clustering", type=["csv"], key="clustering")
    if file:
        df = pd.read_csv(file).dropna()
        colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(colunas_num) < 2:
            st.error("Selecione um dataset com ao menos duas variáveis numéricas.")
            return
        # Selecione as variáveis para clustering (ex.: duas para visualização)
        var1 = st.selectbox("Selecione a primeira variável", colunas_num, key="clust_var1")
        var2 = st.selectbox("Selecione a segunda variável", colunas_num, key="clust_var2")
        data = df[[var1, var2]].values
        
        # Escolha o algoritmo
        algoritmo = st.selectbox("Escolha o algoritmo de clustering", 
                                 options=["KMeans", "Hierarchical", "DBSCAN"])
        
        if algoritmo == "KMeans":
            n_clusters = st.number_input("Número de clusters", min_value=2, value=3, step=1)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(data)
        elif algoritmo == "Hierarchical":
            n_clusters = st.number_input("Número de clusters", min_value=2, value=3, step=1)
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(data)
        elif algoritmo == "DBSCAN":
            eps = st.number_input("eps (distância máxima)", min_value=0.1, value=0.5, step=0.1)
            min_samples = st.number_input("min_samples", min_value=1, value=5, step=1)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)
        
        df["Cluster"] = labels
        st.markdown("**Visualização dos Clusters**")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x=df[var1], y=df[var2], hue=df["Cluster"], palette="viridis", ax=ax)
        ax.set_title(f"Clustering com {algoritmo}")
        st.pyplot(fig)
        st.dataframe(df.head())

# ============================
# FUNÇÃO PARA CSV SINTÉTICO
# ============================
def synthetic_csv_section():
    st.subheader("Gerador de CSV Sintético")
    n_rows = st.number_input("Número de linhas", min_value=10, value=100, step=10)
    df_synth = create_synthetic_data(n_rows)
    st.markdown("**Pré-visualização do dataset sintético:**")
    st.dataframe(df_synth.head())
    csv_bytes = convert_df_to_csv(df_synth)
    st.download_button("Baixar CSV Sintético", data=csv_bytes, file_name="sintetico.csv")

# ============================
# OUTRAS SEÇÕES (exemplos simplificados)
# ============================
def estatistica_descritiva_section():
    st.subheader("Estatísticas Descritivas")
    file = st.file_uploader("Envie um CSV para análise descritiva", type=["csv"], key="desc")
    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())
        st.markdown("**Estatísticas Descritivas:**")
        st.write(estatisticas_descritivas(df))

def intervalo_confianca_section(tipo="proporcao"):
    st.subheader(f"Intervalo de Confiança - {'Proporção' if tipo=='proporcao' else 'Média'}")
    if tipo == "proporcao":
        n = st.number_input("Tamanho da amostra (n)", min_value=1, value=100)
        confianca = st.slider("Nível de confiança (%)", 80, 99, 95)
        p_obs = st.number_input("Proporção observada", 0.0, 1.0, 0.5)
        ic = intervalo_confianca_proporcao(n, confianca, p_obs)
        st.write(f"Intervalo de Confiança: {ic[0]*100:.2f}% a {ic[1]*100:.2f}%")
    else:
        n = st.number_input("Tamanho da amostra (n)", min_value=1, value=50)
        confianca = st.slider("Nível de confiança (%)", 80, 99, 95)
        media = st.number_input("Média observada", value=50.0)
        desvio = st.number_input("Desvio-padrão", value=10.0)
        ic = intervalo_confianca_media(n, confianca, media, desvio)
        st.write(f"Intervalo de Confiança: {ic[0]:.2f} a {ic[1]:.2f}")

def normalidade_section():
    st.subheader("Teste de Normalidade (Shapiro-Wilk)")
    file = st.file_uploader("Envie um CSV", type=["csv"], key="normal")
    if file:
        df = pd.read_csv(file).dropna()
        col = st.selectbox("Selecione a coluna numérica", df.select_dtypes(include=[np.number]).columns)
        stat, p = teste_shapiro(df[col])
        st.write(f"Estatística: {stat:.4f}, p-valor: {p:.4f}")

def nao_parametricos_section():
    st.subheader("Testes Não-Paramétricos")
    file = st.file_uploader("Envie um CSV", type=["csv"], key="nao_param")
    if file:
        df = pd.read_csv(file).dropna()
        num_col = st.selectbox("Variável Numérica", df.select_dtypes(include=[np.number]).columns)
        cat_col = st.selectbox("Variável Categórica", df.select_dtypes(include=["object"]).columns)
        stat, p = teste_mannwhitney(df, num_col, cat_col)
        if stat is None:
            st.error("A variável categórica deve ter exatamente 2 grupos.")
        else:
            st.write(f"Mann-Whitney: Estatística = {stat:.4f}, p-valor = {p:.4f}")

def anova_section():
    st.subheader("Two-Way ANOVA")
    file = st.file_uploader("Envie um CSV", type=["csv"], key="anova")
    if file:
        df = pd.read_csv(file).dropna()
        dep_var = st.selectbox("Variável dependente", df.select_dtypes(include=[np.number]).columns)
        cat1 = st.selectbox("Fator 1", df.select_dtypes(include=["object"]).columns, key="anova_cat1")
        cat2 = st.selectbox("Fator 2", df.select_dtypes(include=["object"]).columns, key="anova_cat2")
        anova_table = anova_two_way(df, dep_var, cat1, cat2)
        st.dataframe(anova_table)

def regressao_section():
    st.subheader("Regressão Linear")
    file = st.file_uploader("Envie um CSV", type=["csv"], key="regressao")
    if file:
        df = pd.read_csv(file).dropna()
        dep_var = st.selectbox("Variável dependente", df.select_dtypes(include=[np.number]).columns)
        indep_vars = st.multiselect("Variáveis independentes", df.columns.tolist())
        if dep_var and indep_vars:
            # Se a variável for categórica, encapsula com C() para tratamento no modelo
            terms = [f"C({var})" if df[var].dtype == object else var for var in indep_vars]
            formula = f"{dep_var} ~ " + " + ".join(terms)
            resultado = regressao_linear(df, formula)
            st.text_area("Resumo da Regressão", resultado, height=300)

def hipotese_section():
    st.subheader("Teste de Hipótese (One-Sample t-test)")
    file = st.file_uploader("Envie um CSV", type=["csv"], key="hipotese")
    if file:
        df = pd.read_csv(file).dropna()
        col = st.selectbox("Selecione a coluna numérica", df.select_dtypes(include=[np.number]).columns)
        media_hipot = st.number_input("Média hipotética", value=0.0)
        stat, p = stats.ttest_1samp(df[col], popmean=media_hipot)
        st.write(f"t = {stat:.4f}, p-valor = {p:.4f}")

def q_estat_section():
    st.subheader("Q-Estatística (Cochrane’s Q)")
    file = st.file_uploader("Envie um CSV com colunas 'effect' e 'variance'", type=["csv"], key="q_estat")
    if file:
        df = pd.read_csv(file).dropna()
        try:
            effects = df["effect"]
            variances = df["variance"]
            Q, p_val = cochrane_q(effects, variances)
            st.write(f"Q = {Q:.4f}, p-valor = {p_val:.4f}")
        except KeyError:
            st.error("Colunas 'effect' e 'variance' não encontradas no CSV.")

def q_exponencial_section():
    st.subheader("Q-Exponencial")
    file = st.file_uploader("Envie um CSV para ajuste Q-Exponencial", type=["csv"], key="q_exponencial")
    if file:
        df = pd.read_csv(file).dropna()
        col = st.selectbox("Selecione a coluna numérica", df.select_dtypes(include=[np.number]).columns)
        data_series = df[col]
        try:
            lam, q = fit_q_exponential(data_series)
            st.write(f"Parâmetros ajustados: λ = {lam:.4f}, q = {q:.4f}")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(data_series, bins=30, stat="density", color="grey", edgecolor="black", alpha=0.5, ax=ax, label="Dados")
            x_smooth = np.linspace(data_series.min(), data_series.max(), 200)
            y_smooth = q_exponential_pdf(x_smooth, lam, q)
            ax.plot(x_smooth, y_smooth, 'r-', label="Curva q-exponencial")
            ax.set_title("Ajuste Q-Exponencial")
            ax.set_xlabel(col)
            ax.set_ylabel("Densidade")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro no ajuste Q-Exponencial: {e}")

# ============================
# FUNÇÃO PRINCIPAL DO APLICATIVO
# ============================
def main():
    st.title("Plataforma de Análises Estatísticas e Machine Learning")
    st.markdown("Esta aplicação integra diversas técnicas estatísticas e de clustering para análise robusta de dados.")
    
    menu = st.sidebar.selectbox("Selecione a Seção", 
        options=[
            "Dataset de Poços Artesianos", 
            "CSV Sintético", 
            "Estatísticas Descritivas", 
            "Intervalo de Confiança - Proporção", 
            "Intervalo de Confiança - Média", 
            "Teste de Normalidade", 
            "Testes Não-Paramétricos", 
            "Two-Way ANOVA", 
            "Regressão Linear", 
            "Teste de Hipótese", 
            "Testes de Correlação", 
            "Q-Estatística", 
            "Q-Exponencial", 
            "Clustering"
        ])
    
    if menu == "Dataset de Poços Artesianos":
        st.subheader("Dataset de Poços Artesianos")
        df_wells = create_well_dataset()
        st.dataframe(df_wells)
        csv_bytes = convert_df_to_csv(df_wells)
        st.download_button("Baixar CSV", data=csv_bytes, file_name="pocos_artesianos.csv")
        if st.checkbox("Exibir Gráfico Exemplo"):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(data=df_wells, x="Salinity_ppm", y="Flow_m3_per_h", hue="Geological_Formation", ax=ax)
            ax.set_title("Salinidade vs Vazão por Formação Geológica")
            st.pyplot(fig)
    
    elif menu == "CSV Sintético":
        synthetic_csv_section()
    
    elif menu == "Estatísticas Descritivas":
        estatistica_descritiva_section()
    
    elif menu == "Intervalo de Confiança - Proporção":
        intervalo_confianca_section(tipo="proporcao")
    
    elif menu == "Intervalo de Confiança - Média":
        intervalo_confianca_section(tipo="media")
    
    elif menu == "Teste de Normalidade":
        normalidade_section()
    
    elif menu == "Testes Não-Paramétricos":
        nao_parametricos_section()
    
    elif menu == "Two-Way ANOVA":
        anova_section()
    
    elif menu == "Regressão Linear":
        regressao_section()
    
    elif menu == "Teste de Hipótese":
        hipotese_section()
    
    elif menu == "Testes de Correlação":
        # Aqui utilizamos a seção de correlações inovadoras
        st.subheader("Testes de Correlação")
        correlacoes_section()
    
    elif menu == "Q-Estatística":
        q_estat_section()
    
    elif menu == "Q-Exponencial":
        q_exponencial_section()
    
    elif menu == "Clustering":
        clustering_section()

if __name__ == "__main__":
    main()