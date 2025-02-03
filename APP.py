import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Tenta importar o dcor; se não for possível, desativa essa funcionalidade
try:
    import dcor
    dcor_installed = True
except ImportError:
    dcor_installed = False

# ============================
# FUNÇÕES AUXILIARES E DATASETS
# ============================
def create_well_dataset():
    """Cria um dataset de Poços Artesianos (exemplo real)"""
    data = {
        "Well_ID": [f"Well_{i:03d}" for i in range(1, 11)],
        "Flow_m3_per_h": [120, 95, 150, 80, 110, 130, 105, 90, 115, 100],
        "Salinity_ppm": [350, 420, 290, 500, 375, 410, 330, 460, 360, 395],
        "pH": [7.2, 6.8, 7.4, 7.0, 7.1, 6.9, 7.3, 6.7, 7.0, 7.2],
        "Calcium_mg_per_L": [150, 180, 130, 160, 155, 170, 140, 165, 150, 158],
        "Magnesium_mg_per_L": [75, 65, 80, 70, 78, 82, 76, 69, 80, 77],
        "Sulfate_mg_per_L": [80, 100, 70, 90, 85, 95, 75, 88, 82, 89],
        "Chloride_mg_per_L": [120, 140, 110, 130, 125, 135, 115, 128, 122, 119],
        "Geological_Formation": ["Granito", "Argilito", "Calcário", "Arenito", "Granito",
                                 "Argilito", "Calcário", "Arenito", "Granito", "Argilito"],
        "Climate_Type": ["Temperado", "Árido", "Subtropical", "Continentais", "Temperado",
                         "Árido", "Subtropical", "Continentais", "Temperado", "Árido"],
        "Depth_m": [250, 300, 210, 275, 240, 320, 230, 290, 260, 310]
    }
    return pd.DataFrame(data)

def create_synthetic_data(n_rows=100):
    """Gera um dataset sintético com dados numéricos (inteiros e reais) e categóricos."""
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
# FUNÇÕES DE ANOVA
# ============================
def anova_unidirecional(df, dep_var, fator):
    """
    Realiza uma ANOVA unidirecional, comparando as médias de dois ou mais grupos.
    H0: as médias dos grupos são iguais.
    """
    st.markdown("**ANOVA Unidirecional:**")
    st.markdown("""
    Este teste envolve uma única variável independente e é usado para comparar as médias de dois ou mais grupos independentes.
    A hipótese nula é que não há diferença significativa entre as médias dos grupos.
    """)
    formula = f"{dep_var} ~ C({fator})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def anova_bidirecional(df, dep_var, fator1, fator2):
    """
    Realiza uma ANOVA bidirecional, avaliando o efeito de duas variáveis independentes e sua interação.
    """
    st.markdown("**ANOVA Bidirecional:**")
    st.markdown("""
    Este teste envolve duas variáveis independentes. Ele investiga o efeito simultâneo de duas variáveis categóricas
    sobre a variável dependente e também verifica se há interação entre esses fatores.
    """)
    formula = f"{dep_var} ~ C({fator1}) + C({fator2}) + C({fator1}):C({fator2})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def manova_analysis(df, dep_vars, fatores):
    """
    Realiza a MANOVA (Análise de Variância Multivariada) para avaliar o efeito de variáveis independentes
    sobre múltiplas variáveis dependentes.
    """
    st.markdown("**MANOVA (Análise de Variância Multivariada):**")
    st.markdown("""
    Este método é utilizado quando há múltiplas variáveis dependentes.
    O objetivo é determinar se as variáveis independentes têm um efeito significativo sobre as variáveis dependentes.
    """)
    dep_formula = " + ".join(dep_vars)
    fatores_formula = " + ".join([f"C({f})" for f in fatores])
    formula = f"{dep_formula} ~ {fatores_formula}"
    from statsmodels.multivariate.manova import MANOVA
    manova = MANOVA.from_formula(formula, data=df)
    result = manova.mv_test()
    return result

# ============================
# FUNÇÕES DE REGRESSÃO
# ============================
def regressao_linear(df, formula: str):
    model = ols(formula, data=df).fit()
    return model.summary().as_text()

# ============================
# FUNÇÕES DE TESTES T
# ============================
def teste_t_independente(df, grupo, valor):
    """
    Realiza o teste t de Student para duas amostras independentes.
    A hipótese nula é que as médias dos dois grupos são iguais.
    """
    grupos = df[grupo].unique()
    if len(grupos) != 2:
        return None, None
    grupo1 = df[df[grupo] == grupos[0]][valor].dropna()
    grupo2 = df[df[grupo] == grupos[1]][valor].dropna()
    return stats.ttest_ind(grupo1, grupo2)

def teste_t_pareado(df, coluna1, coluna2):
    """
    Realiza o teste t pareado, comparando as médias de duas condições emparelhadas.
    """
    paired1 = df[coluna1].dropna()
    paired2 = df[coluna2].dropna()
    min_len = min(len(paired1), len(paired2))
    paired1 = paired1.iloc[:min_len]
    paired2 = paired2.iloc[:min_len]
    return stats.ttest_rel(paired1, paired2)

# ============================
# FUNÇÕES DE Q-ESTATÍSTICA E Q-EXPONENCIAL
# ============================
def cochrane_q(effects, variances):
    w = 1.0 / np.array(variances)
    theta_fixed = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * (effects - theta_fixed)**2)
    df_anova = len(effects) - 1
    p_val = 1 - stats.chi2.cdf(Q, df_anova)
    return Q, p_val

def q_exponential_pdf(x, lam, q):
    return (2 - q) * lam * np.power((1 - (1-q)*lam*x), 1/(1-q))

def fit_q_exponential(data):
    data_clean = data[~np.isnan(data) & ~np.isinf(data)]
    counts, bins = np.histogram(data_clean, bins=30, density=True)
    xvals = 0.5*(bins[1:] + bins[:-1])
    yvals = counts
    initial_guess = [0.1, 1.2]
    popt, _ = stats.curve_fit(q_exponential_pdf, xvals, yvals, p0=initial_guess, maxfev=10000)
    return popt  # lam, q

# ============================
# FUNÇÕES DE CORRELAÇÃO (incluindo inovações)
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
        st.error("O módulo 'dcor' não está instalado. Instale-o via 'pip install dcor' para utilizar esta funcionalidade.")
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
    Nesta seção, você pode aplicar técnicas de clustering para segmentar seus dados.
    Escolha as variáveis de interesse, o algoritmo e visualize o gráfico resultante.
    """)
    file = st.file_uploader("Envie um arquivo CSV para clustering", type=["csv"], key="clustering")
    if file:
        df = pd.read_csv(file).dropna()
        # Seleciona as colunas numéricas disponíveis
        colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(colunas_num) < 1:
            st.error("É necessário ao menos uma variável numérica para realizar o clustering.")
            return
        # Permite que o usuário escolha uma ou mais variáveis para o clustering
        variaveis = st.multiselect("Selecione as variáveis para o clustering", colunas_num, default=colunas_num[:2])
        if len(variaveis) < 1:
            st.error("Selecione pelo menos uma variável.")
            return
        
        data = df[variaveis].values
        
        algoritmo = st.selectbox("Escolha o algoritmo de clustering", options=["KMeans", "Hierarchical", "DBSCAN"])
        
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
        st.markdown("**Resumo dos Clusters (médias das variáveis selecionadas):**")
        resumo = df.groupby("Cluster")[variaveis].mean().reset_index()
        st.dataframe(resumo)
        
        st.markdown("**Visualização dos Clusters**")
        # Se mais de duas variáveis foram selecionadas, utiliza PCA para reduzir a dimensão para 2D
        if len(variaveis) > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            df_plot = pd.DataFrame(data_2d, columns=["PC1", "PC2"])
            df_plot["Cluster"] = labels
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=df_plot, palette="viridis", ax=ax)
            ax.set_title(f"Clustering com {algoritmo} (reduzido por PCA)")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(x=variaveis[0], y=variaveis[1], hue="Cluster", data=df, palette="viridis", ax=ax)
            ax.set_title(f"Clustering com {algoritmo}")
            st.pyplot(fig)
        st.dataframe(df.head())

# ============================
# FUNÇÃO PARA GERAR CSV SINTÉTICO
# ============================
def synthetic_csv_section():
    st.subheader("Gerador de CSV Sintético para Teste")
    n_rows = st.number_input("Número de linhas", min_value=10, value=100, step=10)
    df_synth = create_synthetic_data(n_rows)
    st.markdown("**Pré-visualização do dataset sintético:**")
    st.dataframe(df_synth.head())
    csv_bytes = convert_df_to_csv(df_synth)
    st.download_button("Baixar CSV Sintético", data=csv_bytes, file_name="sintetico.csv")

# ============================
# FUNÇÕES DA CALCULADORA DE TAMANHO DE AMOSTRA
# ============================
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

def sample_size_calculator_section():
    st.subheader("Calculadora de Tamanho de Amostra")
    st.markdown("Esta ferramenta calcula o tamanho de amostra necessário para uma análise estatística robusta, seja para proporção ou para média.")
    tipo = st.radio("Selecione o tipo de cálculo", options=["Proporção", "Média"], key="tipo_amostra")
    populacao = st.number_input("Tamanho da População", min_value=1, value=1000)
    nivel_confianca = st.slider("Nível de Confiança (%)", min_value=80, max_value=99, value=95)
    if tipo == "Proporção":
        margem_erro = st.number_input("Margem de Erro (%)", min_value=0.1, value=5.0)
        p_obs = st.number_input("Proporção estimada (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        resultado = tamanho_amostral_proporcao(populacao, nivel_confianca, margem_erro, p_obs)
        if resultado is not None:
            st.write(f"O tamanho amostral recomendado para proporção é: {resultado}")
        else:
            st.error("Erro no cálculo do tamanho amostral para proporção.")
    else:
        margem_erro = st.number_input("Margem de Erro (valor absoluto)", min_value=0.1, value=5.0)
        desvio_padrao = st.number_input("Desvio-Padrão", min_value=0.1, value=10.0)
        resultado = tamanho_amostral_media(populacao, nivel_confianca, margem_erro, desvio_padrao)
        if resultado is not None:
            st.write(f"O tamanho amostral recomendado para média é: {resultado}")
        else:
            st.error("Erro no cálculo do tamanho amostral para média.")

# ============================
# SEÇÃO DE TESTES T
# ============================
def teste_t_section():
    st.subheader("Testes t")
    tipo_t = st.radio("Selecione o tipo de teste t", 
                      options=["One-Sample", "Duas Amostras Independentes", "Pareado"], key="tipo_t")
    file = st.file_uploader("Envie um CSV para teste t", type=["csv"], key="teste_t")
    if file:
        df = pd.read_csv(file).dropna()
        if tipo_t == "One-Sample":
            col = st.selectbox("Selecione a coluna numérica", df.select_dtypes(include=[np.number]).columns)
            media_hipot = st.number_input("Média hipotética", value=0.0)
            stat, p = stats.ttest_1samp(df[col], popmean=media_hipot)
            st.write(f"t = {stat:.4f}, p-valor = {p:.4f}")
        elif tipo_t == "Duas Amostras Independentes":
            grupo = st.selectbox("Selecione a coluna que define os grupos", df.select_dtypes(include=["object"]).columns)
            valor = st.selectbox("Selecione a coluna numérica para comparar", df.select_dtypes(include=[np.number]).columns)
            grupos = df[grupo].unique()
            if len(grupos) != 2:
                st.error("A coluna de grupos deve ter exatamente 2 grupos para este teste.")
            else:
                stat, p = stats.ttest_ind(df[df[grupo]==grupos[0]][valor].dropna(), df[df[grupo]==grupos[1]][valor].dropna())
                st.write(f"t = {stat:.4f}, p-valor = {p:.4f}")
        elif tipo_t == "Pareado":
            col1 = st.selectbox("Selecione a primeira coluna numérica", df.select_dtypes(include=[np.number]).columns, key="t_pareado1")
            col2 = st.selectbox("Selecione a segunda coluna numérica", df.select_dtypes(include=[np.number]).columns, key="t_pareado2")
            paired1 = df[col1].dropna()
            paired2 = df[col2].dropna()
            min_len = min(len(paired1), len(paired2))
            paired1 = paired1.iloc[:min_len]
            paired2 = paired2.iloc[:min_len]
            stat, p = stats.ttest_rel(paired1, paired2)
            st.write(f"t = {stat:.4f}, p-valor = {p:.4f}")

# ============================
# SEÇÃO DE Q-ESTATÍSTICA E Q-EXPONENCIAL
# ============================
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
    st.title("PhD Tool para Análise e Tratamento de Dados")
    st.markdown("Uma ferramenta robusta e flexível para análises estatísticas, modelagem e tratamento de dados. "
                "Inclui gerador de CSV sintético para teste, calculadora de tamanho de amostra, técnicas de clustering, ANOVA, testes t e muito mais.")
    
    menu = st.sidebar.selectbox("Selecione a Seção", 
        options=[
            "Calculadora de Tamanho de Amostra",
            "Dataset de Poços Artesianos", 
            "CSV Sintético", 
            "Estatísticas Descritivas", 
            "Intervalo de Confiança - Proporção", 
            "Intervalo de Confiança - Média", 
            "Teste de Normalidade", 
            "Testes Não-Paramétricos", 
            "ANOVA", 
            "Regressão Linear", 
            "Teste de Hipótese", 
            "Teste t", 
            "Testes de Correlação", 
            "Q-Estatística", 
            "Q-Exponencial", 
            "Clustering"
        ])
    
    if menu == "Calculadora de Tamanho de Amostra":
        sample_size_calculator_section()
    elif menu == "Dataset de Poços Artesianos":
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
    elif menu == "ANOVA":
        anova_section()
    elif menu == "Regressão Linear":
        regressao_section()
    elif menu == "Teste de Hipótese":
        hipotese_section()
    elif menu == "Teste t":
        teste_t_section()
    elif menu == "Testes de Correlação":
        st.subheader("Testes de Correlação")
        st.markdown("""
        Aqui você pode aplicar os testes tradicionais (Pearson, Spearman, Kendall) e técnicas inovadoras (Correlação de Distância e Correlação Parcial).
        """)
        st.markdown("### Opções de Tratamento de Dados")
        st.markdown("""
        **Escolha como tratar seus dados antes da análise:**
        - **Remover valores ausentes:** Exclui as linhas com dados faltantes.
        - **Substituir com a média:** Preenche os valores faltantes com a média de cada coluna.
        """)
        tratamento = st.radio("Selecione o método de tratamento:", 
                               options=["Remover valores ausentes", "Substituir com a média"], key="trat_corr")
        file_corr = st.file_uploader("Envie um CSV para testes de correlação", type=["csv"], key="corr_inov")
        if file_corr:
            df_corr = pd.read_csv(file_corr)
            if tratamento == "Remover valores ausentes":
                df_corr = df_corr.dropna()
            else:
                for col in df_corr.select_dtypes(include=[np.number]).columns:
                    df_corr[col].fillna(df_corr[col].mean(), inplace=True)
            st.markdown("**Pré-visualização dos dados tratados:**")
            st.dataframe(df_corr.head())
            colunas_num = [c for c in df_corr.columns if pd.api.types.is_numeric_dtype(df_corr[c])]
            if len(colunas_num) < 2:
                st.error("São necessárias ao menos duas variáveis numéricas.")
            else:
                x_var = st.selectbox("Escolha a variável X", colunas_num, key="x_corr")
                y_var = st.selectbox("Escolha a variável Y", colunas_num, key="y_corr")
                st.markdown("### Resultados dos Testes de Correlação")
                if st.button("Calcular Correlação de Pearson"):
                    corr, pval = correlacao_pearson(df_corr[x_var], df_corr[y_var])
                    st.write(f"**Pearson:** r = {corr:.4f}, p-valor = {pval:.4f}")
                    st.info("Interpretação: Um valor de r próximo de 1 (ou -1) indica forte relação linear; p < 0.05 indica significância.")
                if st.button("Calcular Correlação de Spearman"):
                    corr, pval = correlacao_spearman(df_corr[x_var], df_corr[y_var])
                    st.write(f"**Spearman:** r = {corr:.4f}, p-valor = {pval:.4f}")
                    st.info("Interpretação: Indicado para dados não normalmente distribuídos; valores altos indicam forte relação na ordem dos dados.")
                if st.button("Calcular Correlação de Kendall"):
                    corr, pval = correlacao_kendall(df_corr[x_var], df_corr[y_var])
                    st.write(f"**Kendall:** tau = {corr:.4f}, p-valor = {pval:.4f}")
                    st.info("Interpretação: Ideal para conjuntos pequenos ou com outliers; valores altos indicam boa concordância na ordem dos valores.")
                if st.button("Calcular Correlação de Distância"):
                    x_data = df_corr[x_var].to_numpy()
                    y_data = df_corr[y_var].to_numpy()
                    min_len = min(len(x_data), len(y_data))
                    x_data, y_data = x_data[:min_len], y_data[:min_len]
                    corr = correlacao_distancia(x_data, y_data)
                    if corr is not None:
                        st.write(f"**Distância:** correlação = {corr:.4f}")
                        st.info("Interpretação: Valores próximos de 0 indicam pouca relação; próximos de 1 indicam alta dependência, mesmo que não linear.")
                st.markdown("#### Correlação Parcial (Controlando uma terceira variável)")
                control_var = st.selectbox("Escolha a variável de controle", colunas_num, key="control_corr")
                if st.button("Calcular Correlação Parcial"):
                    common_index = df_corr[[x_var, y_var, control_var]].dropna().index
                    x_data = df_corr.loc[common_index, x_var]
                    y_data = df_corr.loc[common_index, y_var]
                    control_data = df_corr.loc[common_index, control_var]
                    r, pval = correlacao_parcial(x_data, y_data, control_data)
                    st.write(f"**Parcial:** r = {r:.4f}, p-valor = {pval:.4f}")
                    st.info("Interpretação: Se a relação entre X e Y se mantém alta após remover o efeito da variável de controle, indica uma conexão robusta.")
                if st.checkbox("Exibir gráfico de dispersão com tendência"):
                    fig, ax = plt.subplots(figsize=(6,4))
                    sns.scatterplot(data=df_corr, x=x_var, y=y_var, ax=ax)
                    sns.regplot(data=df_corr, x=x_var, y=y_var, scatter=False, ax=ax, color="red")
                    ax.set_title(f"Relação entre {x_var} e {y_var}")
                    st.pyplot(fig)
    elif menu == "Q-Estatística":
        q_estat_section()
    elif menu == "Q-Exponencial":
        q_exponencial_section()
    elif menu == "Clustering":
        clustering_section()

if __name__ == "__main__":
    main()