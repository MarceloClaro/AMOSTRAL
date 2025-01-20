import streamlit as st
import pandas as pd
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns  # Para gráficos com Seaborn

# ===================================================
# 1) Dataset de Poços Artesianos (código e funções)
# ===================================================
@st.cache_data
def convert_df_to_csv(dataframe: pd.DataFrame):
    return dataframe.to_csv(index=False).encode('utf-8')

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
        return "Erro: dados inválidos (NaN/Inf)."
    try:
        modelo = ols(formula, data=data_clean).fit()
        return modelo.summary().as_text()
    except Exception as e:
        return f"Erro na regressão linear: {e}"

def regressao_logistica(data: pd.DataFrame, formula: str):
    import statsmodels.formula.api as smf
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data_clean.empty:
        return "Erro: dados inválidos (NaN/Inf)."
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
    sns.set_style("whitegrid")  # Estilo Seaborn

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

    # SEÇÃO 1: Dataset de Poços Artesianos
    if menu == "Dataset de Poços Artesianos":
        st.subheader("Dataset de Poços Artesianos")
        st.dataframe(df_wells)
        csv_bytes = convert_df_to_csv(df_wells)
        st.download_button("Baixar Dataset como CSV", data=csv_bytes, file_name="pocos_artesianos.csv")
        if st.checkbox("Exibir Gráfico Seaborn (Exemplo)"):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(data=df_wells, x="Salinity_ppm", y="Flow_m3_per_h", hue="Geological_Formation", ax=ax)
            ax.set_title("Salinidade vs Vazão por Formação Geológica")
            ax.set_xlabel("Salinidade (ppm)")
            ax.set_ylabel("Vazão (m³/h)")
            st.pyplot(fig)

    # SEÇÃO 2: Cálculo de Amostragem - Proporção
    elif menu == "Cálculo de Amostragem - Proporção":
        st.subheader("Cálculo de Tamanho Amostral para Proporção")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95)
        margem_erro = st.slider("Margem de Erro (%)", 1, 50, 5)
        p_est = st.number_input("Proporção estimada", 0.0, 1.0, 0.5)
        if st.button("Calcular"):
            resultado = tamanho_amostral_proporcao(populacao, nivel_confianca, margem_erro, p_est)
            if resultado:
                st.success(f"Tamanho amostral recomendado: {resultado}")
                st.markdown("**Interpretação**: Calculado com base nos parâmetros fornecidos.")
            else:
                st.error("Erro no cálculo. Verifique os parâmetros.")

    # SEÇÃO 3: Cálculo de Amostragem - Média
    elif menu == "Cálculo de Amostragem - Média":
        st.subheader("Cálculo de Tamanho Amostral para Média")
        populacao = st.number_input("População (N)", min_value=1, value=1000)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95)
        margem_erro_abs = st.number_input("Margem de Erro (absoluto)", 0.1, 1000.0, 5.0)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", 0.1, 1000.0, 10.0)
        if st.button("Calcular"):
            resultado = tamanho_amostral_media(populacao, nivel_confianca, margem_erro_abs, desvio_padrao)
            if resultado:
                st.success(f"Tamanho amostral recomendado: {resultado}")
                st.markdown("**Interpretação**: Calculado com base nos parâmetros fornecidos.")
            else:
                st.error("Erro no cálculo.")

    # SEÇÃO 4: Intervalo de Confiança - Proporção
    elif menu == "Intervalo de Confiança - Proporção":
        st.subheader("IC para Proporção")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=100)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95)
        p_obs = st.number_input("Proporção Observada", 0.0, 1.0, 0.5)
        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_proporcao(n, confianca, p_obs)
            st.info(f"Intervalo: {ic[0]*100:.2f}% a {ic[1]*100:.2f}%")
            st.markdown("**Interpretação**: Intervalo de confiança calculado.")

    # SEÇÃO 5: Intervalo de Confiança - Média
    elif menu == "Intervalo de Confiança - Média":
        st.subheader("IC para Média")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=50)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95)
        media_amostral = st.number_input("Média Observada", value=50.0)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.1, value=10.0)
        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_media(n, confianca, media_amostral, desvio_padrao)
            st.info(f"Intervalo: {ic[0]:.2f} a {ic[1]:.2f}")
            st.markdown("**Interpretação**: Intervalo de confiança calculado.")

    # SEÇÃO 6: Estatísticas Descritivas
    elif menu == "Estatísticas Descritivas":
        st.subheader("Estatísticas Descritivas")
        file = st.file_uploader("Upload de CSV", type=["csv"], key="desc")
        if file:
            df = pd.read_csv(file)
            colunas_num = st.multiselect("Colunas numéricas", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            if colunas_num:
                stats_desc = estatisticas_descritivas(df[colunas_num].dropna())
                st.write(stats_desc)
                if st.checkbox("Exibir Gráficos Seaborn"):
                    for col in colunas_num:
                        fig, ax = plt.subplots(figsize=(6,4))
                        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="blue")
                        ax.set_title(f"Histograma de {col}")
                        st.pyplot(fig)

    # SEÇÃO 7: Testes de Normalidade
    elif menu == "Testes de Normalidade":
        st.subheader("Testes de Normalidade")
        file = st.file_uploader("Upload de CSV", type=["csv"], key="normal")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col = st.selectbox("Coluna numérica", colunas_num)
            if st.button("Shapiro-Wilk"):
                data_series = df[col].dropna()
                stat, p = teste_shapiro(data_series)
                st.write(f"Shapiro-Wilk: Estat={stat:.4f}, p={p:.4f}")
            if st.checkbox("Exibir Histograma e QQ-Plot"):
                data_series = df[col].dropna()
                fig, ax = plt.subplots()
                sns.histplot(data_series, kde=True, ax=ax, color="green")
                ax.set_title(f"Histograma de {col}")
                st.pyplot(fig)
                fig2 = plt.figure()
                sm.qqplot(data_series, line='s')
                plt.title(f"QQ-Plot de {col}")
                st.pyplot(fig2)
                plt.close(fig2)

    # SEÇÃO 8: Testes Não-Paramétricos
    elif menu == "Testes Não-Paramétricos":
        st.subheader("Testes Não-Paramétricos")
        file = st.file_uploader("Upload de CSV", type=["csv"], key="np")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            colunas_cat = [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])]
            col_num = st.selectbox("Coluna Numérica", colunas_num)
            col_cat = st.selectbox("Coluna Categórica", colunas_cat)

            if st.button("Executar Mann-Whitney"):
                stat, pval = teste_mannwhitney(df, col_num, col_cat)
                if stat is not None and pval is not None:
                    st.write(f"Mann-Whitney: Estat={stat:.4f}, p={pval:.4f}")
                else:
                    st.error("A coluna categórica não tem exatamente 2 grupos.")
                fig, ax = plt.subplots()
                sns.boxplot(data=df, x=col_cat, y=col_num, ax=ax)
                ax.set_title(f"Boxplot de {col_num} por {col_cat}")
                st.pyplot(fig)

            if st.button("Executar Kruskal-Wallis"):
                stat, pval = teste_kruskal(df, col_num, col_cat)
                st.write(f"Kruskal-Wallis: Estat={stat:.4f}, p={pval:.4f}")
                fig, ax = plt.subplots()
                sns.boxplot(data=df, x=col_cat, y=col_num, ax=ax, palette="viridis")
                ax.set_title(f"Boxplot de {col_num} por {col_cat}")
                st.pyplot(fig)

    # SEÇÃO 9: Two-Way ANOVA
    # SEÇÃO 9: Two-Way ANOVA
    elif menu == "Two-Way ANOVA":
        st.subheader("Two-Way ANOVA")
        st.markdown("""
            **Orientações para Two-Way ANOVA**:
            - Faça upload de um arquivo CSV contendo seus dados.
            - Selecione uma variável numérica dependente e duas variáveis categóricas como fatores.
            - A Two-Way ANOVA verifica se existem diferenças significativas na média da variável dependente entre 
              os grupos formados pelos fatores, incluindo também a interação entre eles.
            - O teste fornece uma tabela com estatísticas F e p-valores para cada fator e para a interação, 
              ajudando a identificar quais efeitos são estatisticamente significativos.
            - O nível de significância padrão é 5%. Isso significa que há uma chance de 5% de concluir 
              que um efeito existe quando, na verdade, ele não existe (erro tipo I). 
            - Você pode ajustar esse nível conforme sua necessidade para ser mais rigoroso ou mais flexível.
            - Certifique-se de que seus dados estejam limpos, sem valores ausentes (NaN) ou infinitos (Inf), 
              para obter resultados precisos.
        """)

        st.markdown("#### Fórmula Geral para Two-Way ANOVA:")
        st.latex(r"Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}")
        st.markdown(r"""
            Onde:
            - \(Y_{ijk}\) é a observação da variável dependente para o nível \(i\) do Fator 1, nível \(j\) do Fator 2 e réplica \(k\).
            - \(\mu\) é a média geral.
            - \(\alpha_i\) é o efeito do \(i\)-ésimo nível do Fator 1.
            - \(\beta_j\) é o efeito do \(j\)-ésimo nível do Fator 2.
            - \((\alpha\beta)_{ij}\) é o efeito de interação entre os níveis \(i\) e \(j\) dos dois fatores.
            - \(\epsilon_{ijk}\) é o erro aleatório ou residual.
        """)

        st.markdown("""
            **Por que usamos um nível de significância de 5%?**  
            O nível de significância é o limite que definimos para decidir se um resultado é estatisticamente significativo. 
            Um nível de 5% (0,05) é tradicionalmente usado porque oferece um bom equilíbrio entre 
            detectar efeitos reais e evitar falsos positivos. Em termos simples, ao usar 5%, 
            aceitamos uma chance de 5% de identificar um efeito que não existe de fato.
        """)

        st.markdown("""
            **Por que ajustar o nível de significância?**
            - Ao diminuir o nível de significância (por exemplo, de 5% para 1%), nos tornamos mais rigorosos. 
              Isso significa que exigimos mais evidências antes de considerar um efeito como significativo, 
              reduzindo a chance de falsos positivos, mas aumentando a chance de perder efeitos reais (falso negativo).
            - Ao aumentar o nível de significância (por exemplo, de 5% para 10%), estamos mais flexíveis.
              Isso facilita a detecção de efeitos, mas aumenta a chance de obter falsos positivos.
        """)

        # Ajuste do nível de significância
        significance_level = st.slider(
            "Nível de significância (%) para interpretação", 
            min_value=1, max_value=10, value=5
        ) / 100.0
        st.markdown(f"**Nível de significância selecionado: {significance_level*100:.0f}%**")

        file = st.file_uploader("Upload de CSV para Two-Way ANOVA", type=["csv"], key="anova2")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            st.write("Pré-visualização dos dados:")
            st.dataframe(df.head())

            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            colunas_cat = [c for c in df.columns if df[c].dtype == 'object' 
                           or pd.api.types.is_categorical_dtype(df[c])]

            if len(colunas_cat) < 2:
                st.warning("É necessário pelo menos duas variáveis categóricas para Two-Way ANOVA.")
            elif not colunas_num:
                st.warning("Não há colunas numéricas disponíveis para a variável dependente.")
            else:
                col_num = st.selectbox("Selecione a variável numérica dependente", colunas_num)
                cat1 = st.selectbox("Selecione o Fator 1", colunas_cat)
                cat2 = st.selectbox("Selecione o Fator 2", colunas_cat)

                if st.button("Executar Two-Way ANOVA"):
                    res = anova_two_way(df, col_num, cat1, cat2)
                    if res is not None:
                        st.dataframe(res)
                        st.markdown("### Interpretação Detalhada da Tabela ANOVA:")
                        st.markdown(f"""
                        A tabela ANOVA apresenta os resultados do teste. Cada linha representa uma fonte de variação:
                        - **C({cat1})**: efeito do primeiro fator.
                        - **C({cat2})**: efeito do segundo fator.
                        - **C({cat1}):C({cat2})**: interação entre os dois fatores.
                        - **Residual**: variabilidade não explicada pelo modelo.
                        
                        **Estatística F**: Compara a variabilidade entre grupos com a variabilidade dentro dos grupos. 
                        Valores maiores sugerem efeitos mais fortes.
                        
                        **p-valor**: Probabilidade de obter um valor de F tão extremo se não houver efeito real. 
                        Se o p-valor for menor que {significance_level*100:.0f}%, consideramos o efeito significativo.
                        """)
                        for index, row in res.iterrows():
                            st.markdown(f"**{index}**:")
                            st.markdown(f"- Estatística F: {row['F']:.3f}")
                            st.markdown(f"- p-valor: {row['PR(>F)']:.3f}")
                            if row['PR(>F)'] < significance_level:
                                st.markdown(f"- **Significativo**: O efeito de {index} em {col_num} é estatisticamente significativo.")
                                if index == f"C({cat1}):C({cat2})":
                                    st.markdown(f"- Sugere que a interação entre {cat1} e {cat2} altera significativamente a média de {col_num}.")
                                else:
                                    factor = index.split(']')[0].split('[')[-1]
                                    st.markdown(f"- Variações no nível de {factor} afetam significativamente {col_num}.")
                            else:
                                st.markdown(f"- **Não significativo**: Não há efeito estatístico significativo de {index} em {col_num} ao nível de {significance_level*100:.0f}%.")

                        st.markdown(f"""
                        ### Considerações Adicionais:
                        - Um p-valor menor que {significance_level*100:.0f}% para um fator sugere que existe diferença nas médias 
                          entre pelo menos alguns dos seus níveis.
                        - Um p-valor significativo para a interação indica que o efeito de um fator depende do nível do outro.
                        - Altos valores de F com p-valores baixos mostram que os fatores ou a interação explicam bem a variabilidade em {col_num}.
                        - As suposições de homogeneidade de variâncias e normalidade dos resíduos são importantes para a validade dos resultados.
                        """)
                        fig, ax = plt.subplots(figsize=(8,6))
                        sns.boxplot(data=df, x=cat1, y=col_num, hue=cat2, ax=ax)
                        ax.set_title(f"Distribuição de {col_num} por {cat1} e {cat2}")
                        ax.set_xlabel(cat1)
                        ax.set_ylabel(col_num)
                        st.pyplot(fig)
                        st.markdown(
                            "**Interpretação do Boxplot**: "
                            f"O boxplot mostra como {col_num} varia para cada combinação de {cat1} e {cat2}. "
                            "Observações de diferenças nas medianas e dispersões ajudam a visualizar efeitos significativos."
                        )
                    else:
                        st.error("A análise Two-Way ANOVA não pôde ser realizada. Verifique seus dados e seleções.")

    # SEÇÃO 10: Regressões com geração automática de fórmula
    # SEÇÃO 10: Regressões com geração automática de fórmula
    elif menu == "Regressões":
        st.subheader("Regressões")
        st.markdown(r"""
            **Orientações Gerais para Regressões**:
            ### Para Leigos de 9 Anos:
            - **O que é regressão?** Imagine que queremos prever algo com base em outras informações. 
              Por exemplo, prever a altura de uma planta dependendo da quantidade de água e sol. 
              A **regressão linear** tenta encontrar uma linha que melhor explique como uma coisa muda 
              com outra. A **regressão logística** é usada quando queremos prever algo que só pode ser "sim" ou "não" 
              (como aprovar ou reprovar um aluno).
            - **Como o aplicativo gera a fórmula?** Você escolhe o que quer prever e os fatores que 
              acha que influenciam. O aplicativo junta isso em uma equação automaticamente para você.

            ### Para PhDs:
            - Esta seção fornece uma interface interativa para especificação automática de fórmulas de regressão, 
              permitindo a incorporação de variáveis categóricas usando \(C()\).
            - Explicações metodológicas sobre interpretação de métricas como \(R^2\), p-valores e coeficientes 
              para regressão linear e logística são fornecidas.
        """)

        file = st.file_uploader("Upload de CSV para regressão", type=["csv"], key="reg")
        if file:
            df = pd.read_csv(file)
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

            st.markdown("**Selecione a variável dependente:**")
            dep_var = st.selectbox("Variável Dependente", numeric_cols)

            st.markdown("**Selecione as variáveis independentes:**")
            indep_vars = st.multiselect("Variáveis Independentes", df.columns.tolist())

            # Geração automática da fórmula
            terms = []
            for var in indep_vars:
                if var in categorical_cols:
                    terms.append(f"C({var})")
                else:
                    terms.append(var)
            if dep_var and terms:
                auto_formula = f"{dep_var} ~ " + " + ".join(terms)
                st.markdown("**Fórmula gerada automaticamente:**")
                st.code(auto_formula)
            else:
                st.markdown("Por favor, selecione uma variável dependente e variáveis independentes para gerar a fórmula.")

            tipo = st.selectbox("Tipo de Regressão", ["Linear", "Logística"])

            if st.button("Executar Regressão"):
                if not dep_var or not terms:
                    st.error("Variável dependente ou independentes não definidos. Certifique-se de selecionar as variáveis necessárias.")
                else:
                    if tipo == "Linear":
                        # [Código para regressão linear permanece igual...]
                        pass  
                    else:
                        # Bloco para Regressão Logística
                        unique_vals = df[dep_var].dropna().unique()
                        if not set(unique_vals).issubset({0,1}):
                            st.warning("Para regressão logística, a variável dependente deve ser binária (0 ou 1).")
                            st.markdown("""
                                **Escolha um método de binarização para a variável dependente:**
                                Se a variável dependente não for binária, podemos convertê-la automaticamente.
                            """)
                            conversion_method = st.selectbox(
                                "Escolha o método de binarização",
                                ["Mediana", "Média", "Percentil 75", "Percentil 25"]
                            )

                            if conversion_method == "Mediana":
                                threshold = df[dep_var].median()
                                st.markdown(f"**Usando a Mediana como limiar:** {threshold:.3f}")
                                st.latex(r"Y' = \begin{cases} 1 & \text{se } Y > \text{Mediana} \\ 0 & \text{caso contrário} \end{cases}")
                            elif conversion_method == "Média":
                                threshold = df[dep_var].mean()
                                st.markdown(f"**Usando a Média como limiar:** {threshold:.3f}")
                                st.latex(r"Y' = \begin{cases} 1 & \text{se } Y > \text{Média} \\ 0 & \text{caso contrário} \end{cases}")
                            elif conversion_method == "Percentil 75":
                                threshold = df[dep_var].quantile(0.75)
                                st.markdown(f"**Usando o Percentil 75 como limiar:** {threshold:.3f}")
                                st.latex(r"Y' = \begin{cases} 1 & \text{se } Y > \text{Percentil }75\% \\ 0 & \text{caso contrário} \end{cases}")
                            elif conversion_method == "Percentil 25":
                                threshold = df[dep_var].quantile(0.25)
                                st.markdown(f"**Usando o Percentil 25 como limiar:** {threshold:.3f}")
                                st.latex(r"Y' = \begin{cases} 1 & \text{se } Y > \text{Percentil }25\% \\ 0 & \text{caso contrário} \end{cases}")

                            df[dep_var] = (df[dep_var] > threshold).astype(int)
                            unique_vals = df[dep_var].unique()
                            st.markdown(f"Após conversão, os valores únicos da variável dependente são: {unique_vals}")
                            
                            st.markdown(f"**Distribuição de {dep_var} após conversão usando {conversion_method}:**")
                            fig, ax = plt.subplots()
                            sns.countplot(x=df[dep_var], ax=ax, palette="pastel")
                            ax.set_title(f"Contagem de valores binários em {dep_var}")
                            ax.set_xlabel(f"Valores de {dep_var} (0 ou 1)")
                            ax.set_ylabel("Frequência")
                            st.pyplot(fig)
                            
                            st.markdown("""
                                **Pronto para executar a regressão logística:**
                                - A variável dependente agora é binária.
                                - Pressione novamente o botão **Executar Regressão** para rodar a análise logística com a variável convertida.
                            """)
                        else:
                            try:
                                resultado = regressao_logistica(df, auto_formula)
                                st.text_area("Saída da Regressão Logística", resultado, height=300)
                                st.markdown(r"""
                                    **Interpretação da Regressão Logística**:
                                    - A regressão logística estima a probabilidade de ocorrência de um evento (valor 1).
                                    - A fórmula geral é:
                                """)
                                st.latex(r"\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots")
                                st.markdown(r"""
                                    onde \(p\) é a probabilidade de \(Y = 1\).
                                    - Os coeficientes (\(\beta_i\)) indicam como as variáveis independentes afetam o log-odds do evento.
                                    - Coeficientes com p-valores menores que 0.05 sugerem efeito significativo na probabilidade do evento.
                                    - Calcular os odds-ratios (\(\exp(\beta_i)\)) ajuda a entender o impacto prático de cada variável.
                                """)
                            except Exception as e:
                                st.error(f"Erro na regressão logística: {e}")

    # SEÇÃO 11: Teste de Hipótese
    elif menu == "Teste de Hipótese":
        st.subheader("Teste de Hipótese (One-Sample t-test)")
        file = st.file_uploader("Upload de CSV", type=["csv"], key="hipo")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col = st.selectbox("Coluna Numérica", colunas_num)
            media_hipot = st.number_input("Média Hipotética", 0.0, 9999.0, 0.0)
            if st.button("Executar t-test"):
                data_series = df[col].dropna()
                stat, pval = stats.ttest_1samp(data_series, popmean=media_hipot)
                st.write(f"T={stat:.4f}, p={pval:.4f}")
                fig, ax = plt.subplots()
                sns.histplot(data_series, kde=True, ax=ax, color="orange")
                ax.set_title(f"Histograma de {col}")
                st.pyplot(fig)

    # SEÇÃO 12: Testes de Correlação
    elif menu == "Testes de Correlação":
        st.subheader("Testes de Correlação (Pearson, Spearman, Kendall)")
        file = st.file_uploader("Upload de CSV", type=["csv"], key="corr")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(colunas_num) >= 2:
                x_var = st.selectbox("Variável X", colunas_num)
                y_var = st.selectbox("Variável Y", colunas_num)
                if st.button("Pearson"):
                    corr, pval = stats.pearsonr(df[x_var].dropna(), df[y_var].dropna())
                    st.write(f"Pearson: r={corr:.4f}, p={pval:.4f}")
                if st.button("Spearman"):
                    corr, pval = stats.spearmanr(df[x_var].dropna(), df[y_var].dropna())
                    st.write(f"Spearman: r={corr:.4f}, p={pval:.4f}")
                if st.button("Kendall"):
                    corr, pval = stats.kendalltau(df[x_var].dropna(), df[y_var].dropna())
                    st.write(f"Kendall: tau={corr:.4f}, p={pval:.4f}")
                if st.checkbox("Exibir Scatterplot Seaborn"):
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
                    ax.set_title(f"Dispersão entre {x_var} e {y_var}")
                    st.pyplot(fig)

    # SEÇÃO 13: Q-Estatística
    elif menu == "Q-Estatística":
        st.subheader("Cálculo de Q-Estatística (Cochrane’s Q)")
        file = st.file_uploader("Upload de CSV", type=["csv"], key="qstat")
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
                    st.error(f"Erro: {e}")

    # SEÇÃO 14: Q-Exponencial
    elif menu == "Q-Exponencial":
        st.subheader("Ajuste Q-Exponencial (Tsallis)")
        file = st.file_uploader("Upload de CSV", type=["csv"], key="qexp")
        if file:
            df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan)
            colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col = st.selectbox("Selecione a coluna numérica", colunas_num)
            if st.button("Ajustar Q-Exponencial"):
                data_series = df[col].dropna()
                if len(data_series) == 0:
                    st.error("A coluna selecionada não possui dados numéricos válidos.")
                else:
                    try:
                        lam_fit, q_fit = fit_q_exponential(data_series)
                        st.write(f"Parâmetros ajustados: λ = {lam_fit:.4f}, q = {q_fit:.4f}")
                        fig, ax = plt.subplots(figsize=(6,4))
                        sns.histplot(data_series, bins=30, stat="density", color="grey", edgecolor="black", alpha=0.5, label="Dados", ax=ax)
                        x_smooth = np.linspace(data_series.min(), data_series.max(), 200)
                        y_smooth = q_exponential_pdf(x_smooth, lam_fit, q_fit)
                        ax.plot(x_smooth, y_smooth, 'r-', label="Curva q-exponencial")
                        ax.set_title("Ajuste Q-Exponencial (Tsallis)")
                        ax.set_xlabel(col)
                        ax.set_ylabel("Densidade")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Falha ao ajustar Q-Exponencial: {e}")

if __name__ == "__main__":
    main()
