import streamlit as st
import pandas as pd
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import curve_fit

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
def q_exponential_pdf(x, lam, q):
    # Função PDF q-exponencial simplificada.
    return (2 - q) * lam * np.power((1 - (1-q)*lam*x), 1/(1-q))

def fit_q_exponential(data):
    initial_guess = [0.1, 1.2]
    counts, bins = np.histogram(data, bins=30, density=True)
    xvals = 0.5*(bins[1:] + bins[:-1])
    yvals = counts
    popt, _ = curve_fit(q_exponential_pdf, xvals, yvals, p0=initial_guess, maxfev=10000)
    return popt  # lam, q

# ---------------------------------------------------
# INTERFACE STREAMLIT
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

    # ---------------------------------------
    # HOME: Descrição detalhada
    # ---------------------------------------
    if menu == "Home":
        st.subheader("Bem-vindo(a)!")
        st.markdown("""
        Este aplicativo oferece diversas ferramentas estatísticas avançadas:
        
        **1. Cálculo de Amostragem - Proporção**  
        Estima o tamanho de amostra necessário para obter uma **proporção** (por ex. % de respondentes que aprovam algo) 
        com uma dada margem de erro e nível de confiança.

        **2. Cálculo de Amostragem - Média**  
        Estima o tamanho de amostra necessário para estimar uma **média** (por exemplo, tempo médio, rendimento médio etc.) 
        com precisão desejada.

        **3. Intervalo de Confiança - Proporção / Média**  
        Calcula intervalos de confiança para **proporções** e **médias**, com base na informação da amostra.  
        Útil para entender a faixa provável (dentro de um nível de confiança) em que se situa o valor populacional verdadeiro.

        **4. Estatísticas Descritivas**  
        Fornece estatísticas básicas (média, mediana, mínimos, máximos, quartis, etc.) para um conjunto de dados numéricos.

        **5. Testes de Normalidade**  
        Permite verificar se uma variável segue distribuição normal (testes de Shapiro-Wilk e Kolmogorov-Smirnov).

        **6. Testes Não-Paramétricos**  
        Inclui Mann-Whitney (comparação de 2 grupos) e Kruskal-Wallis (comparação de vários grupos) 
        sem assumir normalidade dos dados.

        **7. Two-Way ANOVA**  
        Analisa variações entre grupos considerando dois fatores diferentes ao mesmo tempo 
        (por exemplo, efeito de 2 variáveis categóricas em uma variável numérica).

        **8. Regressões (Linear e Logística)**  
        Permite criar modelos de regressão linear ou logística a partir de uma fórmula especificada pelo usuário.

        **9. Teste de Hipótese**  
        Realiza um teste t de uma amostra (one-sample t-test), verificando se a média amostral difere de um valor hipotético definido.

        **10. Testes de Correlação**  
        Avalia a correlação entre duas variáveis numéricas (Pearson, Spearman, Kendall).

        **11. Q-Estatística**  
        Calcula o Cochrane’s Q, usado geralmente em meta-análise para avaliar a heterogeneidade entre estudos.

        **12. Q-Exponencial**  
        Ajusta uma distribuição q-exponencial (estatística de Tsallis) a dados numéricos, estimando seus parâmetros.

        Selecione um item no menu à esquerda para começar.
        """)

    # ---------------------------------------
    # Cálculo de Amostragem - Proporção
    # ---------------------------------------
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
                st.markdown(
                    f"**Interpretação**: Para uma população de {populacao} indivíduos, com nível de confiança de {nivel_confianca}% "
                    f"e margem de erro de {margem_erro}%, assumindo que a proporção verdadeira gire em torno de {p_est}, "
                    f"o número de {resultado} respondentes na amostra é indicado para obter estimativas confiáveis da proporção."
                )
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados.")

    # ---------------------------------------
    # Cálculo de Amostragem - Média
    # ---------------------------------------
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
                st.markdown(
                    f"**Interpretação**: Para uma população de {populacao} indivíduos, com nível de confiança de {nivel_confianca}%, "
                    f"margem de erro de ±{margem_erro_abs} na média estimada e desvio-padrão populacional estimado em {desvio_padrao}, "
                    f"são necessários {resultado} respondentes para atingir a precisão desejada na estimativa da média."
                )
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados.")

    # ---------------------------------------
    # Intervalo de Confiança - Proporção
    # ---------------------------------------
    if menu == "Intervalo de Confiança - Proporção":
        st.subheader("Cálculo de Intervalo de Confiança para Proporção")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=100, step=1)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        p_obs = st.number_input("Proporção Observada (0.0 a 1.0)", 0.0, 1.0, 0.5, 0.01)

        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_proporcao(n, confianca, p_obs)
            st.info(f"Intervalo de Confiança Aproximado: {ic[0]*100:.2f}% a {ic[1]*100:.2f}%")
            st.markdown(
                f"**Interpretação**: Supondo que a proporção amostral seja {p_obs*100:.2f}%, com tamanho de amostra n={n} e "
                f"{confianca}% de confiança, o intervalo de {ic[0]*100:.2f}% a {ic[1]*100:.2f}% sugere a faixa provável onde está "
                f"a proporção populacional real."
            )

    # ---------------------------------------
    # Intervalo de Confiança - Média
    # ---------------------------------------
    if menu == "Intervalo de Confiança - Média":
        st.subheader("Cálculo de Intervalo de Confiança para Média")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=50, step=1)
        confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        media_amostral = st.number_input("Média Observada", value=50.0, step=0.1)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.001, value=10.0, step=0.1)

        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_media(n, confianca, media_amostral, desvio_padrao)
            st.info(f"Intervalo de Confiança Aproximado: {ic[0]:.2f} a {ic[1]:.2f}")
            st.markdown(
                f"**Interpretação**: Considerando uma amostra de n={n}, média observada de {media_amostral} e desvio-padrão "
                f"de {desvio_padrao}, com {confianca}% de confiança, a média populacional provavelmente está entre "
                f"{ic[0]:.2f} e {ic[1]:.2f}."
            )

    # ---------------------------------------
    # Estatísticas Descritivas
    # ---------------------------------------
    if menu == "Estatísticas Descritivas":
        st.subheader("Estatísticas Descritivas")
        file = st.file_uploader("Faça upload de um arquivo CSV", type=["csv"], key="desc")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            colunas_num = st.multiselect("Selecione colunas numéricas", df.columns.tolist(),
                                         default=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            if colunas_num:
                desc = estatisticas_descritivas(df[colunas_num])
                st.write(desc)
                st.markdown(
                    "**Interpretação**: Aqui, você vê estatísticas básicas como média (`mean`), desvio padrão (`std`), "
                    "valores mínimos (`min`), máximos (`max`) e quartis (`25%`, `50%`, `75%`). Esse resumo ajuda a "
                    "compreender rapidamente a distribuição e dispersão dos dados."
                )

    # ---------------------------------------
    # Testes de Normalidade
    # ---------------------------------------
    if menu == "Testes de Normalidade":
        st.subheader("Testes de Normalidade")
        file = st.file_uploader("Upload CSV para testes de normalidade", type=["csv"], key="normal")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            coluna = st.selectbox("Selecione a coluna numérica para teste de normalidade", 
                                  [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])

            if st.button("Executar Shapiro-Wilk"):
                stat, p = teste_shapiro(df[coluna])
                st.write(f"Shapiro-Wilk: Estatística={stat:.4f}, p-valor={p:.4f}")
                if p < 0.05:
                    st.warning("Resultado sugere que a distribuição não é normal ao nível de 5%.")
                else:
                    st.info("Não há evidência para rejeitar normalidade ao nível de 5%.")
                st.markdown(
                    "**Interpretação**: Se o p-valor < 0.05, rejeitamos a hipótese de que os dados sejam normais. "
                    "Se p-valor >= 0.05, não há evidências suficientes para rejeitar a normalidade."
                )

            if st.button("Executar Kolmogorov-Smirnov"):
                stat, p = teste_ks(df[coluna])
                st.write(f"K-S Test: Estatística={stat:.4f}, p-valor={p:.4f}")
                if p < 0.05:
                    st.warning("Resultado sugere que a distribuição não é normal ao nível de 5%.")
                else:
                    st.info("Não há evidência para rejeitar normalidade ao nível de 5%.")
                st.markdown(
                    "**Interpretação**: O teste K-S avalia a aderência dos dados a uma distribuição normal. "
                    "p-valor < 0.05 indica evidência contra a normalidade."
                )

    # ---------------------------------------
    # Testes Não-Paramétricos
    # ---------------------------------------
    if menu == "Testes Não-Paramétricos":
        st.subheader("Testes Não-Paramétricos")
        file = st.file_uploader("Upload CSV para testes não-paramétricos", type=["csv"], key="np")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            col_num = st.selectbox("Coluna Numérica", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            col_cat = st.selectbox("Coluna Categórica", [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])])

            if st.button("Executar Mann-Whitney"):
                stat, p = teste_mannwhitney(df, col_num, col_cat)
                if stat is not None:
                    st.write(f"Mann-Whitney: Estatística={stat:.4f}, p-valor={p:.4f}")
                    if p < 0.05:
                        st.success("Diferença significativa entre os dois grupos ao nível de 5%.")
                    else:
                        st.info("Não há diferença significativa entre os grupos ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: O teste Mann-Whitney verifica se duas amostras independentes "
                        "têm distribuições com medianas diferentes. p-valor < 0.05 indica diferença significativa."
                    )
                else:
                    st.error("Mann-Whitney requer exatamente 2 grupos na coluna categórica.")

            if st.button("Executar Kruskal-Wallis"):
                stat, p = teste_kruskal(df, col_num, col_cat)
                st.write(f"Kruskal-Wallis: Estatística={stat:.4f}, p-valor={p:.4f}")
                if p < 0.05:
                    st.success("Diferença significativa entre pelo menos um dos grupos ao nível de 5%.")
                else:
                    st.info("Não há diferença significativa entre os grupos ao nível de 5%.")
                st.markdown(
                    "**Interpretação**: O teste Kruskal-Wallis compara múltiplos grupos não-paramétricos. "
                    "p-valor < 0.05 indica que ao menos um grupo difere significativamente dos outros."
                )

    # ---------------------------------------
    # Two-Way ANOVA
    # ---------------------------------------
    if menu == "Two-Way ANOVA":
        st.subheader("Two-Way ANOVA")
        file = st.file_uploader("Upload CSV para Two-Way ANOVA", type=["csv"], key="anova2")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            col_num = st.selectbox("Coluna Numérica", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            cat1 = st.selectbox("Fator 1", [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])])
            cat2 = st.selectbox("Fator 2", [c for c in df.columns if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])])

            if st.button("Executar Two-Way ANOVA"):
                anova_table = anova_two_way(df, col_num, cat1, cat2)
                if anova_table is not None:
                    st.write(anova_table)
                    st.markdown(
                        "**Interpretação**: A tabela ANOVA mostra o efeito de cada fator e da interação entre eles. "
                        "Observe os p-valores para verificar se cada fator ou a interação afeta significativamente "
                        f"a variável '{col_num}'."
                    )

    # ---------------------------------------
    # Regressões
    # ---------------------------------------
    if menu == "Regressões":
        st.subheader("Regressões")
        file = st.file_uploader("Upload CSV para regressões", type=["csv"], key="reg")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
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
                    st.markdown(
                        "**Interpretação**: Nos resultados, observe coeficientes, p-valores e intervalos de confiança. "
                        "Na regressão linear, o R² e o ajuste do modelo são importantes; "
                        "na regressão logística, observe se os coeficientes (odds ratios) e seus p-valores "
                        "indicam relações significativas."
                    )

    # ---------------------------------------
    # Teste de Hipótese (One-Sample t-test)
    # ---------------------------------------
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
                    st.markdown(
                        "**Interpretação**: Os dados sugerem que a média amostral difere significativamente da média hipotética. "
                        "p-valor < 0.05 indica evidências para concluir que a média populacional não é igual a H0."
                    )
                else:
                    st.info("Não rejeitamos H0 ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: Não há evidências suficientes para concluir que a média difira "
                        "da média hipotética informada."
                    )

    # ---------------------------------------
    # Testes de Correlação
    # ---------------------------------------
    if menu == "Testes de Correlação":
        st.subheader("Testes de Correlação (Pearson, Spearman, Kendall)")
        file = st.file_uploader("Upload CSV para correlação", type=["csv"], key="corr")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
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
                    st.markdown(
                        "**Interpretação (Pearson)**: Correlação linear, varia de -1 a +1. "
                        "p-valor < 0.05 indica correlação linear significativa."
                    )

                if st.button("Executar Spearman"):
                    corr, p_val = stats.spearmanr(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Spearman**: {corr:.4f}, p-valor={p_val:.4f}")
                    st.markdown(
                        "**Interpretação (Spearman)**: Mede correlação baseada em ranques, útil se os dados não forem normalmente distribuídos. "
                        "p-valor < 0.05 indica correlação monotônica significativa."
                    )

                if st.button("Executar Kendall"):
                    corr, p_val = stats.kendalltau(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Kendall**: {corr:.4f}, p-valor={p_val:.4f}")
                    st.markdown(
                        "**Interpretação (Kendall)**: Também baseada em ranques, mas com uma abordagem diferente de Spearman. "
                        "p-valor < 0.05 indica correlação significativa."
                    )

    # ---------------------------------------
    # Q-Estatística (Cochrane's Q)
    # ---------------------------------------
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
                    if p_val < 0.05:
                        st.warning("Há heterogeneidade significativa entre os estudos.")
                    else:
                        st.info("Não há evidências de heterogeneidade significativa.")
                    st.markdown(
                        "**Interpretação**: O Cochrane’s Q avalia se há heterogeneidade entre estudos em uma meta-análise. "
                        "Se p-valor < 0.05, os estudos podem ser heterogêneos; caso contrário, são considerados homogêneos."
                    )
                except Exception as e:
                    st.error(f"Erro: {e}. Verifique as colunas 'effect' e 'variance' no CSV.")

    # ---------------------------------------
    # Q-Exponencial
    # ---------------------------------------
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
                    st.write(f"**Parâmetros ajustados**: λ = {lam_fit:.4f}, q = {q_fit:.4f}")
                    st.markdown(
                        "**Interpretação**: O modelo q-exponencial (estatística de Tsallis) "
                        "é uma generalização da distribuição exponencial. "
                        "O parâmetro q reflete o grau de não-extensividade do sistema. "
                        "Valores de q próximos de 1 se aproximam de uma exponencial simples."
                    )
                except Exception as e:
                    st.error(f"Falha ao ajustar Q-Exponencial: {e}")

if __name__ == "__main__":
    main()
