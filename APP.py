import streamlit as st
import pandas as pd
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Função PCA para redução de dimensionalidade
def pca_reducao_dados(df):
    """
    Realiza o PCA para reduzir a dimensionalidade dos dados.
    O PCA é aplicado apenas em variáveis numéricas.
    """
    # Seleciona as colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    df_numerico = df[colunas_numericas].dropna()
    
    # Aplica PCA
    pca = PCA(n_components=min(df_numerico.shape[1], 2))  # Limitar para 2 componentes principais
    pca_resultado = pca.fit_transform(df_numerico)
    
    # Retorna os dados transformados
    pca_df = pd.DataFrame(data=pca_resultado, columns=['PC1', 'PC2'])
    return pca_df

# Função para exibição de gráficos PCA
def exibir_grafico_pca(pca_df):
    """
    Exibe um gráfico de dispersão dos componentes principais.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', edgecolor='black')
    plt.title("Gráfico de Dispersão dos Componentes Principais (PCA)", fontsize=14)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    st.pyplot(plt)

# Função para salvar DataFrame como CSV
@st.cache_data
def convert_df_to_csv(dataframe: pd.DataFrame):
    """
    Converte um DataFrame para CSV em bytes (UTF-8), 
    permitindo o uso no st.download_button.
    """
    return dataframe.to_csv(index=False).encode('utf-8')

# Função para criar o dataset fictício de poços artesianos
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

# Função principal para executar o Streamlit
def main():
    st.title("Ferramenta Avançada de Estatística e Cálculo Amostral")

    # Criação do dataset de poços artesianos para exibição e download
    df_wells = create_well_dataset()

    # Usando uma chave única para o menu
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
    ], key="menu_radio_1")  # Chave exclusiva para o radio

    # SEÇÃO 1: Dataset de Poços Artesianos
    if menu == "Dataset de Poços Artesianos":
        st.subheader("Dataset de Poços Artesianos")
        st.write("Este é um dataset fictício de poços artesianos com informações sobre vazão, salinidade, composição química, clima local e profundidade.")

        # Exibir o DataFrame
        st.write("Pré-visualização dos dados:")
        st.dataframe(df_wells)

        # Aplicar PCA para redução de dimensionalidade
        pca_df = pca_reducao_dados(df_wells)
        exibir_grafico_pca(pca_df)

        # Botão para download do arquivo CSV
        csv_bytes = convert_df_to_csv(df_wells)
        st.download_button(
            label="Baixar Dataset como CSV",
            data=csv_bytes,
            file_name="pocos_artesianos.csv",
            mime="text/csv",
            key="download_button_1"  # Chave única para o botão de download
        )

        st.markdown("""
        **Interpretação**:
        - O PCA foi aplicado para reduzir a dimensionalidade dos dados, destacando as principais variações.
        - O gráfico de dispersão mostra como os dados estão distribuídos ao longo dos dois primeiros componentes principais.
        """)

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_secao_1"):
            return

    # SEÇÃO 2: Cálculo de Amostragem - Proporção
    if menu == "Cálculo de Amostragem - Proporção":
        st.subheader("Cálculo de Tamanho Amostral para Proporção")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1, key="populacao_input_1")
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1, key="nivel_confianca_slider_1")
        margem_erro = st.slider("Margem de Erro (%)", 1, 50, 5, 1, key="margem_erro_slider_1")
        p_est = st.number_input("Proporção estimada (0.0 a 1.0)", 0.0, 1.0, 0.5, 0.01, key="proporcao_input_1")

        if st.button("Calcular", key="calcular_button_2"):
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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_secao_2"):
            return

# SEÇÃO 3: Cálculo de Amostragem - Média
    if menu == "Cálculo de Amostragem - Média":
        st.subheader("Cálculo de Tamanho Amostral para Média")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        margem_erro_abs = st.number_input("Margem de Erro (valor absoluto)", min_value=0.001, value=5.0, step=0.1)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.001, value=10.0, step=0.1)

        if st.button("Calcular", key="calcular_media"):
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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_3"):
            return

# SEÇÃO 4: Intervalo de Confiança - Proporção
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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_4"):
            return

# SEÇÃO 5: Intervalo de Confiança - Média
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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_5"):
            return

# SEÇÃO 6: Estatísticas Descritivas
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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_6"):
            return

# SEÇÃO 7: Testes de Normalidade
    elif menu == "Testes de Normalidade":
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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_7"):
            return

# SEÇÃO 8: Testes Não-Paramétricos
    elif menu == "Testes Não-Paramétricos":
        st.subheader("Testes Não-Paramétricos")
        file = st.file_uploader("Upload CSV para testes não-paramétricos", type=["csv"], key="np")
        if file:
            df = pd.read_csv(file)
            st.write("Exemplo de dados:")
            st.dataframe(df.head())
            col_num = st.selectbox("Coluna Numérica", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
            col_cat = st.selectbox("Coluna Categórica", [c for c in df.columns 
                                                         if df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c])])

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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_8"):
            return

# SEÇÃO 9: Two-Way ANOVA
    elif menu == "Two-Way ANOVA":
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
                        "**Interpretação**: Cada linha representa o efeito de um fator ou da interação entre fatores. "
                        "Verifique os p-valores para saber se há efeitos significativos na variável numérica."
                    )

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_9"):
            return

# SEÇÃO 10: Regressões
    elif menu == "Regressões":
        st.subheader("Regressões")
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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_10"):
            return

# SEÇÃO 11: Teste de Hipótese (One-Sample t-test)
    elif menu == "Teste de Hipótese":
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
                        "**Interpretação**: O p-valor < 0.05 sugere que a média amostral difere significativamente "
                        "da média hipotética definida em H0."
                    )
                else:
                    st.info("Não rejeitamos H0 ao nível de 5%.")
                    st.markdown(
                        "**Interpretação**: Não há evidências para afirmar que a média amostral seja diferente "
                        "da média hipotética (H0)."
                    )

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_11"):
            return

# SEÇÃO 12: Testes de Correlação
    elif menu == "Testes de Correlação":
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
                        "**Interpretação (Pearson)**: Mede correlação linear. p-valor < 0.05 indica relação linear significativa."
                    )

                if st.button("Executar Spearman"):
                    corr, p_val = stats.spearmanr(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Spearman**: {corr:.4f}, p-valor={p_val:.4f}")
                    st.markdown(
                        "**Interpretação (Spearman)**: Mede correlação baseada em ranques. "
                        "p-valor < 0.05 indica correlação significativa."
                    )

                if st.button("Executar Kendall"):
                    corr, p_val = stats.kendalltau(df[col_x].dropna(), df[col_y].dropna())
                    st.write(f"**Correlação de Kendall**: {corr:.4f}, p-valor={p_val:.4f}")
                    st.markdown(
                        "**Interpretação (Kendall)**: Também ranque, mas abordagem diferente de Spearman. p-valor < 0.05 indica correlação significativa."
                    )

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_12"):
            return

# SEÇÃO 13: Q-Estatística (Cochrane's Q)
    elif menu == "Q-Estatística":
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
                        "**Interpretação**: p-valor < 0.05 indica que os estudos não são homogêneos, ou seja, "
                        "há diferença entre eles que pode impactar a meta-análise."
                    )
                except Exception as e:
                    st.error(f"Erro: {e}. Verifique se as colunas 'effect' e 'variance' existem no CSV.")

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção", key="continuar_13"):
            return

# SEÇÃO 14: Q-Exponencial
    elif menu == "Q-Exponencial":
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
                        "**Interpretação**: A distribuição q-exponencial é uma generalização da exponencial. "
                        "Valores de q próximos de 1 indicam comportamento semelhante à exponencial simples; "
                        "valores de q diferentes de 1 sugerem sistemas não-extensivos (Tsallis)."
                    )
                except Exception as e:
                    st.error(f"Falha ao ajustar Q-Exponencial: {e}")

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Finalizar Análise", key="finalizar_14"):
            return

if __name__ == "__main__":
    main()

    main()


if __name__ == "__main__":
    main()
