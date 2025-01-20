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
        )

        st.markdown("""
        **Interpretação**:
        - O PCA foi aplicado para reduzir a dimensionalidade dos dados, destacando as principais variações.
        - O gráfico de dispersão mostra como os dados estão distribuídos ao longo dos dois primeiros componentes principais.
        """)

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção"):
            return

    # O código continua de forma semelhante para cada seção
    # Cada seção terá o código específico para executar a análise correspondente e apresentar os resultados
    # Seguindo a estrutura de perguntar ao usuário se ele deseja continuar
    # Como a estrutura das seções é semelhante, a parte seguinte do código segue o mesmo padrão.
# =========================================================
# SEÇÃO 2: Cálculo de Amostragem - Proporção
# =========================================================
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
                    f"e margem de erro de {margem_erro}%, assumindo proporção verdadeira por volta de {p_est}, "
                    f"o tamanho de amostra recomendado é {resultado} para alcançar a precisão desejada."
                )
            else:
                st.error("Erro no cálculo. Verifique os parâmetros informados.")

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção"):
            return

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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção"):
            return

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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção"):
            return

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

        # Perguntar se o usuário quer continuar para a próxima seção
        if st.button("Continuar para a próxima seção"):
            return

# O processo continua para cada seção subsequente com o mesmo padrão:
# - Cada seção realiza a análise necessária.
# - Pergunta ao usuário se deseja continuar para a próxima.

# No final, a execução continua até o fim, permitindo ao usuário fazer todas as análises sequenciais.




if __name__ == "__main__":
    main()
