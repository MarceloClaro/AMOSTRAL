import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import dcor  # Certifique-se de ter instalado: pip install dcor

# ===================================================
# FUNÇÕES DE CORRELAÇÃO (TRADICIONAIS E INOVADORAS)
# ===================================================
def correlacao_pearson(x, y):
    """Mede a relação linear entre duas variáveis."""
    corr, pval = stats.pearsonr(x, y)
    return corr, pval

def correlacao_spearman(x, y):
    """Mede a relação entre duas variáveis usando os ranks (ordem dos valores)."""
    corr, pval = stats.spearmanr(x, y)
    return corr, pval

def correlacao_kendall(x, y):
    """Mede a concordância entre pares de dados (ordem)."""
    corr, pval = stats.kendalltau(x, y)
    return corr, pval

def correlacao_distancia(x, y):
    """
    Avalia a relação geral (linear ou não) entre as variáveis.
    Retorna um valor entre 0 (sem relação) e 1 (relação perfeita).
    """
    corr = dcor.distance_correlation(x, y)
    return corr

def correlacao_parcial(x, y, control):
    """
    Calcula a correlação entre x e y removendo o efeito de uma terceira variável.
    
    Parâmetros:
      - x, y: arrays com os dados das variáveis de interesse.
      - control: dados da variável a ser controlada.
    
    Retorna:
      - r: valor da correlação parcial.
      - pval: p-valor associado.
    """
    # Regressão de x em relação ao controle
    model_x = ols("x ~ control", data=pd.DataFrame({'x': x, 'control': control})).fit()
    res_x = model_x.resid

    # Regressão de y em relação ao controle
    model_y = ols("y ~ control", data=pd.DataFrame({'y': y, 'control': control})).fit()
    res_y = model_y.resid

    r, pval = stats.pearsonr(res_x, res_y)
    return r, pval

# ===================================================
# SEÇÃO: TESTES DE CORRELAÇÃO (COM TRATAMENTO DE DADOS)
# ===================================================
def correlacoes_section():
    st.subheader("Testes de Correlação Inovadores")
    st.markdown("""
    **Bem-vindo à seção de correlações!**  
    Aqui você pode utilizar os testes tradicionais (Pearson, Spearman e Kendall) e duas técnicas inovadoras:
    
    - **Correlação de Distância:** Detecta relações lineares e não-lineares.
    - **Correlação Parcial:** Exibe a relação entre duas variáveis, descontando o efeito de uma terceira variável.
    
    **O que estes testes informam?**
    - **Pearson:** Valores próximos de 1 ou -1 indicam forte relação linear.
    - **Spearman/Kendall:** Indicados quando os dados não são normalmente distribuídos; valores altos apontam forte relação de ordem.
    - **Distância:** Valores próximos de 0 sugerem pouca ou nenhuma relação; próximos de 1 indicam alta dependência, mesmo que não linear.
    - **Parcial:** Se, ao controlar outra variável, a relação entre X e Y se mantiver alta, isso mostra uma conexão robusta.
    """)

    # --------------------------------------------------
    # OPÇÕES DE TRATAMENTO DE DADOS
    # --------------------------------------------------
    st.markdown("### Opções de Tratamento de Dados")
    st.markdown("""
    **Antes de realizar os testes, escolha como tratar seus dados:**
    
    - **Remover valores ausentes:** Exclui as linhas com dados faltantes.
    - **Substituir com a média:** Preenche os dados faltantes com a média de cada coluna.
    """)
    tratamento = st.radio("Selecione o método de tratamento:", 
                           options=["Remover valores ausentes", "Substituir com a média"])

    file = st.file_uploader("Envie um arquivo CSV para análise", type=["csv"], key="corr_inov")
    if file:
        df = pd.read_csv(file)
        
        # Aplicação do tratamento de dados conforme a escolha do usuário
        if tratamento == "Remover valores ausentes":
            df = df.dropna()
        else:
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].mean(), inplace=True)
        
        st.markdown("**Pré-visualização dos dados tratados:**")
        st.dataframe(df.head())

        # Verifica se há pelo menos duas colunas numéricas
        colunas_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(colunas_num) < 2:
            st.error("São necessárias ao menos duas variáveis numéricas para a análise.")
            return
        
        # Seleção das variáveis para a análise
        x_var = st.selectbox("Escolha a variável X", colunas_num, key="x_corr")
        y_var = st.selectbox("Escolha a variável Y", colunas_num, key="y_corr")
        
        st.markdown("### Resultados dos Testes de Correlação")
        
        # 1. Correlação de Pearson
        if st.button("Calcular Correlação de Pearson"):
            corr, pval = correlacao_pearson(df[x_var], df[y_var])
            st.write(f"**Pearson:** r = {corr:.4f}, p-valor = {pval:.4f}")
            st.info("Interpretação: Um valor de r próximo de 1 (ou -1) indica uma forte relação linear. Se p < 0.05, o resultado é estatisticamente confiável.")
        
        # 2. Correlação de Spearman
        if st.button("Calcular Correlação de Spearman"):
            corr, pval = correlacao_spearman(df[x_var], df[y_var])
            st.write(f"**Spearman:** r = {corr:.4f}, p-valor = {pval:.4f}")
            st.info("Interpretação: Indicado para dados não normalmente distribuídos; valores altos mostram uma forte relação na ordem dos dados.")
        
        # 3. Correlação de Kendall
        if st.button("Calcular Correlação de Kendall"):
            corr, pval = correlacao_kendall(df[x_var], df[y_var])
            st.write(f"**Kendall:** tau = {corr:.4f}, p-valor = {pval:.4f}")
            st.info("Interpretação: Ideal para conjuntos de dados pequenos ou com outliers; valores altos indicam uma boa concordância na ordem dos valores.")
        
        # 4. Correlação de Distância
        if st.button("Calcular Correlação de Distância"):
            x_data = df[x_var].to_numpy()
            y_data = df[y_var].to_numpy()
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]
            corr = correlacao_distancia(x_data, y_data)
            st.write(f"**Distância:** correlação = {corr:.4f}")
            st.info("Interpretação: Valores próximos de 0 indicam pouca ou nenhuma relação; valores próximos de 1 indicam alta dependência, mesmo que a relação não seja linear.")
        
        # 5. Correlação Parcial (opcional)
        st.markdown("#### Correlação Parcial (Controlando uma terceira variável)")
        control_var = st.selectbox("Escolha a variável de controle", colunas_num, key="control_corr")
        if st.button("Calcular Correlação Parcial"):
            common_index = df[[x_var, y_var, control_var]].dropna().index
            x_data = df.loc[common_index, x_var]
            y_data = df.loc[common_index, y_var]
            control_data = df.loc[common_index, control_var]
            r, pval = correlacao_parcial(x_data, y_data, control_data)
            st.write(f"**Parcial:** r = {r:.4f}, p-valor = {pval:.4f}")
            st.info("Interpretação: Se a correlação entre X e Y se mantém alta após descontar o efeito da variável de controle, isso indica uma forte relação intrínseca entre elas.")
        
        # Exibe um gráfico de dispersão para complementar a análise
        if st.checkbox("Exibir gráfico de dispersão com tendência"):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
            sns.regplot(data=df, x=x_var, y=y_var, scatter=False, ax=ax, color="red")
            ax.set_title(f"Relação entre {x_var} e {y_var}")
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            st.pyplot(fig)

# ===================================================
# FUNÇÃO PRINCIPAL DO APLICATIVO
# ===================================================
def main():
    st.title("Aplicativo de Testes de Correlação")
    menu = st.sidebar.selectbox("Selecione a seção", ["Testes de Correlação"])
    
    if menu == "Testes de Correlação":
        correlacoes_section()

if __name__ == "__main__":
    main()