import streamlit as st
import math

def obter_z(conf):
    """
    Retorna o valor de Z aproximado com base em faixas de nível de confiança (em %).
    Alguns valores padrão:
      - 80%  ~ 1.28
      - 85%  ~ 1.44
      - 90%  ~ 1.64
      - 95%  ~ 1.96
      - 99%  ~ 2.58
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
        # Caso o usuário selecione 100% ou mais,
        # assumiremos 2.58 (aprox. 99%+).
        return 2.58

def calcular_tamanho_amostral(populacao, nivel_confianca, margem_erro):
    """
    Calcula o tamanho amostral usando:
      n0 = (Z^2 * p * (1-p)) / e^2
      n = n0 / (1 + (n0 - 1)/N)  (com correção para população finita)
    Aqui, p = 0.5 (max. variância), e = margem_erro/100.
    """
    # Obtém valor de Z com base no nível de confiança
    Z = obter_z(nivel_confianca)
    p = 0.5
    e = margem_erro / 100.0  # Converter porcentagem para valor decimal

    # Evitar divisão por zero ou margens de erro muito pequenas
    if e == 0:
        st.warning("Margem de erro não pode ser 0%. Ajuste para um valor > 0.")
        return None

    # Fórmula para tamanho amostral sem correção (n0)
    n0 = (Z**2) * p * (1 - p) / (e**2)

    # Aplicando correção de população finita
    n_ajustado = n0 / (1 + (n0 - 1) / populacao)

    # Arredonda para inteiro
    n_ajustado = math.ceil(n_ajustado)

    return n_ajustado

def main():
    st.title("Calculadora de Tamanho Amostral")

    st.write("""
    **Descrição**: Esta calculadora ajuda a determinar o tamanho de uma amostra para sua pesquisa com base na 
    população total, no nível de confiança e na margem de erro selecionados.
    """)

    # SIDEBAR
    st.sidebar.title("Configurações")

    # 1. Tamanho da População
    populacao = st.sidebar.number_input(
        "Tamanho da População",
        min_value=1,
        value=1000,
        step=1,
        help="Informe o total de indivíduos/elementos em sua população."
    )

    # 2. Nível de Confiança (%)
    nivel_confianca = st.sidebar.slider(
        "Nível de Confiança (%)",
        min_value=0,
        max_value=100,
        value=95,
        step=1,
        help="Selecione o nível de confiança desejado (0% a 100%)."
    )

    # 3. Margem de Erro (%)
    margem_erro = st.sidebar.slider(
        "Margem de Erro (%)",
        min_value=0,
        max_value=80,
        value=5,
        step=1,
        help="Selecione a margem de erro (0% a 80%)."
    )

    # Botão para calcular
    if st.sidebar.button("Calcular Tamanho Amostral"):
        tamanho = calcular_tamanho_amostral(populacao, nivel_confianca, margem_erro)
        if tamanho is not None:
            st.success(f"Tamanho Amostral Recomendado: **{tamanho}** respondentes.")
        else:
            st.error("Não foi possível calcular o tamanho da amostra. Verifique os valores fornecidos.")

if __name__ == "__main__":
    main()
