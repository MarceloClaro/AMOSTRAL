import streamlit as st
import math

# ----------------------------------------------
# FUNÇÕES DE APOIO
# ----------------------------------------------

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


def tamanho_amostral_proporcao(populacao, nivel_confianca, margem_erro, p=None):
    """
    Calcula o tamanho amostral para estimar uma proporção.
    
    Fórmula utilizada:
        n0 = (Z^2 * p * (1-p)) / e^2
        n = n0 / (1 + (n0 - 1)/N)  (com correção de população finita)
    
    - populacao: tamanho da população (N).
    - nivel_confianca: nível de confiança em %.
    - margem_erro: margem de erro em % (ex.: 5 -> 0.05).
    - p: proporção estimada. Caso None, assume p=0.5 (máxima variância).
    """
    if p is None:
        p = 0.5  # Caso o usuário não informe, assume p=0.5
    
    Z = obter_z(nivel_confianca)
    e = margem_erro / 100.0  # Converter porcentagem em decimal

    if e == 0:
        return None  # margem de erro não pode ser zero

    # Cálculo inicial (sem correção)
    n0 = (Z**2 * p * (1 - p)) / (e**2)

    # Correção de população finita
    n_ajustado = n0 / (1 + (n0 - 1) / populacao)

    # Arredonda para inteiro
    return math.ceil(n_ajustado)


def intervalo_confianca_proporcao(n, confianca, p_observado):
    """
    Retorna o intervalo de confiança para uma proporção observada, 
    dado tamanho amostral (n), nível de confiança (confianca, em %) e p_observado (em decimal).
    
    Intervalo aproximado (z * erro padrão):
        IC = p ± Z * sqrt( p*(1-p)/n )
    """
    Z = obter_z(confianca)
    erro_padrao = math.sqrt(p_observado * (1 - p_observado) / n)
    margem = Z * erro_padrao
    return (p_observado - margem, p_observado + margem)


def tamanho_amostral_media(populacao, nivel_confianca, margem_erro, desvio_padrao):
    """
    Calcula tamanho amostral para estimar média (supondo desvio-padrão populacional conhecido),
    usando distribuição Z. Fórmula:
    
        n0 = (Z * σ / e)^2
        n = n0 / [1 + (n0 - 1)/N]  (correção de população finita)
    
    - populacao: tamanho da população (N)
    - nivel_confianca: nível de confiança em %
    - margem_erro: margem de erro (valor absoluto desejado, não em %; 
      ex.: se o erro permitido for ±2 unidades na média, margin_erro=2)
    - desvio_padrao: desvio-padrão populacional (σ).
    """
    Z = obter_z(nivel_confianca)

    if margem_erro <= 0:
        return None  # margem de erro não pode ser zero ou negativa

    # Cálculo inicial (sem correção)
    n0 = (Z * desvio_padrao / margem_erro)**2

    # Correção de população finita
    n_ajustado = n0 / (1 + (n0 - 1) / populacao)

    return math.ceil(n_ajustado)


def intervalo_confianca_media(n, confianca, media_amostral, desvio_padrao):
    """
    Calcula intervalo de confiança para a média, supondo 
    desvio-padrão populacional conhecido ou n grande (z-interval).
    
    Intervalo:
        media ± Z * (σ / sqrt(n))
    
    - n: tamanho amostral.
    - confianca: nível de confiança em %.
    - media_amostral: valor da média observada na amostra.
    - desvio_padrao: desvio-padrão populacional (ou estimado).
    """
    Z = obter_z(confianca)
    erro_padrao = desvio_padrao / math.sqrt(n)
    margem = Z * erro_padrao
    return (media_amostral - margem, media_amostral + margem)


# ----------------------------------------------
# INTERFACE STREAMLIT
# ----------------------------------------------

def main():
    st.title("Calculadora Avançada de Tamanho Amostral")

    st.markdown("""
    Esta aplicação permite **estimar o tamanho de amostra** necessário para:
    1. **Proporções (pesquisas de opinião, incidência, etc.)**.
    2. **Média (quando há desvio-padrão populacional conhecido ou amostra grande)**.
    
    Além disso, você pode **calcular intervalos de confiança** para a proporção ou média 
    após ter coletado dados. 
    """)

    # Seções do aplicativo
    menu = st.sidebar.selectbox(
        "Selecione a Seção",
        ["Cálculo de Amostragem - Proporção", 
         "Cálculo de Amostragem - Média",
         "Intervalo de Confiança - Proporção",
         "Intervalo de Confiança - Média"]
    )

    st.sidebar.markdown("---")

    if menu == "Cálculo de Amostragem - Proporção":
        st.subheader("Cálculo de Tamanho Amostral (Proporção)")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        margem_erro = st.slider("Margem de Erro (%)", 1, 50, 5, 1)
        
        st.markdown("""
        **p (proporção estimada)**:
        Se não tiver ideia, utilize `0.5` para a máxima variabilidade 
        (gerando o maior tamanho amostral). Se você tem alguma estimativa 
        (ex.: incidência, prevalência anterior, etc.), use-a para refinar o cálculo.
        """)
        p_est = st.number_input("Proporção estimada (decimal) [0.0 a 1.0]", min_value=0.0, max_value=1.0, value=0.5)
        
        if st.button("Calcular"):
            n_amostra = tamanho_amostral_proporcao(
                populacao=populacao,
                nivel_confianca=nivel_confianca,
                margem_erro=margem_erro,
                p=p_est
            )
            if n_amostra:
                st.success(f"Tamanho de amostra recomendado: **{n_amostra}** respondentes.")
            else:
                st.error("Não foi possível calcular. Verifique se a margem de erro não está zero.")

    elif menu == "Cálculo de Amostragem - Média":
        st.subheader("Cálculo de Tamanho Amostral (Média)")
        populacao = st.number_input("Tamanho da População (N)", min_value=1, value=1000, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        st.markdown("""
        A margem de erro aqui deve ser **a variação absoluta máxima** que você 
        aceita na média. Por exemplo, se você quer estimar a média com precisão 
        de ±2 pontos, coloque `2`.
        """)
        margem_erro = st.number_input("Margem de Erro (valor absoluto)", min_value=0.001, value=5.0, step=0.1)
        st.markdown("""
        O desvio-padrão (σ) pode ser **populacional** (se conhecido) 
        ou uma **estimativa** baseada em estudos anteriores.
        """)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", min_value=0.001, value=10.0, step=0.1)

        if st.button("Calcular"):
            n_amostra = tamanho_amostral_media(
                populacao=populacao,
                nivel_confianca=nivel_confianca,
                margem_erro=margem_erro,
                desvio_padrao=desvio_padrao
            )
            if n_amostra:
                st.success(f"Tamanho de amostra recomendado: **{n_amostra}** indivíduos.")
            else:
                st.error("Não foi possível calcular. Verifique se a margem de erro não está zero ou negativa.")

    elif menu == "Intervalo de Confiança - Proporção":
        st.subheader("Intervalo de Confiança para Proporção")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=100, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        st.markdown("""
        **p Observado** (proporção observada na amostra). Exemplo: 
        se 45 de 100 pessoas responderam "Sim", p_observado = 0.45.
        """)
        p_obs = st.number_input("Proporção Observada (decimal) [0.0 a 1.0]", min_value=0.0, max_value=1.0, value=0.5)

        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_proporcao(n, nivel_confianca, p_obs)
            st.write(f"Intervalo de Confiança (aprox.):")
            st.info(f"{ic[0]*100:.2f}% a {ic[1]*100:.2f}%")

    elif menu == "Intervalo de Confiança - Média":
        st.subheader("Intervalo de Confiança para Média")
        n = st.number_input("Tamanho da Amostra (n)", min_value=1, value=50, step=1)
        nivel_confianca = st.slider("Nível de Confiança (%)", 0, 100, 95, 1)
        st.markdown("**Média Amostral**: valor médio observado na amostra.")
        media_amostral = st.number_input("Média Observada", value=50.0, step=0.1)
        st.markdown("""
        **Desvio-Padrão Populacional ou Estimado** (σ):
        Se não tiver o valor exato, utilize um valor estimado com base em estudos ou 
        estimativas prévias.
        """)
        desvio_padrao = st.number_input("Desvio-Padrão (σ)", value=10.0, step=0.1)

        if st.button("Calcular Intervalo"):
            ic = intervalo_confianca_media(n, nivel_confianca, media_amostral, desvio_padrao)
            st.write("Intervalo de Confiança (aprox.):")
            st.info(f"{ic[0]:.2f} a {ic[1]:.2f}")

if __name__ == "__main__":
    main()
