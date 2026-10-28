import numpy as np
import pandas as pd
import streamlit as st

# Configuração da Página
st.set_page_config(
    page_title="Otimizador Inteligente de Concursos", layout="wide"
)

st.title("🎯 Otimizador de Escolha de Circunscrição em Concursos")
st.markdown(
    """
Esta ferramenta foi desenhada para ajudar você a decidir **onde se inscrever** em um edital de concurso público, baseando-se em lógica matemática, dados históricos e critérios ponderados. 
Siga o passo a passo abaixo para rodar a sua análise personalizada.
"""
)

# ==========================================
# BARRA LATERAL: CONFIGURAÇÕES E FILTROS
# ==========================================
st.sidebar.header("⚙️ Configurações da Análise")

st.sidebar.markdown("### 1. Filtro da Base Histórica")
tipo_filtro = st.sidebar.radio(
    "Quais circunscrições do concurso passado você quer incluir?",
    [
        "Todas as circunscrições disponíveis",
        "Apenas circunscrições com vagas imediatas (sem CR puro)",
    ],
)

st.sidebar.markdown("### 2. Pesos de Importância (Seus Critérios)")
st.sidebar.markdown(
    "Ajuste os seletores abaixo conforme o que você mais valoriza neste"
    " momento:"
)

peso_vagas = st.sidebar.slider(
    "📈 Quantidade de Vagas (Edital Atual)",
    0.0,
    10.0,
    4.0,
    help=(
        "Quanto maior o peso, mais o sistema vai priorizar cidades com muitas"
        " vagas abertas agora."
    ),
)
peso_inscritos = st.sidebar.slider(
    "👥 Nº de Inscritos (Histórico - Menos é melhor)",
    0.0,
    10.0,
    3.0,
    help=(
        "Quanto maior o peso, mais o sistema vai penalizar cidades que atraem"
        " multidões de candidatos."
    ),
)
peso_corte = st.sidebar.slider(
    "📉 Ponto de Corte (Histórico - Menor é melhor)",
    0.0,
    10.0,
    3.0,
    help=(
        "Quanto maior o peso, mais o sistema vai priorizar locais onde as notas"
        " de corte anteriores foram menores."
    ),
)

# Normalizando os pesos para somarem 1 (100%)
soma_pesos = peso_vagas + peso_inscritos + peso_corte
if soma_pesos > 0:
  w_vagas = peso_vagas / soma_pesos
  w_inscritos = peso_inscritos / soma_pesos
  w_corte = peso_corte / soma_pesos
else:
  w_vagas, w_inscritos, w_corte = 0.33, 0.33, 0.33

pesos = np.array([w_vagas, w_inscritos, w_corte])

# ==========================================
# PASSO 1: DADOS DE ENTRADA (TABELA EDITÁVEL)
# ==========================================
st.subheader("Passo 1: Inspecione e ajuste os dados das circunscrições")
st.markdown(
    """
* **Vagas (Atual):** Número de vagas imediatas do edital que acabou de sair.
* **Inscritos Passado:** Quantidade total de pessoas que se inscreveram naquela região no concurso anterior.
* **Ponto de Corte Passado:** A nota obtida pelo último candidato classificado no concurso anterior.
"""
)

# Dados de exemplo didáticos
dados_iniciais = pd.DataFrame(
    {
        "Circunscricao": ["Região A", "Região B", "Região C", "Região D"],
        "Tipo_Passado": [
            "Vagas Imediatas",
            "Cadastro Reserva",
            "Vagas Imediatas",
            "Vagas Imediatas",
        ],
        "Vagas_Atual": [15, 40, 10, 25],
        "Inscritos_Passado": [1200, 4500, 800, 2000],
        "PontoCorte_Passado": [75.5, 82.0, 68.0, 78.5],
    }
)

df_usuario = st.data_editor(
    dados_iniciais, num_rows="dynamic", width="stretch"
)

# Aplicando o filtro escolhido na barra lateral
if (
    tipo_filtro
    == "Apenas circunscrições com vagas imediatas (sem CR puro)"
):
  df_analise = df_usuario[
      df_usuario["Tipo_Passado"] == "Vagas Imediatas"
  ].copy()
  st.info(
      "🔍 **Filtro ativo:** Excluindo circunscrições que foram apenas de"
      " cadastro reserva no passado."
  )
else:
  df_analise = df_usuario.copy()

# Alerta de segurança estatística para amostras muito pequenas
if len(df_analise) < 3:
  st.warning(
      "⚠️ **Atenção:** Sua base de dados filtrada tem menos de 3 circunscrições."
      " Com poucas opções, os algoritmos matemáticos podem perder sensibilidade."
  )


# ==========================================
# MOTORES MATEMÁTICOS (SAW & TOPSIS)
# ==========================================
tipos_criterios = [True, False, False]


def calcular_saw(df, pesos, tipos):
  matriz = df.iloc[:, 2:].values  # Pula Circunscricao e Tipo_Passado
  n_cols = matriz.shape[1]
  matriz_norm = np.zeros_like(matriz, dtype=float)

  for j in range(n_cols):
    coluna = matriz[:, j]
    min_val, max_val = coluna.min(), coluna.max()
    if max_val - min_val == 0:
      matriz_norm[:, j] = 1.0
    else:
      if tipos[j]:  # Benefício
        matriz_norm[:, j] = (coluna - min_val) / (max_val - min_val)
      else:  # Custo (invertido)
        matriz_norm[:, j] = (max_val - coluna) / (max_val - min_val)

  return np.dot(matriz_norm, pesos)


def calcular_topsis(df, pesos, tipos):
  matriz = df.iloc[:, 2:].values.astype(float)
  denominadores = np.sqrt(np.sum(matriz**2, axis=0))
  denominadores[denominadores == 0] = 1.0
  matriz_norm = matriz / denominadores
  matriz_ponderada = matriz_norm * pesos

  ideal_positivo = np.zeros(matriz.shape[1])
  ideal_negativo = np.zeros(matriz.shape[1])

  for j in range(matriz.shape[1]):
    if tipos[j]:
      ideal_positivo[j] = np.max(matriz_ponderada[:, j])
      ideal_negativo[j] = np.min(matriz_ponderada[:, j])
    else:
      ideal_positivo[j] = np.min(matriz_ponderada[:, j])
      ideal_negativo[j] = np.max(matriz_ponderada[:, j])

  dist_pos = np.sqrt(np.sum((matriz_ponderada - ideal_positivo) ** 2, axis=1))
  dist_neg = np.sqrt(np.sum((matriz_ponderada - ideal_negativo) ** 2, axis=1))

  soma_dist = dist_pos + dist_neg
  soma_dist[soma_dist == 0] = 1.0
  return dist_neg / soma_dist


# ==========================================
# PASSO 2: EXECUÇÃO E APRESENTAÇÃO DOS RESULTADOS
# ==========================================
st.subheader("Passo 2: Resultado da Análise Multicritério")
st.markdown(
    "Clique no botão abaixo para processar os algoritmos **SAW** (soma"
    " ponderada) e **TOPSIS** (distância geométrica da cidade ideal)."
)

if st.button("🚀 Processar Escolha Estratégica", type="primary"):
  if len(df_analise) < 2:
    st.error("Erro: É necessário ter pelo menos 2 circunscrições para comparar.")
  else:
    df_resultados = df_analise.copy()

    # Rodando os modelos
    df_resultados["Score_SAW"] = calcular_saw(
        df_analise, pesos, tipos_criterios
    )
    df_resultados["Score_TOPSIS"] = calcular_topsis(
        df_analise, pesos, tipos_criterios
    )

    # Média consolidada dos dois métodos para gerar uma nota final de 0 a 1
    df_resultados["Score_Final"] = (
        df_resultados["Score_SAW"] + df_resultados["Score_TOPSIS"]
    ) / 2

    # Ordenação decrescente (do melhor para o pior)
    df_resultados = df_resultados.sort_values(
        by="Score_Final", ascending=False
    ).reset_index(drop=True)

    st.success("✨ Análise concluída com sucesso!")

    st.markdown("### 🏆 Ranking de Melhores Circunscrições para Inscrição")
    st.markdown(
        "A **primeira linha** da tabela abaixo representa a escolha de melhor"
        " custo-benefício matemático de acordo com os pesos que você definiu"
        " na barra lateral."
    )

    # Exibindo a tabela formatada corretamente dentro do escopo do botão
    st.dataframe(
        df_resultados.style.format({
            "Score_Final": "{:.4f}",
            "Score_SAW": "{:.4f}",
            "Score_TOPSIS": "{:.4f}",
        }),
        width="stretch",
    )

    st.markdown(
        """
        > **💡 Dica Didática:** 
        > * O **Score Final** varia de 0 a 1. Quanto mais próximo de 1, mais próxima aquela circunscrição está do "cenário perfeito" (muitas vagas, poucos concorrentes históricos e nota de corte acessível).
        > * Se quiser testar cenários extremos (ex: *"E se eu não me importar com concorrência e focar só em vagas?"*), basta alterar os pesos na barra lateral esquerda e o ranking se atualizará instantaneamente.
        """
    )
