import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1 Introdução
    Esse trabalho busca comparar diferentes modelos e plataformas de OCR no contexto de reconhecimento de receitas médicas, como parte de um projeto do REVAI 4.0 do LIAD (Laboratório de Inteligência Artifical E Arquiteturas Dedicadas) na UFCG. Nesse notebook, iremos comparar o desempenho de quatro alternativas de OCR: Tesseract, PaddleOCR, TrOCR e LLaVA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2 Modelos e Plataformas
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Tesseract
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2  PaddleOCR
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 TrOCR
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.4 LLaVA
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3 Dataset
    O dataset utilizado é o RxHand
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #4 Métricas

    - CER
    - WER
    - Levenshtein
    - Word Accuracy
    - Word Accuracy (70%)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -----------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5 Importando biliotecas
    """)
    return


@app.cell
def _():
    import pandas as pd
    import os

    return os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6 Carregando o Dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo, os, pd):
    # ── Caminhos do dataset (já extraído em RxHandBD) ──
    DATASET_DIR = "RxHandBD"
    CAMINHO_CSV_TREINO = os.path.join(DATASET_DIR, "Train_Label.csv")
    CAMINHO_CSV_TESTE = os.path.join(DATASET_DIR, "Test_Labels.csv")
    DIRETORIO_TREINO = os.path.join(DATASET_DIR, "Train_Set")
    DIRETORIO_TESTE = os.path.join(DATASET_DIR, "Test_Set")

    # ── Carregar CSVs ──
    df_train = pd.read_csv(CAMINHO_CSV_TREINO)
    df_test  = pd.read_csv(CAMINHO_CSV_TESTE)

    # ── Amostra fixa (500 imagens, random_state=42 para reprodutibilidade) ──
    TAMANHO_AMOSTRA = 500
    df_amostra = df_test.sample(n=TAMANHO_AMOSTRA, random_state=42).reset_index(drop=True)

    # ── Vocabulário global (treino + teste, para pós-processamento fuzzy) ──
    VOCABULARIO = (
        pd.concat([df_train["Text"], df_test["Text"]])
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    # ── Resumo ──
    mo.md(f"""
    **Dataset carregado com sucesso**

    | Conjunto | Quantidade |
    |---|---|
    | Amostra de teste | **{len(df_amostra):,}** imagens |
    | Vocabulário de treino | **{len(VOCABULARIO):,}** palavras únicas |
    | Total treino (Train_Set) | **{len(df_train):,}** imagens |
    | Total teste (Test_Set) | **{len(df_test):,}** imagens |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7 Funções auxiliares
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8 Carregando os Modelos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9 Rodando o Experimento
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10 Resultados (Tabelas)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11 Resultados (Gráficos)
    """)
    return


if __name__ == "__main__":
    app.run()
