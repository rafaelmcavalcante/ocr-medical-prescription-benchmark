import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    return


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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6 Carregando o Dataset
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
