import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1 Introdução
    Esse trabalho busca comparar diferentes modelos e plataformas de OCR no contexto de reconhecimento de receitas médicas, como parte de um projeto REVAI 4.0 no LIAD (Laboratório de Inteligência Artifical E Arquiteturas Dedicadas) na UFCG. Nesse notebook, iremos comparar o desempenho de quatro alternativas de OCR: Tesseract, PaddleOCR, TrOCR e LLaVA.
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
    import os
    import jiwer
    import Levenshtein
    import pandas as pd
    import pytesseract
    import torch

    from paddleocr import PaddleOCR
    from PIL import Image
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        ViTImageProcessor,
        XLMRobertaTokenizer,
    )

    return (
        Levenshtein,
        PaddleOCR,
        TrOCRProcessor,
        ViTImageProcessor,
        VisionEncoderDecoderModel,
        XLMRobertaTokenizer,
        jiwer,
        os,
        pd,
        pytesseract,
        torch,
    )


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
    # 7 Métricas e Funções Auxiliares
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.1 Métricas utilizadas

    | Métrica | Descrição |
    |---|---|
    | **Levenshtein (distância)** | Nº mínimo de inserções, remoções ou substituições de caracteres para transformar a predição no gabarito. Quanto menor, melhor. |
    | **Levenshtein (similaridade)** | Versão normalizada (0 a 1) da distância. 1 = idêntico, 0 = totalmente diferente. |
    | **CER** (*Character Error Rate*) | Taxa de erro a nível de caractere: `(inserçōes + deleções + substituições) / total de caracteres`. |
    | **WER** (*Word Error Rate*) | Taxa de erro a nível de palavra. Mesma lógica do CER, mas operando sobre palavras. |
    | **Word Accuracy** | Acurácia exata: 1 se o texto predito for **idêntico** ao gabarito, 0 caso contrário. |
    | **Word Accuracy (80%)** | Acurácia "flexível": considera correto se a similaridade de Levenshtein for **≥ 80%**. |
    """)
    return


@app.cell(hide_code=True)
def _(Levenshtein, jiwer):
    def normalizar(texto: str) -> str:
        """Normaliza texto: strip + lowercase."""
        return str(texto).strip().lower()


    def calcular_metricas(gabarito: str, predicao: str):
        """
        Calcula todas as métricas entre gabarito e predição.

        Retorna (lev_dist, lev_ratio, cer, wer, word_acc, word_acc_80).
        """
        g = normalizar(gabarito)
        p = normalizar(predicao)

        # Casos de string vazia
        if not g and not p:
            return (0, 1.0, 0.0, 0.0, 1.0, 1.0)
        if not g or not p:
            return (Levenshtein.distance(g, p), 0.0, 1.0, 1.0, 0.0, 0.0)

        lev_dist = Levenshtein.distance(g, p)
        lev_ratio = Levenshtein.ratio(g, p)
        cer = jiwer.cer(g, p)
        wer = jiwer.wer(g, p)
        word_acc = 1.0 if g == p else 0.0
        word_acc_80 = 1.0 if lev_ratio >= 0.80 else 0.0

        return (lev_dist, lev_ratio, cer, wer, word_acc, word_acc_80)

    return (normalizar,)


@app.cell(hide_code=True)
def _(Levenshtein, normalizar):
    def corrigir_fuzzy(texto_ocr: str, vocabulario: list[str], limiar: float = 0.0) -> str:
        """
        Corrige texto via fuzzy matching contra um vocabulário conhecido.

        Usa similaridade normalizada de Levenshtein. Se nenhum termo
        atingir o limiar, retorna o texto original sem alteração.
        """
        texto_ocr = normalizar(texto_ocr)
        # Remove ruídos comuns de OCR (pontuação isolada)
        texto_ocr = "".join(c for c in texto_ocr if c.isalnum() or c.isspace()).strip()

        if len(texto_ocr) < 2 or not vocabulario:
            return texto_ocr

        melhor_termo = texto_ocr
        melhor_score = 0.0

        for termo in vocabulario:
            score = Levenshtein.ratio(texto_ocr, normalizar(termo))
            if score > melhor_score:
                melhor_score = score
                melhor_termo = normalizar(termo)

        return melhor_termo if melhor_score >= limiar else texto_ocr

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
    ## 8.1 Tesseract

    Engine de OCR open-source clássica. Rápida, leve, mas sensível a ruído e variação de caligrafia.

    ⚠️ **Dependência de sistema:** requer `tesseract-ocr` instalado (`sudo apt install tesseract-ocr`).
    """)
    return


@app.cell(hide_code=True)
def _(mo, pytesseract):
    mo.md(f"""
    ✅ **Tesseract** carregado

    Versão: `pytesseract {pytesseract.__version__}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.2 PaddleOCR

    Modelo baseado em PaddleOCR, arquitetura PP-OCRv6. Reconhecimento robusto com detector + recognizer integrados.
    """)
    return


@app.cell(hide_code=True)
def _(PaddleOCR, mo, torch):
    _device = "gpu" if torch.cuda.is_available() else "cpu"

    ocr_paddle = PaddleOCR(
        ocr_version="PP-OCRv6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="transformers",
        device=_device,
        lang="en",
    )

    mo.md(f"""
    ✅ **PaddleOCR** carregado

    Dispositivo: `{_device}` | Engine: `transformers`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.3 TrOCR

    Modelo Transformer (Encoder-Decoder) pré-treinado para OCR de texto manuscrito.
    Arquitetura: `microsoft/trocr-large-handwritten` (ViT + GPT-2).
    """)
    return


@app.cell(hide_code=True)
def _(
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    XLMRobertaTokenizer,
    mo,
    torch,
):
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Workaround: usar XLMRobertaTokenizer + ViTImageProcessor em vez de
    # TrOCRProcessor.from_pretrained() para evitar erro de sentencepiece.
    _tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/trocr-large-handwritten")
    _image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    processor_trocr = TrOCRProcessor(image_processor=_image_processor, tokenizer=_tokenizer)

    model_trocr = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten"
    ).to(_device)

    mo.md(f"""
    ✅ **TrOCR** carregado

    Dispositivo: `{_device}` | Modelo: `trocr-large-handwritten`
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
