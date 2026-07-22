# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate>=1.14.0",
#     "gdown>=5.2.0",
#     "jiwer>=4.0.0",
#     "marimo>=0.23.14",
#     "matplotlib>=3.10.9",
#     "numpy>=2.2.6",
#     "opencv-python>=5.0.0.93",
#     "paddleocr>=3.7.0",
#     "pandas>=2.3.3",
#     "pillow>=12.3.0",
#     "pytesseract>=0.13.3",
#     "python-levenshtein>=0.27.3",
#     "torch>=2.13.0",
#     "torchvision>=0.28.0",
#     "transformers>=5.14.1",
# ]
# ///

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
    Esse trabalho busca comparar diferentes modelos e plataformas de OCR no contexto de reconhecimento de receitas médicas, como parte do projeto REVAI 4.0 no LIAD (Laboratório de Inteligência Artifical E Arquiteturas Dedicadas) na UFCG. Nesse notebook, iremos comparar o desempenho de quatro alternativas de OCR: Tesseract, PaddleOCR, TrOCR e Qwen3-VL.
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
    ## 2.4 Qwen3-VL
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3 Dataset
    O dataset utilizado é o [RxHandBD](https://zenodo.org/records/18478741), que prove uma coleção padronizada e pronta para uso de 5.578 palavras manuscritas extraídas de prescrições médicas.
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
    import accelerate
    import os
    import time
    import urllib.request
    import zipfile
    import cv2
    import jiwer
    import Levenshtein
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pytesseract
    import torch

    from functools import partial

    from paddleocr import PaddleOCR
    from PIL import Image
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        RobertaTokenizer,
        VisionEncoderDecoderModel,
        ViTImageProcessor,
    )

    return (
        AutoModelForMultimodalLM,
        AutoProcessor,
        Image,
        Levenshtein,
        PaddleOCR,
        RobertaTokenizer,
        ViTImageProcessor,
        VisionEncoderDecoderModel,
        cv2,
        jiwer,
        np,
        os,
        partial,
        pd,
        plt,
        pytesseract,
        time,
        torch,
        urllib,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6 Carregando o Dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.1 Download (se necessário)

    Se a pasta `RxHandBD/` não existir, o dataset será baixado automaticamente.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os, urllib, zipfile):
    DATASET_URL = "https://drive.google.com/file/d/1l6f9mAehHoBySLmWuIsn4Af833EAX54o/view?usp=sharing"
    DATASET_DIR = "RxHandBD"

    if not os.path.exists(DATASET_DIR):
        zip_path = "RxHandBD.zip"
        if not os.path.exists(zip_path):
            mo.md("Baixando dataset...")
            if "drive.google.com" in DATASET_URL:
                try:
                    import gdown

                    gdown.download(DATASET_URL, zip_path, quiet=False)
                except ImportError:
                    raise ImportError(
                        "Para links do Google Drive, instale gdown: pip install gdown"
                    )
            else:
                urllib.request.urlretrieve(DATASET_URL, zip_path)

        # Extrai diretamente para a pasta RxHandBD/
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall("RxHandBD")
        mo.md("Dataset extraído!")
    else:
        mo.md("Dataset já existe localmente.")
    return (DATASET_DIR,)


@app.cell(hide_code=True)
def _(DATASET_DIR, mo, os, pd):
    # ── Caminhos do dataset ──
    CAMINHO_CSV_TREINO = os.path.join(DATASET_DIR, "Train_Label.csv")
    CAMINHO_CSV_TESTE = os.path.join(DATASET_DIR, "Test_Labels.csv")
    DIRETORIO_TREINO = os.path.join(DATASET_DIR, "Train_Set")
    DIRETORIO_TESTE = os.path.join(DATASET_DIR, "Test_Set")

    # ── Carregar CSVs ──
    df_train = pd.read_csv(CAMINHO_CSV_TREINO)
    df_test = pd.read_csv(CAMINHO_CSV_TESTE)

    # ── Amostra fixa (500 imagens, random_state=42 para reprodutibilidade) ──
    TAMANHO_AMOSTRA = 500
    df_amostra = df_test.sample(n=TAMANHO_AMOSTRA, random_state=42).reset_index(
        drop=True
    )

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
    return DIRETORIO_TESTE, VOCABULARIO, df_amostra


@app.cell(hide_code=True)
def _(DIRETORIO_TESTE, df_amostra, mo, os):
    mo.md("### Exemplos do dataset")

    _imagens_exemplos = []
    qtd_amostras = 16
    amostra_exemplos = df_amostra.head(qtd_amostras)
    for i in range(0, qtd_amostras, 4):
        cells = []
        for j in range(i, min(i + 4, qtd_amostras)):
            row = amostra_exemplos.iloc[j]
            caminho_img = os.path.join(DIRETORIO_TESTE, str(row["Images"]))
            if os.path.exists(caminho_img):
                cells.append(mo.image(caminho_img, width=120, caption=str(row["Text"])))
        if cells:
            _imagens_exemplos.append(mo.hstack(cells, gap=0.5))
    mo.vstack(_imagens_exemplos)
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

    return calcular_metricas, normalizar


@app.cell(hide_code=True)
def _(Levenshtein, normalizar):
    def corrigir_fuzzy(
        texto_ocr: str, vocabulario: list[str], score_minimo: float = 0.0
    ) -> str:
        """
        Corrige texto via fuzzy matching contra um vocabulário conhecido.

        Usa similaridade normalizada de Levenshtein. Se nenhum termo
        atingir o score_minimo, retorna o texto original sem alteração.
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

        return melhor_termo if melhor_score >= score_minimo else texto_ocr

    return (corrigir_fuzzy,)


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
    **Tesseract** carregado

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
    **PaddleOCR** carregado

    Dispositivo: `{_device}` | Engine: `transformers`
    """)
    return (ocr_paddle,)


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
    RobertaTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    mo,
    torch,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELO_TROCR = "microsoft/trocr-large-handwritten"

    image_processor_trocr = ViTImageProcessor.from_pretrained(MODELO_TROCR)
    tokenizer_trocr = RobertaTokenizer.from_pretrained(MODELO_TROCR)
    model_trocr = VisionEncoderDecoderModel.from_pretrained(MODELO_TROCR).to(device)

    mo.md(f"""
    **TrOCR** carregado

    Dispositivo: `{device}` | Modelo: `trocr-large-handwritten`
    """)
    return image_processor_trocr, model_trocr, tokenizer_trocr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.4 Qwen3-VL

    Modelo multimodal (Vision-Language) de 8B parâmetros. Processa a imagem
    como um todo via chat template, ideal para OCR baseado em instrução.
    """)
    return


@app.cell(hide_code=True)
def _(AutoModelForMultimodalLM, AutoProcessor, mo):
    MODELO_QWEN = "Qwen/Qwen3-VL-8B-Instruct"

    processor_qwen = AutoProcessor.from_pretrained(MODELO_QWEN)
    model_qwen = AutoModelForMultimodalLM.from_pretrained(
        MODELO_QWEN, device_map="auto"
    )

    mo.md(f"""
    **Qwen3-VL** carregado

    Modelo: `Qwen3-VL-8B-Instruct` | device_map: `auto`
    """)
    return model_qwen, processor_qwen


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9 Rodando o Experimento
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.1 Funções de Predição

    Cada OCR tem sua própria função de predição com assinatura padronizada:
    `(caminho_imagem) → (texto_predito, tempo_segundos)`.

    Para adicionar um novo OCR, basta criar uma função com essa assinatura
    e registrá-la no dicionário `MOTORES_OCR` na célula seguinte.
    """)
    return


@app.cell(hide_code=True)
def _(Image, cv2, normalizar, os, pytesseract, time):
    # ──────────────────────────────────────────────────────────
    # Funções de predição — uma por motor de OCR
    # Assinatura: (caminho_imagem, ...) -> (texto, tempo)
    # ──────────────────────────────────────────────────────────

    def predizer_trocr(caminho_imagem, image_processor, tokenizer, model):
        """TrOCR — modelo Transformer para texto manuscrito."""
        try:
            start = time.time()
            image = Image.open(caminho_imagem).convert("RGB")
            pixel_values = image_processor(
                images=image, return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(model.device)
            generated_ids = model.generate(pixel_values, max_new_tokens=64)
            text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return normalizar(text), time.time() - start
        except Exception as e:
            print(f" TrOCR [{os.path.basename(caminho_imagem)}]: {e}")
            return "", 0.0

    def predizer_paddle(caminho_imagem, ocr):
        """PaddleOCR — PP-OCRv6 com detector + recognizer integrados."""
        start = time.time()
        try:
            img = cv2.imread(caminho_imagem)
            if img is None:
                return "", time.time() - start

            resultado = ocr.predict(img)
            if not resultado or not isinstance(resultado, list):
                return "", time.time() - start

            textos = []
            for res in resultado:
                if hasattr(res, "rec_texts") and res.rec_texts:
                    for t in res.rec_texts:
                        if str(t).strip():
                            textos.append(str(t).strip())
                elif isinstance(res, dict) and "rec_texts" in res:
                    for t in res["rec_texts"]:
                        if str(t).strip():
                            textos.append(str(t).strip())

            return " ".join(textos).strip().lower(), time.time() - start
        except Exception as e:
            print(f" PaddleOCR [{os.path.basename(caminho_imagem)}]: {e}")
            return "", time.time() - start

    def predizer_tesseract(caminho_imagem):
        """Tesseract — engine clássica de OCR."""
        start = time.time()
        try:
            image = Image.open(caminho_imagem)
            text = pytesseract.image_to_string(image)
            return normalizar(text), time.time() - start
        except Exception as e:
            print(f" Tesseract [{os.path.basename(caminho_imagem)}]: {e}")
            return "", 0.0

    def predizer_qwen(caminho_imagem, processor, model):
        """Qwen3-VL — modelo multimodal para OCR via instrução."""
        try:
            start = time.time()
            image = Image.open(caminho_imagem).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": (
                                "Transcreva fielmente o nome do medicamento manuscrito nesta imagem. "
                                "É uma prescrição médica de Bangladesh — o texto pode conter nomes "
                                "de fármacos, abreviações ou termos técnicos. "
                                "Retorne apenas o texto transcrito, usando somente caracteres latinos (A-Z). "
                                "Sem explicações."
                            ),
                        },
                    ],
                },
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(**inputs, max_new_tokens=64)
            text = processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            return normalizar(text), time.time() - start
        except Exception as e:
            print(f" Qwen [{os.path.basename(caminho_imagem)}]: {e}")
            return "", 0.0

    return predizer_paddle, predizer_qwen, predizer_tesseract, predizer_trocr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.2 Função Auxiliar de Benchmark

    Executa o benchmark para **um único motor de OCR** e salva o CSV.
    Cada motor tem sua própria célula de execução abaixo — rode apenas os que quiser.

    **Para adicionar um novo OCR:**
    1. Crie a função `predizer_*` na célula 9.1
    2. Copie uma das células 9.3–9.5 e ajuste os parâmetros
    """)
    return


@app.cell(hide_code=True)
def _(calcular_metricas, corrigir_fuzzy, normalizar, os, pd):
    def rodar_benchmark_ocr(
        nome, predizer, score_minimo, csv_saida, df, diretorio, vocabulario
    ):
        """Roda benchmark para um único motor OCR e salva CSV."""
        resultados = []
        total = len(df)
        print(f"[{nome}] Iniciando benchmark em {total} imagens...")

        for idx, linha in df.iterrows():
            gabarito = normalizar(linha["Text"])
            caminho = os.path.join(diretorio, str(linha["Images"]))

            if not os.path.exists(caminho):
                continue

            pred_raw, tempo = predizer(caminho)
            pred_raw = normalizar(pred_raw)
            pred_fuzzy = corrigir_fuzzy(
                pred_raw, vocabulario, score_minimo=score_minimo
            )
            pred_fuzzy = normalizar(pred_fuzzy)

            lev_d, lev_r, cer, wer, acc, acc80 = calcular_metricas(gabarito, pred_raw)
            lev_df, lev_rf, cer_f, wer_f, acc_f, acc80_f = calcular_metricas(
                gabarito, pred_fuzzy
            )

            resultados.append(
                {
                    "Arquivo": os.path.basename(caminho),
                    "Gabarito": gabarito,
                    "Predicao_Raw": pred_raw,
                    "Predicao_Fuzzy": pred_fuzzy,
                    "Levenshtein_Raw": lev_d,
                    "Levenshtein_Fuzzy": lev_df,
                    "Similaridade_Raw": lev_r,
                    "Similaridade_Fuzzy": lev_rf,
                    "CER_Raw": cer,
                    "CER_Fuzzy": cer_f,
                    "WER_Raw": wer,
                    "WER_Fuzzy": wer_f,
                    "Accuracy_Raw": acc,
                    "Accuracy80_Raw": acc80,
                    "Accuracy_Fuzzy": acc_f,
                    "Accuracy80_Fuzzy": acc80_f,
                    "Tempo_Inferencia": tempo,
                }
            )

            if (idx + 1) % 50 == 0:
                print(f"  [{nome}] {idx + 1}/{total} imagens...")

        df_resultado = pd.DataFrame(resultados)
        df_resultado.to_csv(csv_saida, index=False, encoding="utf-8")
        print(f"[{nome}]  {len(df_resultado)} amostras → {csv_saida}")
        return df_resultado

    return (rodar_benchmark_ocr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.3 Tesseract

    OCR clássico. Rápido, roda localmente sem GPU.
    ⚠️ Requer `tesseract-ocr` instalado no sistema.
    """)
    return


@app.cell(hide_code=True)
def _(
    DIRETORIO_TESTE,
    VOCABULARIO,
    df_amostra,
    mo,
    predizer_tesseract,
    rodar_benchmark_ocr,
):
    NOME_CSV_TESSERACT = "resultados_tesseract.csv"
    df_tesseract = rodar_benchmark_ocr(
        nome="Tesseract",
        predizer=predizer_tesseract,
        score_minimo=0.1,
        csv_saida=NOME_CSV_TESSERACT,
        df=df_amostra,
        diretorio=DIRETORIO_TESTE,
        vocabulario=VOCABULARIO,
    )
    mo.md(f"Tesseract: **{len(df_tesseract)}** amostras → `{NOME_CSV_TESSERACT}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.4 PaddleOCR

    PP-OCRv6 com detector + recognizer. Requer GPU para desempenho aceitável.
    """)
    return


@app.cell(hide_code=True)
def _(
    DIRETORIO_TESTE,
    VOCABULARIO,
    df_amostra,
    mo,
    ocr_paddle,
    partial,
    predizer_paddle,
    rodar_benchmark_ocr,
):
    NOME_CSV_PADDLE = "resultados_paddle.csv"
    df_paddle = rodar_benchmark_ocr(
        nome="PaddleOCR",
        predizer=partial(predizer_paddle, ocr=ocr_paddle),
        score_minimo=0.1,
        csv_saida=NOME_CSV_PADDLE,
        df=df_amostra,
        diretorio=DIRETORIO_TESTE,
        vocabulario=VOCABULARIO,
    )
    mo.md(f"PaddleOCR: **{len(df_paddle)}** amostras → `{NOME_CSV_PADDLE}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.5 TrOCR

    Transformer Encoder-Decoder (ViT + GPT-2). Modelo pesado, ideal rodar com GPU.
    """)
    return


@app.cell(hide_code=True)
def _(
    DIRETORIO_TESTE,
    VOCABULARIO,
    df_amostra,
    image_processor_trocr,
    mo,
    model_trocr,
    partial,
    predizer_trocr,
    rodar_benchmark_ocr,
    tokenizer_trocr,
):
    NOME_CSV_TROCR = "resultados_trocr.csv"
    df_trocr = rodar_benchmark_ocr(
        nome="TrOCR",
        predizer=partial(
            predizer_trocr,
            image_processor=image_processor_trocr,
            tokenizer=tokenizer_trocr,
            model=model_trocr,
        ),
        score_minimo=0.1,
        csv_saida=NOME_CSV_TROCR,
        df=df_amostra,
        diretorio=DIRETORIO_TESTE,
        vocabulario=VOCABULARIO,
    )
    mo.md(f"TrOCR: **{len(df_trocr)}** amostras → `{NOME_CSV_TROCR}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.6 Qwen3-VL

    Modelo multimodal de 8B. Pesado (~16 GB VRAM). Ideal rodar com GPU.
    """)
    return


@app.cell(hide_code=True)
def _(
    DIRETORIO_TESTE,
    VOCABULARIO,
    df_amostra,
    mo,
    model_qwen,
    partial,
    predizer_qwen,
    processor_qwen,
    rodar_benchmark_ocr,
):
    NOME_CSV_QWEN = "resultados_qwen.csv"
    df_qwen = rodar_benchmark_ocr(
        nome="Qwen3-VL",
        predizer=partial(
            predizer_qwen,
            processor=processor_qwen,
            model=model_qwen,
        ),
        score_minimo=0.1,
        csv_saida=NOME_CSV_QWEN,
        df=df_amostra,
        diretorio=DIRETORIO_TESTE,
        vocabulario=VOCABULARIO,
    )
    mo.md(f"Qwen3-VL: **{len(df_qwen)}** amostras → `{NOME_CSV_QWEN}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10 Resultados (Tabelas)

    As células abaixo carregam os CSVs disponíveis e montam as comparações.
    Rode os benchmarks que quiser (9.3–9.6) e depois venha aqui.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Lista centralizada de CSVs — usada pelas tabelas (10) e gráficos (11)
    CSVS_RESULTADO = [
        ("Tesseract", "resultados_tesseract.csv"),
        ("PaddleOCR", "resultados_paddle.csv"),
        ("TrOCR", "resultados_trocr.csv"),
        ("Qwen3-VL", "resultados_qwen.csv"),
    ]
    return (CSVS_RESULTADO,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10.1 Amostra de Predições

    Primeiras linhas de cada CSV disponível.
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, os, pd):
    # Detecta quais CSVs de resultado existem
    dfs_disponiveis = {}
    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            dfs_disponiveis[_nome] = _df

    if not dfs_disponiveis:
        mo.md(
            "⚠️  Nenhum CSV de resultado encontrado. Rode as células 9.3–9.5 primeiro."
        )
    else:
        mo.md(f"CSVs encontrados: **{', '.join(dfs_disponiveis.keys())}**")

        for _nome, _df in dfs_disponiveis.items():
            mo.md(f"### {_nome}")
            mo.ui.table(
                _df[["Arquivo", "Gabarito", "Predicao_Raw", "Predicao_Fuzzy"]].head(5)
            )
    return (dfs_disponiveis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10.2 Painel Comparativo

    Resumo agregado das métricas — todos os motores lado a lado.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, mo, pd):
    def fmt_pct(v):
        return round(v * 100, 2)

    def fmt_abs(v):
        return round(v, 2)

    if not dfs_disponiveis:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _linhas = []
        for _nome, _df in dfs_disponiveis.items():
            for variante in ("Raw", "Fuzzy"):
                _linhas.append(
                    {
                        "Motor": _nome,
                        "Variante": variante,
                        "Word Acc (%)": fmt_pct(_df[f"Accuracy_{variante}"].mean()),
                        "Acc @80% (%)": fmt_pct(_df[f"Accuracy80_{variante}"].mean()),
                        "Lev Médio": fmt_abs(_df[f"Levenshtein_{variante}"].mean()),
                        "CER (%)": fmt_pct(_df[f"CER_{variante}"].mean()),
                        "WER (%)": fmt_pct(_df[f"WER_{variante}"].mean()),
                        "Tempo Médio (s)": fmt_abs(_df["Tempo_Inferencia"].mean()),
                    }
                )

        df_painel = pd.DataFrame(_linhas)
        resultado = mo.ui.table(data=df_painel, pagination=True)
        print("to aqui")
    resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11 Resultados (Gráficos)

    Gráficos de barras agrupadas comparando Raw vs Fuzzy para cada motor.
    Detecta automaticamente os CSVs disponíveis na pasta.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.1 Word Accuracy
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, np, os, pd, plt):
    # Carregar CSVs disponíveis
    _nomes = []
    _raw_acc = []
    _fuzzy_acc = []
    _raw_acc80 = []
    _fuzzy_acc80 = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes.append(_nome)
            _raw_acc.append(_df["Accuracy_Raw"].mean() * 100)
            _fuzzy_acc.append(_df["Accuracy_Fuzzy"].mean() * 100)
            _raw_acc80.append(_df["Accuracy80_Raw"].mean() * 100)
            _fuzzy_acc80.append(_df["Accuracy80_Fuzzy"].mean() * 100)

    _x = np.arange(len(_nomes))
    _w = 0.35

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Word Accuracy
    _ax1.bar(_x - _w / 2, _raw_acc, _w, label="Raw", color="#4C72B0")
    _ax1.bar(_x + _w / 2, _fuzzy_acc, _w, label="Fuzzy", color="#DD8452")
    _ax1.set_ylabel("%")
    _ax1.set_title("Word Accuracy")
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_nomes, rotation=0, ha="right")
    _ax1.legend()
    _ax1.set_ylim(0, 105)
    for _i, (_r, _f) in enumerate(zip(_raw_acc, _fuzzy_acc)):
        _ax1.text(_i - _w / 2, _r + 1.5, f"{_r:.1f}", ha="center", fontsize=8)
        _ax1.text(_i + _w / 2, _f + 1.5, f"{_f:.1f}", ha="center", fontsize=8)

    # Acc @80%
    _ax2.bar(_x - _w / 2, _raw_acc80, _w, label="Raw", color="#4C72B0")
    _ax2.bar(_x + _w / 2, _fuzzy_acc80, _w, label="Fuzzy", color="#DD8452")
    _ax2.set_ylabel("%")
    _ax2.set_title("Word Accuracy @80%")
    _ax2.set_xticks(_x)
    _ax2.set_xticklabels(_nomes, rotation=20, ha="right")
    _ax2.legend()
    _ax2.set_ylim(0, 105)
    for _i, (_r, _f) in enumerate(zip(_raw_acc80, _fuzzy_acc80)):
        _ax2.text(_i - _w / 2, _r + 1.5, f"{_r:.1f}", ha="center", fontsize=8)
        _ax2.text(_i + _w / 2, _f + 1.5, f"{_f:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    _resultado = mo.mpl.interactive(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.2 Levenshtein e CER
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, np, os, pd, plt):
    _nomes = []
    _raw_lev = []
    _fuzzy_lev = []
    _raw_cer = []
    _fuzzy_cer = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes.append(_nome)
            _raw_lev.append(_df["Levenshtein_Raw"].mean())
            _fuzzy_lev.append(_df["Levenshtein_Fuzzy"].mean())
            _raw_cer.append(_df["CER_Raw"].mean() * 100)
            _fuzzy_cer.append(_df["CER_Fuzzy"].mean() * 100)

    if _nomes:
        _x = np.arange(len(_nomes))
        _w = 0.35

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Levenshtein Médio
        _ax1.bar(_x - _w / 2, _raw_lev, _w, label="Raw", color="#4C72B0")
        _ax1.bar(_x + _w / 2, _fuzzy_lev, _w, label="Fuzzy", color="#DD8452")
        _ax1.set_ylabel("Distância")
        _ax1.set_title("Levenshtein Médio (menor = melhor)")
        _ax1.set_xticks(_x)
        _ax1.set_xticklabels(_nomes, rotation=20, ha="right")
        _ax1.legend()
        for _i, (_r, _f) in enumerate(zip(_raw_lev, _fuzzy_lev)):
            _ax1.text(_i - _w / 2, _r + 0.1, f"{_r:.1f}", ha="center", fontsize=8)
            _ax1.text(_i + _w / 2, _f + 0.1, f"{_f:.1f}", ha="center", fontsize=8)

        # CER
        _ax2.bar(_x - _w / 2, _raw_cer, _w, label="Raw", color="#4C72B0")
        _ax2.bar(_x + _w / 2, _fuzzy_cer, _w, label="Fuzzy", color="#DD8452")
        _ax2.set_ylabel("%")
        _ax2.set_title("CER (menor = melhor)")
        _ax2.set_xticks(_x)
        _ax2.set_xticklabels(_nomes, rotation=20, ha="right")
        _ax2.legend()
        for _i, (_r, _f) in enumerate(zip(_raw_cer, _fuzzy_cer)):
            _ax2.text(_i - _w / 2, _r + 1.5, f"{_r:.1f}", ha="center", fontsize=8)
            _ax2.text(_i + _w / 2, _f + 1.5, f"{_f:.1f}", ha="center", fontsize=8)

        plt.tight_layout()
        _resultado = mo.mpl.interactive(_fig)

    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.3 Tempo de Inferência

    Comparação do tempo médio de inferência por modelo. Barras Raw vs Fuzzy
    são idênticas aqui (o tempo é o mesmo), então usamos apenas uma barra
    por modelo com anotação do valor.
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, os, pd, plt):
    _nomes = []
    _tempos = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes.append(_nome)
            _tempos.append(_df["Tempo_Inferencia"].mean())

    if _nomes:
        _colors = ["#55A868", "#4C72B0", "#C44E52", "#DD8452"][: len(_nomes)]

        _fig, _ax = plt.subplots(figsize=(10, 5))
        _bars = _ax.bar(
            _nomes, _tempos, color=_colors, edgecolor="white", linewidth=0.8
        )

        for _bar, _t in zip(_bars, _tempos):
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height() + max(_tempos) * 0.02,
                f"{_t:.2f}s",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        _ax.set_ylabel("Tempo médio (s)")
        _ax.set_title("Tempo de Inferência por Modelo (menor = melhor)")
        _ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        _resultado = mo.mpl.interactive(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.4 Boxplot da Similaridade (Levenshtein Ratio)

    Distribuição da similaridade entre predição e gabarito para cada modelo.
    Mostra mediana, quartis e outliers — vai além da média para revelar
    a consistência de cada OCR.
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, os, pd, plt):
    _nomes_raw = []
    _dados_raw = []
    _dados_fuzzy = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes_raw.append(_nome)
            _dados_raw.append(_df["Similaridade_Raw"].dropna().values)
            _dados_fuzzy.append(_df["Similaridade_Fuzzy"].dropna().values)

    if _nomes_raw:
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Raw
        _bp1 = _ax1.boxplot(
            _dados_raw,
            labels=_nomes_raw,
            patch_artist=True,
        )
        for _patch, _color in zip(
            _bp1["boxes"],
            ["#55A868", "#4C72B0", "#C44E52", "#DD8452"][: len(_nomes_raw)],
        ):
            _patch.set_facecolor(_color)
            _patch.set_alpha(0.7)
        _ax1.set_ylabel("Similaridade (Levenshtein Ratio)")
        _ax1.set_title("Similaridade — Raw")
        _ax1.set_ylim(0, 1.05)
        _ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Fuzzy
        _bp2 = _ax2.boxplot(
            _dados_fuzzy,
            labels=_nomes_raw,
            patch_artist=True,
        )
        for _patch, _color in zip(
            _bp2["boxes"],
            ["#55A868", "#4C72B0", "#C44E52", "#DD8452"][: len(_nomes_raw)],
        ):
            _patch.set_facecolor(_color)
            _patch.set_alpha(0.7)
        _ax2.set_ylabel("Similaridade (Levenshtein Ratio)")
        _ax2.set_title("Similaridade — Fuzzy")
        _ax2.set_ylim(0, 1.05)
        _ax2.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        _resultado = mo.mpl.interactive(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.5 Scatter: Word Accuracy vs Tempo de Inferência

    Visualização de trade-off: cada ponto é um modelo. O eixo X é o tempo
    de inferência (log-scale), o eixo Y é a Word Accuracy. O quadrante
    ideal é o canto **superior esquerdo** (rápido + preciso).
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, os, pd, plt):
    _nomes = []
    _acc_raw = []
    _acc_fuzzy = []
    _tempos = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes.append(_nome)
            _acc_raw.append(_df["Accuracy_Raw"].mean() * 100)
            _acc_fuzzy.append(_df["Accuracy_Fuzzy"].mean() * 100)
            _tempos.append(_df["Tempo_Inferencia"].mean())

    if _nomes:
        _colors = ["#55A868", "#4C72B0", "#C44E52", "#DD8452"][: len(_nomes)]

        _fig, _ax = plt.subplots(figsize=(10, 6))

        # Raw (círculo)
        for _i, _nome in enumerate(_nomes):
            _ax.scatter(
                _tempos[_i],
                _acc_raw[_i],
                s=180,
                color=_colors[_i],
                edgecolors="black",
                linewidth=0.8,
                zorder=5,
                marker="o",
                label=f"{_nome} (Raw)",
            )
            _ax.annotate(
                f"{_nome}\nRaw",
                (_tempos[_i], _acc_raw[_i]),
                textcoords="offset points",
                xytext=(10, 8),
                fontsize=8,
                fontweight="bold",
                color=_colors[_i],
            )

        # Fuzzy (triângulo)
        for _i, _nome in enumerate(_nomes):
            _ax.scatter(
                _tempos[_i],
                _acc_fuzzy[_i],
                s=140,
                color=_colors[_i],
                edgecolors="black",
                linewidth=0.8,
                zorder=5,
                marker="D",
                label=f"{_nome} (Fuzzy)",
            )
            _ax.annotate(
                f"Fuzzy",
                (_tempos[_i], _acc_fuzzy[_i]),
                textcoords="offset points",
                xytext=(10, -12),
                fontsize=7,
                fontstyle="italic",
                color=_colors[_i],
            )

        _ax.set_xscale("log")
        _ax.set_xlabel("Tempo de Inferência (s, log scale)")
        _ax.set_ylabel("Word Accuracy (%)")
        _ax.set_title("Trade-off: Velocidade × Acurácia")
        _ax.grid(alpha=0.3, linestyle="--")
        _ax.legend(loc="lower right", fontsize=7)

        # Sombra do quadrante ideal
        _ax.axhspan(0, 100, xmin=0, xmax=0.35, alpha=0.05, color="green")

        plt.tight_layout()
        _resultado = mo.mpl.interactive(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.6 WER (Word Error Rate)

    Taxa de erro a nível de palavra. Complementa o CER plotado em 11.2.
    Quanto menor, melhor.
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, np, os, pd, plt):
    _nomes = []
    _raw_wer = []
    _fuzzy_wer = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes.append(_nome)
            _raw_wer.append(_df["WER_Raw"].mean() * 100)
            _fuzzy_wer.append(_df["WER_Fuzzy"].mean() * 100)

    if _nomes:
        _x = np.arange(len(_nomes))
        _w = 0.35

        _fig, _ax = plt.subplots(figsize=(10, 5))

        _ax.bar(_x - _w / 2, _raw_wer, _w, label="Raw", color="#4C72B0")
        _ax.bar(_x + _w / 2, _fuzzy_wer, _w, label="Fuzzy", color="#DD8452")
        _ax.set_ylabel("%")
        _ax.set_title("WER — Word Error Rate (menor = melhor)")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_nomes, rotation=0, ha="center")
        _ax.legend()
        _ax.grid(axis="y", alpha=0.3, linestyle="--")

        for _i, (_r, _f) in enumerate(zip(_raw_wer, _fuzzy_wer)):
            _ax.text(
                _i - _w / 2,
                _r + max(_raw_wer) * 0.02,
                f"{_r:.1f}",
                ha="center",
                fontsize=8,
            )
            _ax.text(
                _i + _w / 2,
                _f + max(_fuzzy_wer) * 0.02,
                f"{_f:.1f}",
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()
        _resultado = mo.mpl.interactive(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.7 Heatmap de Métricas

    Visão panorâmica de todos os modelos × todas as métricas (variante Raw).
    Cores mais intensas = melhor desempenho. Passe o mouse sobre as células
    para ver os valores exatos. Permite identificar rapidamente o melhor
    modelo em cada métrica.
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, np, os, pd):
    import plotly.graph_objects as go

    _nomes = []
    _metricas_data = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes.append(_nome)
            _metricas_data.append(
                {
                    "Word Acc (%)": _df["Accuracy_Raw"].mean() * 100,
                    "Acc @80% (%)": _df["Accuracy80_Raw"].mean() * 100,
                    "Similaridade": _df["Similaridade_Raw"].mean() * 100,
                    "CER (%)": _df["CER_Raw"].mean() * 100,
                    "WER (%)": _df["WER_Raw"].mean() * 100,
                    "Lev. Médio": _df["Levenshtein_Raw"].mean(),
                    "Tempo (s)": _df["Tempo_Inferencia"].mean(),
                }
            )

    if _nomes:
        _metricas_nomes = [
            "Word Acc (%)",
            "Acc @80% (%)",
            "Similaridade",
            "CER (%)",
            "WER (%)",
            "Lev. Médio",
            "Tempo (s)",
        ]
        _n_metrica = len(_metricas_nomes)
        _n_modelo = len(_nomes)

        # Construir matriz de valores reais
        _matriz = np.zeros((_n_modelo, _n_metrica))
        for _i, _md in enumerate(_metricas_data):
            for _j, _mn in enumerate(_metricas_nomes):
                _matriz[_i, _j] = _md[_mn]

        # Normalizar cada métrica para 0-1 (1 = melhor)
        _matriz_norm = np.zeros_like(_matriz)
        _alta_melhor = [True, True, True, False, False, False, False]

        for _j in range(_n_metrica):
            _col = _matriz[:, _j]
            _min, _max = _col.min(), _col.max()
            if _max - _min < 1e-9:
                _matriz_norm[:, _j] = 0.5
            elif _alta_melhor[_j]:
                _matriz_norm[:, _j] = (_col - _min) / (_max - _min)
            else:
                _matriz_norm[:, _j] = (_max - _col) / (_max - _min)

        # Texto formatado para hover e anotações
        _texto_celulas = []
        for _i in range(_n_modelo):
            _linha = []
            for _j in range(_n_metrica):
                _val = _matriz[_i, _j]
                _fmt = f"{_val:.1f}" if _val < 10 else f"{_val:.0f}"
                _linha.append(_fmt)
            _texto_celulas.append(_linha)

        # Cores: RdYlGn invertido para preferências das métricas normalizadas
        _fig = go.Figure(
            data=go.Heatmap(
                z=_matriz_norm,
                x=_metricas_nomes,
                y=_nomes,
                text=_texto_celulas,
                texttemplate="%{text}",
                textfont=dict(size=13, family="sans-serif"),
                colorscale=[
                    [0.0, "#d73027"],  # vermelho = pior
                    [0.25, "#fc8d59"],
                    [0.5, "#fee08b"],  # amarelo = mediano
                    [0.75, "#91cf60"],
                    [1.0, "#1a9850"],  # verde = melhor
                ],
                zmin=0,
                zmax=1,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Score<br>normalizado", side="right"),
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["Pior", "", "Médio", "", "Melhor"],
                    len=0.6,
                    thickness=15,
                ),
                hovertemplate=(
                    "<b>Modelo:</b> %{y}<br>"
                    "<b>Métrica:</b> %{x}<br>"
                    "<b>Valor:</b> %{text}<br>"
                    "<b>Score:</b> %{z:.2f}<extra></extra>"
                ),
                xgap=3,
                ygap=3,
            )
        )

        _fig.update_layout(
            title=dict(
                text="Heatmap de Métricas — Raw (verde = melhor, vermelho = pior)",
                font=dict(size=15),
                x=0.5,
            ),
            xaxis=dict(
                title="Métrica",
                side="bottom",
                tickangle=0,
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title="Modelo",
                tickfont=dict(size=13, family="sans-serif"),
            ),
            width=850,
            height=320,
            margin=dict(l=20, r=20, t=50, b=80),
            plot_bgcolor="white",
        )

        _resultado = mo.ui.plotly(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.8 Acurácia por Comprimento da Palavra

    O desempenho varia com o tamanho da palavra? Agrupa as palavras do
    gabarito por faixa de comprimento e plota a Word Accuracy em cada faixa.
    Revela se algum modelo sofre mais com palavras longas ou curtas.
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, mo, np, os, pd, plt):
    _nomes = []
    _dataframes = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes.append(_nome)
            _dataframes.append(_df)

    if _nomes:
        # Faixas de comprimento (número de caracteres)
        _faixas = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 20)]
        _rotulos = [f"{a}-{b}" for a, b in _faixas]

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(15, 5))

        _colors = ["#55A868", "#4C72B0", "#C44E52", "#DD8452"][: len(_nomes)]
        _x = np.arange(len(_rotulos))
        _w = 0.8 / len(_nomes)

        for _variante, _ax, _titulo in [
            ("Raw", _ax1, "Word Accuracy por Comprimento — Raw"),
            ("Fuzzy", _ax2, "Word Accuracy por Comprimento — Fuzzy"),
        ]:
            for _idx, (_nome, _df) in enumerate(zip(_nomes, _dataframes)):
                _accs = []
                for _lo, _hi in _faixas:
                    _mask = _df["Gabarito"].astype(str).str.len().between(_lo, _hi)
                    if _mask.sum() > 0:
                        _accs.append(
                            _df.loc[_mask, f"Accuracy_{_variante}"].mean() * 100
                        )
                    else:
                        _accs.append(0)

                _offset = (_idx - len(_nomes) / 2 + 0.5) * _w
                _ax.bar(
                    _x + _offset,
                    _accs,
                    _w,
                    label=_nome,
                    color=_colors[_idx],
                    edgecolor="white",
                    linewidth=0.5,
                )

            _ax.set_xlabel("Comprimento da palavra (caracteres)")
            _ax.set_ylabel("Word Accuracy (%)")
            _ax.set_title(_titulo)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_rotulos)
            _ax.legend(fontsize=7)
            _ax.grid(axis="y", alpha=0.3, linestyle="--")
            _ax.set_ylim(0, 105)

        plt.tight_layout()
        _resultado = mo.mpl.interactive(_fig)
    _resultado
    return


if __name__ == "__main__":
    app.run()
