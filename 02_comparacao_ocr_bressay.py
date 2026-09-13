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
#     "plotly>=6.2.0",
#     "psutil>=7.0.0",
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
    Esse trabalho busca comparar diferentes modelos de OCR no contexto de reconhecimento de texto manuscrito, como parte do projeto REVAI 4.0 no LIAD (Laboratório de Inteligência Artifical E Arquiteturas Dedicadas) na UFCG. Nesse notebook, iremos comparar o desempenho de três alternativas de OCR — **PaddleOCR**, **TrOCR** e **Qwen3-VL** — utilizando o dataset **BRESSAY** de redações manuscritas em português brasileiro.

    Além da acurácia, este notebook também avalia a **eficiência computacional** de cada modelo, comparando execução em **GPU vs CPU** com métricas de latência, uso de VRAM e throughput.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2 Modelos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 PaddleOCR (PP-OCRv6)

    O **PaddleOCR** é uma plataforma de OCR desenvolvida pela Baidu. A versão PP-OCRv6 usa redes neurais profundas em dois estágios: detector + recognizer.

    | Aspecto | Detalhe |
    |---|---|
    | **GPU** | Suporta CUDA (recomendado) |
    | **CPU** | Suporta inferência em CPU |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 TrOCR

    O **TrOCR** (*Transformer-based OCR*) da Microsoft Research usa arquitetura encoder-decoder (ViT + GPT-2). Versão: `trocr-large-handwritten` (~340M parâmetros).

    | Aspecto | Detalhe |
    |---|---|
    | **GPU** | CUDA recomendado (~4 GB VRAM) |
    | **CPU** | Possível, porém lento (~10-30× mais lento que GPU) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 Qwen3-VL (8B)

    Modelo multimodal Vision-Language do Alibaba Cloud. 8 bilhões de parâmetros.

    | Aspecto | Detalhe |
    |---|---|
    | **GPU** | CUDA necessário (~16 GB VRAM) |
    | **CPU** | Tecnicamente possível, mas extremamente lento (minutos por palavra). Não recomendado. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3 Dataset
    O dataset utilizado é o [BRESSAY](https://github.com/arthurflor23/BRESSAY) (*Brazilian Essays Dataset for Handwritten Text Recognition*), apresentado na competição ICDAR 2024. Contém redações manuscritas em português brasileiro com imagens segmentadas em palavras, linhas, parágrafos e páginas.

    Neste notebook usamos o subconjunto de **palavras** (*words/`) do conjunto de **teste**. O dataset já está na pasta `bressay/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4 Métricas

    ## 4.1 Métricas de Acurácia

    | Métrica | Descrição |
    |---|---|
    | **Levenshtein (distância)** | Nº mínimo de inserções, remoções ou substituições de caracteres. Quanto menor, melhor. |
    | **CER** (*Character Error Rate*) | Taxa de erro a nível de caractere. |
    | **WER** (*Word Error Rate*) | Taxa de erro a nível de palavra. |
    | **Word Accuracy** | Acurácia exata: 1 se idêntico ao gabarito, 0 caso contrário. |
    | **Word Accuracy (80%)** | Similaridade de Levenshtein ≥ 80%. |

    ## 4.2 Métricas de Eficiência

    | Métrica | Descrição | Unidade |
    |---|---|---|
    | **Latência** | Tempo médio de inferência por imagem | segundos (s) |
    | **Throughput** | Imagens processadas por segundo | imagens/s |
    | **VRAM Pico** | Memória máxima alocada na GPU durante inferência | MB (apenas GPU) |
    | **RAM Pico** | Memória máxima alocada na CPU durante inferência | MB (apenas CPU) |

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
    # 5 Importando bibliotecas
    """)
    return


@app.cell
def _():
    import torchvision
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
    import plotly.graph_objects as go
    import psutil
    import torch

    from functools import partial
    from plotly.subplots import make_subplots

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
        go,
        jiwer,
        make_subplots,
        np,
        os,
        partial,
        pd,
        psutil,
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

    Se a pasta `bressay/` não existir, o dataset será baixado automaticamente do Google Drive.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os, urllib, zipfile):
    DATASET_URL = "https://drive.google.com/file/d/1XACLMLMLuMs_6EpNaOD8nd-Nn5X9T7eH/view?usp=sharing"
    DATASET_DIR = "bressay"

    if not os.path.exists(DATASET_DIR):
        zip_path = "bressay.zip"
        if not os.path.exists(zip_path):
            mo.md("Baixando dataset BRESSAY...")
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

        # Extrai na raiz (o zip já contém a pasta bressay/ internamente)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(".")
        mo.md("Dataset extraído!")
    else:
        mo.md("Dataset BRESSAY já existe localmente.")
    return (DATASET_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.2 Estrutura do BRESSAY

    O dataset BRESSAY contém redações manuscritas em português brasileiro, segmentadas em quatro níveis: **palavras** (`data/words/`), **linhas** (`data/lines/`), **parágrafos** (`data/paragraphs/`) e **páginas** (`data/pages/`). Cada imagem PNG tem um arquivo TXT correspondente com a transcrição. As partições são definidas em `sets/`.

    Neste notebook usamos três níveis para as comparações:
    - **Palavras** — imagens de palavras isoladas;
    - **Frases** — linhas de texto manuscrito;
    - **Textos** — parágrafos completos.

    > O dataset não possui recorte por **letras isoladas**, então esse nível não é avaliado.
    """)
    return


@app.cell(hide_code=True)
def _(DATASET_DIR, mo, os, pd):
    # ── Caminhos do dataset BRESSAY ──
    _ARQUIVO_TESTE = os.path.join(DATASET_DIR, "sets", "test.txt")

    # ── Carregar lista de páginas do conjunto de teste ──
    with open(_ARQUIVO_TESTE, "r") as f:
        _paginas_teste = [linha.strip() for linha in f if linha.strip()]

    # ── Níveis de segmentação usados na comparação ──
    # O BRESSAY não possui recorte por letras isoladas.
    _TIPOS_SEGMENTACAO = {
        "palavras": "words",
        "frases": "lines",
        "textos": "paragraphs",
    }

    def _carregar_tipo(nome_tipo: str, pasta_tipo: str) -> pd.DataFrame:
        """Carrega um nível de segmentação do conjunto de teste."""
        registros = []
        raiz = os.path.join(DATASET_DIR, "data", pasta_tipo)
        for pagina in _paginas_teste:
            pasta_pagina = os.path.join(raiz, pagina)
            if not os.path.isdir(pasta_pagina):
                continue
            for arquivo in os.listdir(pasta_pagina):
                if arquivo.endswith(".png"):
                    caminho_img = os.path.join(pasta_pagina, arquivo)
                    caminho_txt = os.path.join(
                        pasta_pagina, arquivo.replace(".png", ".txt")
                    )
                    if os.path.exists(caminho_txt):
                        with open(caminho_txt, "r") as f:
                            transcricao = f.read().strip()
                        registros.append(
                            {
                                "Tipo": nome_tipo,
                                "Images": caminho_img,
                                "Text": transcricao,
                            }
                        )
        return pd.DataFrame(registros)

    _dfs_por_tipo = {
        nome: _carregar_tipo(nome, pasta)
        for nome, pasta in _TIPOS_SEGMENTACAO.items()
    }

    _df_test = pd.concat(_dfs_por_tipo.values(), ignore_index=True)

    # ── Amostra geral estratificada por tipo (500 imagens, random_state=42) ──
    _TAMANHO_AMOSTRA_POR_TIPO = {"palavras": 200, "frases": 200, "textos": 100}
    _amostras = []
    for tipo, df_tipo in _dfs_por_tipo.items():
        n = min(_TAMANHO_AMOSTRA_POR_TIPO.get(tipo, 0), len(df_tipo))
        if n > 0:
            _amostras.append(df_tipo.sample(n=n, random_state=42))
    if _amostras:
        df_amostra = (
            pd.concat(_amostras, ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
    else:
        df_amostra = pd.DataFrame(columns=["Tipo", "Images", "Text"])

    # ── Resumo ──
    _linhas_resumo = []
    for tipo, df_tipo in _dfs_por_tipo.items():
        n_amostra = int((df_amostra["Tipo"] == tipo).sum())
        _linhas_resumo.append(
            f"| {tipo.capitalize()} | **{len(df_tipo):,}** | **{n_amostra:,}** |"
        )

    _resumo_linhas = "\n".join(_linhas_resumo)
    mo.md(f"""
    **Dataset BRESSAY carregado com sucesso**

    | Nível | Total no teste | Amostra usada |
    |---|---|---|
    {_resumo_linhas}
    | **Geral** | **{len(_df_test):,}** | **{len(df_amostra):,}** |
    | Páginas de teste | **{len(_paginas_teste):,}** | — |
    """)
    return df_amostra


@app.cell(hide_code=True)
def _(df_amostra, mo, os):
    mo.md("### Exemplos do dataset")

    _imagens_exemplos = []
    _qtd_amostras = 16
    _amostra_exemplos = df_amostra.head(_qtd_amostras)
    for i in range(0, _qtd_amostras, 4):
        _cells = []
        for j in range(i, min(i + 4, _qtd_amostras)):
            row = _amostra_exemplos.iloc[j]
            _caminho_exemplo = str(row["Images"])
            if os.path.exists(_caminho_exemplo):
                _cells.append(
                    mo.image(
                        _caminho_exemplo,
                        width=120,
                        caption=f"{row['Tipo']}: {row['Text']}",
                    )
                )
        if _cells:
            _imagens_exemplos.append(mo.hstack(_cells, gap=0.5))
    mo.vstack(_imagens_exemplos)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7 Métricas e Funções Auxiliares
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
def _(mo):
    mo.md(r"""
    # 8 Carregando os Modelos

    Cada modelo é carregado em duas instâncias: **GPU** (CUDA) e **CPU**.
    As células de benchmark abaixo usam a instância apropriada conforme o dispositivo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.1 Disponibilidade de GPU
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    gpu_disponivel = torch.cuda.is_available()
    if gpu_disponivel:
        _nome_gpu = torch.cuda.get_device_name(0)
        _vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mo.md(
            f"✅ **GPU detectada:** {_nome_gpu} ({_vram_total:.1f} GB VRAM total)"
        )
    else:
        mo.md("⚠️ **Nenhuma GPU detectada.** Apenas benchmarks em CPU serão executados.")
    return (gpu_disponivel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.2 PaddleOCR (GPU + CPU)
    """)
    return


@app.cell(hide_code=True)
def _(PaddleOCR, gpu_disponivel, mo):
    ocr_paddle_gpu = None
    ocr_paddle_cpu = PaddleOCR(
        ocr_version="PP-OCRv6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="transformers",
        device="cpu",
        lang="pt",
    )

    if gpu_disponivel:
        ocr_paddle_gpu = PaddleOCR(
            ocr_version="PP-OCRv6",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="transformers",
            device="gpu",
            lang="pt",
        )

    _status_gpu = "✅ GPU" if ocr_paddle_gpu is not None else "❌ Indisponível"
    mo.md(f"""
    **PaddleOCR** carregado

    | Dispositivo | Status |
    |---|---|
    | GPU | {_status_gpu} |
    | CPU | ✅ CPU |
    """)
    return ocr_paddle_cpu, ocr_paddle_gpu


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.3 TrOCR (GPU + CPU)
    """)
    return


@app.cell(hide_code=True)
def _(
    RobertaTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    gpu_disponivel,
    mo,
    torch,
):
    MODELO_TROCR = "microsoft/trocr-large-handwritten"

    # GPU
    model_trocr_gpu = None
    image_processor_trocr_gpu = None
    tokenizer_trocr_gpu = None

    # CPU
    device_cpu = torch.device("cpu")
    image_processor_trocr_cpu = ViTImageProcessor.from_pretrained(MODELO_TROCR)
    tokenizer_trocr_cpu = RobertaTokenizer.from_pretrained(MODELO_TROCR)
    model_trocr_cpu = VisionEncoderDecoderModel.from_pretrained(MODELO_TROCR).to(
        device_cpu
    )

    if gpu_disponivel:
        device_gpu = torch.device("cuda")
        image_processor_trocr_gpu = ViTImageProcessor.from_pretrained(MODELO_TROCR)
        tokenizer_trocr_gpu = RobertaTokenizer.from_pretrained(MODELO_TROCR)
        model_trocr_gpu = VisionEncoderDecoderModel.from_pretrained(MODELO_TROCR).to(
            device_gpu
        )

    _status_gpu = "✅ GPU" if model_trocr_gpu is not None else "❌ Indisponível"
    mo.md(f"""
    **TrOCR** carregado

    | Dispositivo | Status |
    |---|---|
    | GPU | {_status_gpu} |
    | CPU | ✅ CPU |
    """)
    return (
        image_processor_trocr_cpu,
        image_processor_trocr_gpu,
        model_trocr_cpu,
        model_trocr_gpu,
        tokenizer_trocr_cpu,
        tokenizer_trocr_gpu,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.4 Qwen3-VL (GPU + CPU)
    """)
    return


@app.cell(hide_code=True)
def _(AutoModelForMultimodalLM, AutoProcessor, gpu_disponivel, mo):
    MODELO_QWEN = "Qwen/Qwen3-VL-8B-Instruct"

    model_qwen_gpu = None
    processor_qwen_gpu = None

    # CPU
    processor_qwen_cpu = AutoProcessor.from_pretrained(MODELO_QWEN)
    model_qwen_cpu = AutoModelForMultimodalLM.from_pretrained(
        MODELO_QWEN, device_map="cpu"
    )

    if gpu_disponivel:
        processor_qwen_gpu = AutoProcessor.from_pretrained(MODELO_QWEN)
        model_qwen_gpu = AutoModelForMultimodalLM.from_pretrained(
            MODELO_QWEN, device_map="auto"
        )

    _status_gpu = "✅ GPU" if model_qwen_gpu is not None else "❌ Indisponível"
    mo.md(f"""
    **Qwen3-VL** carregado

    | Dispositivo | Status |
    |---|---|
    | GPU | {_status_gpu} |
    | CPU | ⚠️ CPU (extremamente lento — minutos por palavra) |
    """)
    return (
        model_qwen_cpu,
        model_qwen_gpu,
        processor_qwen_cpu,
        processor_qwen_gpu,
    )


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
    `(caminho_imagem) → (texto_predito, tempo_segundos, vram_peak_mb, ram_peak_mb)`.
    """)
    return


@app.cell(hide_code=True)
def _(Image, cv2, normalizar, os, psutil, time, torch):
    # ──────────────────────────────────────────────────────────
    # Funções de predição — uma por motor de OCR
    # Assinatura: (caminho_imagem, ...) -> (texto, tempo, vram_mb, ram_mb)
    # ──────────────────────────────────────────────────────────

    def _medir_ram_peak():
        """Retorna o pico de RAM do processo atual em MB."""
        proc = psutil.Process()
        return proc.memory_info().rss / 1024**2

    def _medir_vram_peak():
        """Retorna o pico de VRAM alocada em MB (0 se sem GPU)."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    def predizer_trocr(caminho_imagem, image_processor, tokenizer, model):
        """TrOCR — modelo Transformer para texto manuscrito."""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            ram_antes = _medir_ram_peak()
            start = time.time()

            image = Image.open(caminho_imagem).convert("RGB")
            pixel_values = image_processor(
                images=image, return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(model.device)
            generated_ids = model.generate(pixel_values, max_new_tokens=256)
            text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            elapsed = time.time() - start
            vram_mb = _medir_vram_peak()
            ram_mb = _medir_ram_peak() - ram_antes
            return normalizar(text), elapsed, vram_mb, max(ram_mb, 0)
        except Exception as e:
            print(f" TrOCR [{os.path.basename(caminho_imagem)}]: {e}")
            return "", 0.0, 0.0, 0.0

    def predizer_paddle(caminho_imagem, ocr):
        """PaddleOCR — PP-OCRv6 com detector + recognizer integrados."""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            ram_antes = _medir_ram_peak()
            start = time.time()

            img = cv2.imread(caminho_imagem)
            if img is None:
                return "", time.time() - start, 0.0, 0.0

            resultado = ocr.predict(img)
            if not resultado or not isinstance(resultado, list):
                return "", time.time() - start, 0.0, 0.0

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

            elapsed = time.time() - start
            vram_mb = _medir_vram_peak()
            ram_mb = _medir_ram_peak() - ram_antes
            return (
                " ".join(textos).strip().lower(),
                elapsed,
                vram_mb,
                max(ram_mb, 0),
            )
        except Exception as e:
            print(f" PaddleOCR [{os.path.basename(caminho_imagem)}]: {e}")
            return "", 0.0, 0.0, 0.0

    def predizer_qwen(caminho_imagem, processor, model):
        """Qwen3-VL — modelo multimodal para OCR via instrução."""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            ram_antes = _medir_ram_peak()
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
                                "Transcreva fielmente o texto manuscrito nesta imagem. "
                                "É uma redação em português brasileiro (pode ser uma "
                                "palavra, uma frase ou um parágrafo). Retorne apenas o "
                                "texto transcrito, sem explicações."
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

            outputs = model.generate(**inputs, max_new_tokens=256)
            text = processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            elapsed = time.time() - start
            vram_mb = _medir_vram_peak()
            ram_mb = _medir_ram_peak() - ram_antes
            return normalizar(text), elapsed, vram_mb, max(ram_mb, 0)
        except Exception as e:
            print(f" Qwen [{os.path.basename(caminho_imagem)}]: {e}")
            return "", 0.0, 0.0, 0.0

    return predizer_paddle, predizer_qwen, predizer_trocr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.2 Função Auxiliar de Benchmark

    Executa o benchmark para **um único motor de OCR** e salva o CSV.
    Registra métricas de acurácia e eficiência (latência, VRAM, RAM).
    """)
    return


@app.cell(hide_code=True)
def _(calcular_metricas, normalizar, os, pd):
    def rodar_benchmark_ocr(nome, predizer, csv_saida, df):
        """Roda benchmark para um único motor OCR e salva CSV."""
        resultados = []
        total = len(df)
        print(f"[{nome}] Iniciando benchmark em {total} imagens...")

        for idx, linha in df.iterrows():
            gabarito = normalizar(linha["Text"])
            caminho = str(linha["Images"])

            if not os.path.exists(caminho):
                continue

            predicao, tempo, vram_mb, ram_mb = predizer(caminho)
            predicao = normalizar(predicao)

            lev_d, lev_r, cer, wer, acc, acc80 = calcular_metricas(
                gabarito, predicao
            )

            resultados.append(
                {
                    "Tipo": linha.get("Tipo", "geral"),
                    "Arquivo": os.path.basename(caminho),
                    "Gabarito": gabarito,
                    "Predicao": predicao,
                    "Levenshtein": lev_d,
                    "Similaridade": lev_r,
                    "CER": cer,
                    "WER": wer,
                    "Accuracy": acc,
                    "Accuracy80": acc80,
                    "Tempo_Inferencia": tempo,
                    "VRAM_Pico_MB": vram_mb,
                    "RAM_Pico_MB": ram_mb,
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
    ## 9.3 PaddleOCR — GPU
    """)
    return


@app.cell(hide_code=True)
def _(
    df_amostra,
    gpu_disponivel,
    mo,
    ocr_paddle_gpu,
    partial,
    predizer_paddle,
    rodar_benchmark_ocr,
):
    if gpu_disponivel and ocr_paddle_gpu is not None:
        _NOME_CSV = "resultados_paddle_gpu_bressay.csv"
        df_paddle_gpu = rodar_benchmark_ocr(
            nome="PaddleOCR-GPU",
            predizer=partial(predizer_paddle, ocr=ocr_paddle_gpu),
            csv_saida=_NOME_CSV,
            df=df_amostra,
        )
        mo.md(f"PaddleOCR GPU: **{len(df_paddle_gpu)}** amostras → `{_NOME_CSV}`")
    else:
        df_paddle_gpu = None
        mo.md("⚠️ GPU indisponível. Pule esta célula.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.4 PaddleOCR — CPU
    """)
    return


@app.cell(hide_code=True)
def _(
    df_amostra,
    mo,
    ocr_paddle_cpu,
    partial,
    predizer_paddle,
    rodar_benchmark_ocr,
):
    _NOME_CSV = "resultados_paddle_cpu_bressay.csv"
    df_paddle_cpu = rodar_benchmark_ocr(
        nome="PaddleOCR-CPU",
        predizer=partial(predizer_paddle, ocr=ocr_paddle_cpu),
        csv_saida=_NOME_CSV,
        df=df_amostra,
    )
    mo.md(f"PaddleOCR CPU: **{len(df_paddle_cpu)}** amostras → `{_NOME_CSV}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.5 TrOCR — GPU
    """)
    return


@app.cell(hide_code=True)
def _(
    df_amostra,
    gpu_disponivel,
    image_processor_trocr_gpu,
    mo,
    model_trocr_gpu,
    partial,
    predizer_trocr,
    rodar_benchmark_ocr,
    tokenizer_trocr_gpu,
):
    if gpu_disponivel and model_trocr_gpu is not None:
        _NOME_CSV = "resultados_trocr_gpu_bressay.csv"
        df_trocr_gpu = rodar_benchmark_ocr(
            nome="TrOCR-GPU",
            predizer=partial(
                predizer_trocr,
                image_processor=image_processor_trocr_gpu,
                tokenizer=tokenizer_trocr_gpu,
                model=model_trocr_gpu,
            ),
            csv_saida=_NOME_CSV,
            df=df_amostra,
        )
        mo.md(f"TrOCR GPU: **{len(df_trocr_gpu)}** amostras → `{_NOME_CSV}`")
    else:
        df_trocr_gpu = None
        mo.md("⚠️ GPU indisponível. Pule esta célula.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.6 TrOCR — CPU
    """)
    return


@app.cell(hide_code=True)
def _(
    df_amostra,
    image_processor_trocr_cpu,
    mo,
    model_trocr_cpu,
    partial,
    predizer_trocr,
    rodar_benchmark_ocr,
    tokenizer_trocr_cpu,
):
    _NOME_CSV = "resultados_trocr_cpu_bressay.csv"
    df_trocr_cpu = rodar_benchmark_ocr(
        nome="TrOCR-CPU",
        predizer=partial(
            predizer_trocr,
            image_processor=image_processor_trocr_cpu,
            tokenizer=tokenizer_trocr_cpu,
            model=model_trocr_cpu,
        ),
        csv_saida=_NOME_CSV,
        df=df_amostra,
    )
    mo.md(f"TrOCR CPU: **{len(df_trocr_cpu)}** amostras → `{_NOME_CSV}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.7 Qwen3-VL — GPU
    """)
    return


@app.cell(hide_code=True)
def _(
    df_amostra,
    gpu_disponivel,
    mo,
    model_qwen_gpu,
    partial,
    predizer_qwen,
    processor_qwen_gpu,
    rodar_benchmark_ocr,
):
    if gpu_disponivel and model_qwen_gpu is not None:
        _NOME_CSV = "resultados_qwen_gpu_bressay.csv"
        df_qwen_gpu = rodar_benchmark_ocr(
            nome="Qwen3-VL-GPU",
            predizer=partial(
                predizer_qwen,
                processor=processor_qwen_gpu,
                model=model_qwen_gpu,
            ),
            csv_saida=_NOME_CSV,
            df=df_amostra,
        )
        mo.md(f"Qwen3-VL GPU: **{len(df_qwen_gpu)}** amostras → `{_NOME_CSV}`")
    else:
        df_qwen_gpu = None
        mo.md("⚠️ GPU indisponível. Pule esta célula.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.8 Qwen3-VL — CPU

    ⚠️ **Atenção:** Qwen3-VL 8B em CPU é extremamente lento (minutos por palavra).
    Rode apenas se tiver muita paciência ou para poucas imagens de teste.
    """)
    return


@app.cell(hide_code=True)
def _(
    df_amostra,
    mo,
    model_qwen_cpu,
    partial,
    predizer_qwen,
    processor_qwen_cpu,
    rodar_benchmark_ocr,
):
    _NOME_CSV = "resultados_qwen_cpu_bressay.csv"
    df_qwen_cpu = rodar_benchmark_ocr(
        nome="Qwen3-VL-CPU",
        predizer=partial(
            predizer_qwen,
            processor=processor_qwen_cpu,
            model=model_qwen_cpu,
        ),
        csv_saida=_NOME_CSV,
        df=df_amostra,
    )
    mo.md(f"Qwen3-VL CPU: **{len(df_qwen_cpu)}** amostras → `{_NOME_CSV}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10 Resultados (Tabelas)

    As células abaixo carregam os CSVs disponíveis e montam as comparações.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Lista centralizada de CSVs — usada por tabelas (10) e gráficos (11)
    CSVS_RESULTADO = [
        ("PaddleOCR GPU", "resultados_paddle_gpu_bressay.csv"),
        ("PaddleOCR CPU", "resultados_paddle_cpu_bressay.csv"),
        ("TrOCR GPU", "resultados_trocr_gpu_bressay.csv"),
        ("TrOCR CPU", "resultados_trocr_cpu_bressay.csv"),
        ("Qwen3-VL GPU", "resultados_qwen_gpu_bressay.csv"),
        ("Qwen3-VL CPU", "resultados_qwen_cpu_bressay.csv"),
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
    def _normalizar_resultado(df):
        """Garante colunas do novo formato e adiciona o nível (Tipo) quando ausente."""
        df = df.copy()
        if "Tipo" not in df.columns:
            df["Tipo"] = "palavras"
        renomear = {
            "Predicao_Raw": "Predicao",
            "Levenshtein_Raw": "Levenshtein",
            "Similaridade_Raw": "Similaridade",
            "CER_Raw": "CER",
            "WER_Raw": "WER",
            "Accuracy_Raw": "Accuracy",
            "Accuracy80_Raw": "Accuracy80",
        }
        df = df.rename(columns={k: v for k, v in renomear.items() if k in df.columns})
        colunas_fuzzy = [c for c in df.columns if "Fuzzy" in c]
        if colunas_fuzzy:
            df = df.drop(columns=colunas_fuzzy)
        return df

    dfs_disponiveis = {}
    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            dfs_disponiveis[_nome] = _normalizar_resultado(pd.read_csv(_csv))

    if not dfs_disponiveis:
        mo.md(
            "⚠️  Nenhum CSV de resultado encontrado. Rode as células 9.3–9.8 primeiro."
        )
    else:
        mo.md(f"CSVs encontrados: **{', '.join(dfs_disponiveis.keys())}**")
        for _nome, _df in dfs_disponiveis.items():
            mo.md(f"### {_nome}")
            mo.ui.table(
                _df[["Tipo", "Arquivo", "Gabarito", "Predicao"]].head(5)
            )
    return (dfs_disponiveis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10.2 Painel Comparativo — Acurácia

    Resumo agregado das métricas de acurácia — todos os motores lado a lado.
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
            _linhas.append(
                {
                    "Motor": _nome,
                    "Word Acc (%)": fmt_pct(_df["Accuracy"].mean()),
                    "Acc @80% (%)": fmt_pct(_df["Accuracy80"].mean()),
                    "Lev Médio": fmt_abs(_df["Levenshtein"].mean()),
                    "CER (%)": fmt_pct(_df["CER"].mean()),
                    "WER (%)": fmt_pct(_df["WER"].mean()),
                }
            )

        df_painel = pd.DataFrame(_linhas)
        _resultado = mo.ui.table(data=df_painel, pagination=True)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10.3 Painel Comparativo — Eficiência

    Latência, throughput, VRAM e RAM. Comparação direta GPU vs CPU.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, mo, pd):
    if not dfs_disponiveis:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _linhas_eff = []
        for _nome, _df in dfs_disponiveis.items():
            _tempo_medio = _df["Tempo_Inferencia"].mean()
            _throughput = 1.0 / _tempo_medio if _tempo_medio > 0 else 0
            _vram_medio = _df["VRAM_Pico_MB"].mean()
            _ram_medio = _df["RAM_Pico_MB"].mean()

            _linhas_eff.append(
                {
                    "Motor": _nome,
                    "Latência Média (s)": round(_tempo_medio, 4),
                    "Throughput (img/s)": round(_throughput, 4),
                    "VRAM Pico (MB)": round(_vram_medio, 1) if _vram_medio > 0 else "—",
                    "RAM Pico (MB)": round(_ram_medio, 1),
                }
            )

        df_eff = pd.DataFrame(_linhas_eff)
        _resultado = mo.ui.table(data=df_eff, pagination=True)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10.4 Speedup GPU vs CPU

    Razão entre o tempo de CPU e GPU para cada modelo. Valores > 1 indicam
    quantas vezes a GPU é mais rápida.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, mo, pd):
    if not dfs_disponiveis:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _modelos = ["PaddleOCR", "TrOCR", "Qwen3-VL"]
        _linhas_speedup = []

        for _mod in _modelos:
            _key_gpu = f"{_mod} GPU"
            _key_cpu = f"{_mod} CPU"
            if _key_gpu in dfs_disponiveis and _key_cpu in dfs_disponiveis:
                _t_gpu = dfs_disponiveis[_key_gpu]["Tempo_Inferencia"].mean()
                _t_cpu = dfs_disponiveis[_key_cpu]["Tempo_Inferencia"].mean()
                _speedup = _t_cpu / _t_gpu if _t_gpu > 0 else float("inf")
                _linhas_speedup.append(
                    {
                        "Modelo": _mod,
                        "Tempo GPU (s)": round(_t_gpu, 4),
                        "Tempo CPU (s)": round(_t_cpu, 4),
                        "Speedup (GPU×)": (
                            f"{_speedup:.1f}× mais rápida"
                            if _speedup < 100
                            else f"{_speedup:.0f}× mais rápida"
                        ),
                    }
                )
            elif _key_gpu in dfs_disponiveis:
                _t_gpu = dfs_disponiveis[_key_gpu]["Tempo_Inferencia"].mean()
                _linhas_speedup.append(
                    {
                        "Modelo": _mod,
                        "Tempo GPU (s)": round(_t_gpu, 4),
                        "Tempo CPU (s)": "—",
                        "Speedup (GPU×)": "CPU não disponível",
                    }
                )
            elif _key_cpu in dfs_disponiveis:
                _t_cpu = dfs_disponiveis[_key_cpu]["Tempo_Inferencia"].mean()
                _linhas_speedup.append(
                    {
                        "Modelo": _mod,
                        "Tempo GPU (s)": "—",
                        "Tempo CPU (s)": round(_t_cpu, 4),
                        "Speedup (GPU×)": "GPU não disponível",
                    }
                )

        if _linhas_speedup:
            df_speedup = pd.DataFrame(_linhas_speedup)
            _resultado = mo.ui.table(data=df_speedup, pagination=True)
        else:
            _resultado = mo.md("⚠️ Dados insuficientes para comparar GPU vs CPU.")
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10.5 Painel Comparativo por Nível (Palavras / Frases / Textos)

    Mesmas métricas de acurácia, separadas por nível de segmentação do BRESSAY.
    Útil para comparar, por exemplo, o desempenho de um modelo em **palavras
    isoladas** vs **frases** vs **textos completos**.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, mo, pd):
    if not dfs_disponiveis:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _linhas = []
        for _nome, _df in dfs_disponiveis.items():
            for _tipo, _sub in _df.groupby("Tipo"):
                _linhas.append(
                    {
                        "Motor": _nome,
                        "Nível": _tipo,
                        "N": len(_sub),
                        "Word Acc (%)": round(_sub["Accuracy"].mean() * 100, 2),
                        "Acc @80% (%)": round(_sub["Accuracy80"].mean() * 100, 2),
                        "Lev Médio": round(_sub["Levenshtein"].mean(), 2),
                        "CER (%)": round(_sub["CER"].mean() * 100, 2),
                        "WER (%)": round(_sub["WER"].mean() * 100, 2),
                    }
                )

        df_painel_tipo = pd.DataFrame(_linhas)
        _resultado = mo.ui.table(data=df_painel_tipo, pagination=True)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11 Resultados (Gráficos)

    Gráficos comparando os motores de OCR. Detecta automaticamente os CSVs
    disponíveis na pasta.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.1 Word Accuracy
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, go, make_subplots, mo):
    _nomes = list(dfs_disponiveis.keys())
    _acc = [dfs_disponiveis[n]["Accuracy"].mean() * 100 for n in _nomes]
    _acc80 = [dfs_disponiveis[n]["Accuracy80"].mean() * 100 for n in _nomes]

    if not _nomes:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Word Accuracy", "Word Accuracy @80%"),
        )

        _fig.add_trace(
            go.Bar(
                name="Word Accuracy", x=_nomes, y=_acc,
                text=[f"{v:.1f}" for v in _acc], textposition="outside",
                marker_color="#4C72B0", marker_line=dict(color="white", width=0.8),
                hovertemplate="<b>%{x}</b><br>Word Acc: %{y:.1f}%<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Bar(
                name="Acc @80%", x=_nomes, y=_acc80,
                text=[f"{v:.1f}" for v in _acc80], textposition="outside",
                marker_color="#55A868", marker_line=dict(color="white", width=0.8),
                hovertemplate="<b>%{x}</b><br>Acc @80%: %{y:.1f}%<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

        _fig.update_layout(
            barmode="group",
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=50, b=80),
        )
        _fig.update_yaxes(range=[0, 110], title="%", row=1, col=1)
        _fig.update_yaxes(range=[0, 110], title="%", row=1, col=2)

        _resultado = mo.ui.plotly(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.2 Levenshtein e CER
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, go, make_subplots, mo):
    _nomes = list(dfs_disponiveis.keys())
    _lev = [dfs_disponiveis[n]["Levenshtein"].mean() for n in _nomes]
    _cer = [dfs_disponiveis[n]["CER"].mean() * 100 for n in _nomes]

    if not _nomes:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Levenshtein Médio (menor = melhor)",
                "CER — menor = melhor",
            ),
        )

        _fig.add_trace(
            go.Bar(
                name="Levenshtein", x=_nomes, y=_lev,
                text=[f"{v:.1f}" for v in _lev], textposition="outside",
                marker_color="#4C72B0", marker_line=dict(color="white", width=0.8),
                hovertemplate="<b>%{x}</b><br>Lev: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Bar(
                name="CER", x=_nomes, y=_cer,
                text=[f"{v:.1f}" for v in _cer], textposition="outside",
                marker_color="#DD8452", marker_line=dict(color="white", width=0.8),
                hovertemplate="<b>%{x}</b><br>CER: %{y:.1f}%<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

        _fig.update_layout(
            barmode="group",
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=50, b=80),
        )
        _fig.update_yaxes(title="Distância", row=1, col=1)
        _fig.update_yaxes(title="%", row=1, col=2)

        _resultado = mo.ui.plotly(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.3 Tempo de Inferência

    Comparação do tempo médio de inferência por modelo. Inclui GPU e CPU lado a lado.
    """)
    return


@app.cell(hide_code=True)
def _(CSVS_RESULTADO, go, mo, os, pd):
    _nomes = []
    _tempos = []

    for _nome, _csv in CSVS_RESULTADO:
        if os.path.exists(_csv):
            _df = pd.read_csv(_csv)
            _nomes.append(_nome)
            _tempos.append(_df["Tempo_Inferencia"].mean())

    if _nomes:
        # Cores: GPU = tons frios, CPU = tons quentes
        _colors = []
        for _n in _nomes:
            if "GPU" in _n:
                _colors.append("#4C72B0")  # azul
            else:
                _colors.append("#DD8452")  # laranja

        _fig = go.Figure()

        _fig.add_trace(
            go.Bar(
                x=_nomes,
                y=_tempos,
                marker_color=_colors,
                marker_line=dict(color="white", width=0.8),
                text=[f"{t:.3f}s" for t in _tempos],
                textposition="outside",
                textfont=dict(size=10, family="sans-serif"),
                hovertemplate="<b>%{x}</b><br>Tempo: %{y:.4f}s<extra></extra>",
            )
        )

        _fig.update_layout(
            title=dict(
                text="Tempo de Inferência por Modelo (menor = melhor)",
                font=dict(size=15),
            ),
            yaxis=dict(title="Tempo médio (s)", gridcolor="rgba(0,0,0,0.1)"),
            template="plotly_white",
            height=430,
            margin=dict(l=20, r=20, t=50, b=80),
            showlegend=False,
        )

        _resultado = mo.ui.plotly(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.4 Speedup GPU vs CPU

    Gráfico de barras com a razão de speedup (tempo CPU / tempo GPU).
    Barras acima de 1× indicam ganho com GPU.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, go, mo):
    if not dfs_disponiveis:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _modelos = ["PaddleOCR", "TrOCR", "Qwen3-VL"]
        _nomes_speedup = []
        _speedups = []
        _textos = []

        for _mod in _modelos:
            _key_gpu = f"{_mod} GPU"
            _key_cpu = f"{_mod} CPU"
            if _key_gpu in dfs_disponiveis and _key_cpu in dfs_disponiveis:
                _t_gpu = dfs_disponiveis[_key_gpu]["Tempo_Inferencia"].mean()
                _t_cpu = dfs_disponiveis[_key_cpu]["Tempo_Inferencia"].mean()
                _sp = _t_cpu / _t_gpu if _t_gpu > 0 else 0
                _nomes_speedup.append(_mod)
                _speedups.append(_sp)
                _textos.append(f"{_sp:.1f}×")

        if _nomes_speedup:
            _fig = go.Figure()

            _fig.add_trace(
                go.Bar(
                    x=_nomes_speedup,
                    y=_speedups,
                    marker_color=["#55A868", "#4C72B0", "#C44E52"][
                        : len(_nomes_speedup)
                    ],
                    marker_line=dict(color="white", width=0.8),
                    text=_textos,
                    textposition="outside",
                    textfont=dict(size=13, family="sans-serif"),
                    hovertemplate="<b>%{x}</b><br>Speedup: %{y:.1f}×<extra></extra>",
                )
            )

            _fig.add_hline(
                y=1,
                line_dash="dash",
                line_color="gray",
                annotation_text="CPU = GPU (1×)",
                annotation_position="bottom right",
            )

            _fig.update_layout(
                title=dict(
                    text="Speedup GPU vs CPU (maior = melhor)",
                    font=dict(size=15),
                ),
                yaxis=dict(
                    title="Vezes mais rápida (GPU/CPU)",
                    gridcolor="rgba(0,0,0,0.1)",
                ),
                template="plotly_white",
                height=430,
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False,
            )

            _resultado = mo.ui.plotly(_fig)
        else:
            _resultado = mo.md("⚠️ Dados insuficientes para comparar GPU vs CPU.")
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.5 WER (Word Error Rate)

    Taxa de erro a nível de palavra. Complementa o CER plotado em 11.2.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, go, mo):
    _nomes = list(dfs_disponiveis.keys())
    _wer = [dfs_disponiveis[n]["WER"].mean() * 100 for n in _nomes]

    if not _nomes:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _fig = go.Figure()

        _fig.add_trace(
            go.Bar(
                x=_nomes, y=_wer,
                text=[f"{v:.1f}" for v in _wer], textposition="outside",
                marker_color="#4C72B0", marker_line=dict(color="white", width=0.8),
                hovertemplate="<b>%{x}</b><br>WER: %{y:.1f}%<extra></extra>",
            )
        )

        _fig.update_layout(
            title=dict(
                text="WER — Word Error Rate (menor = melhor)", font=dict(size=15)
            ),
            yaxis=dict(title="%", gridcolor="rgba(0,0,0,0.1)"),
            template="plotly_white",
            height=430,
            margin=dict(l=20, r=20, t=50, b=80),
            showlegend=False,
        )

        _resultado = mo.ui.plotly(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.6 Heatmap de Métricas

    Visão panorâmica de todos os modelos × métricas de acurácia.
    Verde = melhor, vermelho = pior.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, go, mo, np):
    _nomes = list(dfs_disponiveis.keys())
    _metricas_data = []

    for _nome in _nomes:
        _df = dfs_disponiveis[_nome]
        _metricas_data.append(
            {
                "Word Acc (%)": _df["Accuracy"].mean() * 100,
                "Acc @80% (%)": _df["Accuracy80"].mean() * 100,
                "Similaridade": _df["Similaridade"].mean() * 100,
                "CER (%)": _df["CER"].mean() * 100,
                "WER (%)": _df["WER"].mean() * 100,
                "Lev. Médio": _df["Levenshtein"].mean(),
                "Tempo (s)": _df["Tempo_Inferencia"].mean(),
                "VRAM (MB)": _df["VRAM_Pico_MB"].mean(),
            }
        )

    if not _nomes:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _metricas_nomes = [
            "Word Acc (%)",
            "Acc @80% (%)",
            "Similaridade",
            "CER (%)",
            "WER (%)",
            "Lev. Médio",
            "Tempo (s)",
            "VRAM (MB)",
        ]
        _n_metrica = len(_metricas_nomes)
        _n_modelo = len(_nomes)

        _matriz = np.zeros((_n_modelo, _n_metrica))
        for _i, _md in enumerate(_metricas_data):
            for _j, _mn in enumerate(_metricas_nomes):
                _matriz[_i, _j] = _md[_mn]

        _matriz_norm = np.zeros_like(_matriz)
        _alta_melhor = [True, True, True, False, False, False, False, False]

        for _j in range(_n_metrica):
            _col = _matriz[:, _j]
            _min, _max = _col.min(), _col.max()
            if _max - _min < 1e-9:
                _matriz_norm[:, _j] = 0.5
            elif _alta_melhor[_j]:
                _matriz_norm[:, _j] = (_col - _min) / (_max - _min)
            else:
                _matriz_norm[:, _j] = (_max - _col) / (_max - _min)

        _texto_celulas = []
        for _i in range(_n_modelo):
            _linha = []
            for _j in range(_n_metrica):
                _val = _matriz[_i, _j]
                _fmt = f"{_val:.1f}" if _val < 10 else f"{_val:.0f}"
                _linha.append(_fmt)
            _texto_celulas.append(_linha)

        _fig = go.Figure(
            data=go.Heatmap(
                z=_matriz_norm,
                x=_metricas_nomes,
                y=_nomes,
                text=_texto_celulas,
                texttemplate="%{text}",
                textfont=dict(size=12, family="sans-serif"),
                colorscale=[
                    [0.0, "#d73027"],
                    [0.25, "#fc8d59"],
                    [0.5, "#fee08b"],
                    [0.75, "#91cf60"],
                    [1.0, "#1a9850"],
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
                text="Heatmap de Métricas (verde = melhor, vermelho = pior)",
                font=dict(size=15),
            ),
            xaxis=dict(title="Métrica", side="bottom", tickangle=45, tickfont=dict(size=10)),
            yaxis=dict(title="Modelo", tickfont=dict(size=12, family="sans-serif")),
            width=950,
            height=400,
            margin=dict(l=20, r=20, t=50, b=100),
            plot_bgcolor="white",
        )

        _resultado = mo.ui.plotly(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.7 Acurácia por Comprimento da Palavra

    O desempenho varia com o tamanho da palavra? Agrupa as palavras do
    gabarito por faixa de comprimento e plota a Word Accuracy em cada faixa.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, go, mo):
    _nomes = list(dfs_disponiveis.keys())

    if not _nomes:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _faixas = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 20)]
        _rotulos = [f"{a}-{b}" for a, b in _faixas]

        _fig = go.Figure()
        for _nome in _nomes:
            _df = dfs_disponiveis[_nome]
            _accs = []
            for _lo, _hi in _faixas:
                _mask = _df["Gabarito"].astype(str).str.len().between(_lo, _hi)
                _accs.append(
                    _df.loc[_mask, "Accuracy"].mean() * 100 if _mask.sum() > 0 else 0
                )

            _fig.add_trace(
                go.Bar(
                    name=_nome,
                    x=_rotulos,
                    y=_accs,
                    hovertemplate=(
                        f"<b>{_nome}</b><br>Word Acc: %{{y:.1f}}%<br>"
                        "Faixa: %{x}<extra></extra>"
                    ),
                )
            )

        _fig.update_layout(
            barmode="group",
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=9),
            ),
            yaxis=dict(
                range=[0, 110], title="Word Accuracy (%)",
                gridcolor="rgba(0,0,0,0.1)",
            ),
            xaxis=dict(title="Comprimento da palavra (caracteres)"),
        )

        _resultado = mo.ui.plotly(_fig)
    _resultado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.8 Acurácia por Nível (Palavras / Frases / Textos)

    Compara a Word Accuracy de cada motor em cada nível de segmentação do
    BRESSAY. Permite enxergar, por exemplo, se um modelo é forte em palavras
    isoladas mas fraco em frases ou textos longos.
    """)
    return


@app.cell(hide_code=True)
def _(dfs_disponiveis, go, mo, pd):
    if not dfs_disponiveis:
        _resultado = mo.md("⚠️ Nenhum dado para comparar.")
    else:
        _registros = []
        for _nome, _df in dfs_disponiveis.items():
            for _tipo, _sub in _df.groupby("Tipo"):
                _registros.append(
                    {
                        "Motor": _nome,
                        "Nível": _tipo,
                        "Word Acc (%)": _sub["Accuracy"].mean() * 100,
                    }
                )
        _df_plot = pd.DataFrame(_registros)
        _niveis = ["palavras", "frases", "textos"]

        _fig = go.Figure()
        for _motor in _df_plot["Motor"].unique():
            _sub = _df_plot[_df_plot["Motor"] == _motor]
            _accs = [
                _sub.loc[_sub["Nível"] == _nivel, "Word Acc (%)"].iloc[0]
                if (_sub["Nível"] == _nivel).any()
                else 0
                for _nivel in _niveis
            ]
            _fig.add_trace(
                go.Bar(
                    name=_motor,
                    x=_niveis,
                    y=_accs,
                    text=[f"{v:.1f}" for v in _accs],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        f"<b>{_motor}</b><br>Word Acc: %{{y:.1f}}%<extra></extra>"
                    ),
                )
            )

        _fig.update_layout(
            barmode="group",
            template="plotly_white",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            yaxis=dict(
                range=[0, 110],
                title="Word Accuracy (%)",
                gridcolor="rgba(0,0,0,0.1)",
            ),
            xaxis=dict(title="Nível de segmentação"),
        )

        _resultado = mo.ui.plotly(_fig)
    _resultado
    return


if __name__ == "__main__":
    app.run()
