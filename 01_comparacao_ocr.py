# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate>=1.14.0",
#     "jiwer>=4.0.0",
#     "marimo>=0.23.14",
#     "matplotlib>=3.10.9",
#     "numpy>=2.2.6",
#     "opencv-python>=5.0.0.93",
#     "paddleocr>=3.7.0",
#     "pandas>=2.3.3",
#     "pillow>=12.3.0",
#     "pytesseract>=0.3.13",
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
    import time
    import cv2
    import jiwer
    import Levenshtein
    import pandas as pd
    import pytesseract
    import torch

    from paddleocr import PaddleOCR
    from PIL import Image
    from transformers import (
        RobertaTokenizer,
        VisionEncoderDecoderModel,
        ViTImageProcessor,
    )

    return (
        Image,
        Levenshtein,
        PaddleOCR,
        RobertaTokenizer,
        ViTImageProcessor,
        VisionEncoderDecoderModel,
        cv2,
        jiwer,
        os,
        pd,
        pytesseract,
        time,
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
    return DIRETORIO_TESTE, VOCABULARIO, df_amostra


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

    return calcular_metricas, normalizar


@app.cell(hide_code=True)
def _(Levenshtein, normalizar):
    def corrigir_fuzzy(texto_ocr: str, vocabulario: list[str], score_minimo: float = 0.0) -> str:
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
    MODELO = "microsoft/trocr-large-handwritten"

    image_processor_trocr = ViTImageProcessor.from_pretrained(MODELO)
    tokenizer_trocr = RobertaTokenizer.from_pretrained(MODELO)
    model_trocr = VisionEncoderDecoderModel.from_pretrained(MODELO).to(device)

    mo.md(f"""
    ✅ **TrOCR** carregado

    Dispositivo: `{device}` | Modelo: `trocr-large-handwritten`
    """)
    return image_processor_trocr, model_trocr, tokenizer_trocr


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
            pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(model.device)
            generated_ids = model.generate(pixel_values, max_new_tokens=64)
            text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return normalizar(text), time.time() - start
        except Exception as e:
            print(f"⚠️  TrOCR [{os.path.basename(caminho_imagem)}]: {e}")
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
            print(f"⚠️  PaddleOCR [{os.path.basename(caminho_imagem)}]: {e}")
            return "", time.time() - start


    def predizer_tesseract(caminho_imagem):
        """Tesseract — engine clássica de OCR."""
        start = time.time()
        try:
            image = Image.open(caminho_imagem)
            text = pytesseract.image_to_string(image)
            return normalizar(text), time.time() - start
        except Exception as e:
            print(f"⚠️  Tesseract [{os.path.basename(caminho_imagem)}]: {e}")
            return "", time.time() - start

    return predizer_paddle, predizer_tesseract, predizer_trocr


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

            resultados.append({
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
            })

            if (idx + 1) % 50 == 0:
                print(f"  [{nome}] {idx + 1}/{total} imagens...")

        df_resultado = pd.DataFrame(resultados)
        df_resultado.to_csv(csv_saida, index=False, encoding="utf-8")
        print(f"[{nome}] ✅ {len(df_resultado)} amostras → {csv_saida}")
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
    NOME_CSV = "resultados_tesseract.csv"
    df_tesseract = rodar_benchmark_ocr(
        nome="Tesseract",
        predizer=predizer_tesseract,
        score_minimo=0.0,
        csv_saida=NOME_CSV,
        df=df_amostra,
        diretorio=DIRETORIO_TESTE,
        vocabulario=VOCABULARIO,
    )
    mo.md(f"✅ Tesseract: **{len(df_tesseract)}** amostras → `{NOME_CSV}`")
    return (df_tesseract,)


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
    predizer_paddle,
    rodar_benchmark_ocr,
):
    from functools import partial

    NOME_CSV = "resultados_paddle.csv"
    df_paddle = rodar_benchmark_ocr(
        nome="PaddleOCR",
        predizer=partial(predizer_paddle, ocr=ocr_paddle),
        score_minimo=0.75,
        csv_saida=NOME_CSV,
        df=df_amostra,
        diretorio=DIRETORIO_TESTE,
        vocabulario=VOCABULARIO,
    )
    mo.md(f"✅ PaddleOCR: **{len(df_paddle)}** amostras → `{NOME_CSV}`")
    return (df_paddle,)


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
    predizer_trocr,
    rodar_benchmark_ocr,
    tokenizer_trocr,
):
    from functools import partial

    NOME_CSV = "resultados_trocr.csv"
    df_trocr = rodar_benchmark_ocr(
        nome="TrOCR",
        predizer=partial(
            predizer_trocr,
            image_processor=image_processor_trocr,
            tokenizer=tokenizer_trocr,
            model=model_trocr,
        ),
        score_minimo=0.1,
        csv_saida=NOME_CSV,
        df=df_amostra,
        diretorio=DIRETORIO_TESTE,
        vocabulario=VOCABULARIO,
    )
    mo.md(f"✅ TrOCR: **{len(df_trocr)}** amostras → `{NOME_CSV}`")
    return (df_trocr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10 Resultados (Tabelas)

    As células abaixo carregam os CSVs disponíveis e montam as comparações.
    Rode os benchmarks que quiser (9.3–9.5) e depois venha aqui.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10.1 Amostra de Predições

    Primeiras linhas de cada CSV disponível.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os, pd):
    # Detecta quais CSVs de resultado existem
    CSVS_PADRAO = [
        ("Tesseract", "resultados_tesseract.csv"),
        ("PaddleOCR", "resultados_paddle.csv"),
        ("TrOCR", "resultados_trocr.csv"),
    ]

    dfs_disponiveis = {}
    for nome, csv_path in CSVS_PADRAO:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs_disponiveis[nome] = df

    if not dfs_disponiveis:
        mo.md("⚠️  Nenhum CSV de resultado encontrado. Rode as células 9.3–9.5 primeiro.")
    else:
        mo.md(f"📂 CSVs encontrados: **{', '.join(dfs_disponiveis.keys())}**")

        for nome, df in dfs_disponiveis.items():
            mo.md(f"### {nome}")
            mo.ui.table(
                df[["Arquivo", "Gabarito", "Predicao_Raw", "Predicao_Fuzzy"]].head(5)
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
        mo.md("⚠️  Nenhum dado para comparar.")
    else:
        linhas = []
        for nome, df in dfs_disponiveis.items():
            for variante in ("Raw", "Fuzzy"):
                linhas.append({
                    "Motor": nome,
                    "Variante": variante,
                    "Word Acc (%)": fmt_pct(df[f"Accuracy_{variante}"].mean()),
                    "Acc @80% (%)": fmt_pct(df[f"Accuracy80_{variante}"].mean()),
                    "Lev Médio": fmt_abs(df[f"Levenshtein_{variante}"].mean()),
                    "CER (%)": fmt_pct(df[f"CER_{variante}"].mean()),
                    "WER (%)": fmt_pct(df[f"WER_{variante}"].mean()),
                    "Tempo Médio (s)": fmt_abs(df["Tempo_Inferencia"].mean()),
                })

        df_painel = pd.DataFrame(linhas)
        mo.ui.table(df_painel)
    return (df_painel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11 Resultados (Gráficos)
    """)
    return


if __name__ == "__main__":
    app.run()
