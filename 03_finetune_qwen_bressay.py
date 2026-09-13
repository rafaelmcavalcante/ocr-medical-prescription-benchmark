# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate>=1.14.0",
#     "gdown>=5.2.0",
#     "marimo>=0.23.14",
#     "numpy>=2.2.6",
#     "pandas>=2.3.3",
#     "peft>=0.14.0",
#     "pillow>=12.3.0",
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
def _():
    import os
    import re
    import urllib.request
    import zipfile
    from types import SimpleNamespace

    import gdown
    import pandas as pd
    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    return (
        AutoModelForMultimodalLM,
        AutoProcessor,
        Image,
        LoraConfig,
        SimpleNamespace,
        Trainer,
        TrainingArguments,
        gdown,
        get_peft_model,
        os,
        pd,
        re,
        torch,
        urllib,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fine-tuning do Qwen3-VL no BRESSAY (LoRA)

    Este notebook faz **LoRA fine-tuning** do `Qwen/Qwen3-VL-8B-Instruct` no
    dataset BRESSAY e salva o modelo ajustado em `qwen3vl_ft_bressay/`, para ser
    comparado no notebook `02_comparacao_ocr_bressay.py`.

    **Estratégia** (pensada para a RTX 6000 Blackwell ~96 GB do molab):

    - LoRA no LLM, **vision tower congelado** → treino rápido e leve;
    - treina no split `training.txt`, valida em `validation.txt`, e o
      `test.txt` (usado no benchmark do notebook 02) **não é visto** no treino;
    - gabaritos passam pela mesma limpeza de anotações do BRESSAY usada no
      notebook 02 (remove `@@???@@`, `--riscado--`, desembrulha `##texto##`);
    - ao final, o adaptador é **mergeado** e salvo como modelo completo, para o
      notebook 02 carregar sem depender do `peft`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 Configuração

    Ajuste `AMOSTRAS` conforme o tempo disponível. Um total de ~30 mil amostras
    é suficiente para um experimento rápido; o split de treino completo tem
    ~290 mil.
    """)
    return


@app.cell
def _(SimpleNamespace):
    CFG = SimpleNamespace(
        MODELO_BASE="Qwen/Qwen3-VL-8B-Instruct",
        DATASET_DIR="bressay",
        DIR_SAIDA="qwen3vl_ft_bressay",
        DIR_CKPT="qwen3vl_ft_bressay_ckpt",
        # quantas amostras de cada nível usar no treino
        AMOSTRAS={"palavras": 15000, "frases": 10000, "textos": 5000},
        N_VALIDACAO=300,
        PROMPT=(
            "Transcreva fielmente o texto manuscrito nesta imagem. "
            "É uma redação em português brasileiro (pode ser uma palavra, uma "
            "frase ou um parágrafo). Retorne apenas o texto transcrito, sem "
            "explicações."
        ),
        MAX_LEN=2048,
        EPOCHS=1,
        BATCH=8,
        GRAD_ACCUM=4,
        LR=1e-4,
        LORA_R=16,
        LORA_ALPHA=32,
        LORA_DROPOUT=0.05,
        SEED=42,
        FORCAR_RETREINO=False,
    )
    return (CFG,)


@app.cell(hide_code=True)
def _(CFG, gdown, mo, os, urllib, zipfile):
    # ── Download do dataset BRESSAY (se necessário) ──
    # Mesmo procedimento usado no notebook 02_comparacao_ocr_bressay.py.
    DATASET_URL = "https://drive.google.com/file/d/1XACLMLMLuMs_6EpNaOD8nd-Nn5X9T7eH/view?usp=sharing"

    if not os.path.exists(CFG.DATASET_DIR):
        zip_path = "bressay.zip"
        if not os.path.exists(zip_path):
            print("Baixando dataset BRESSAY...")
            if "drive.google.com" in DATASET_URL:
                gdown.download(DATASET_URL, zip_path, quiet=False)
            else:
                urllib.request.urlretrieve(DATASET_URL, zip_path)

        # Extrai na raiz (o zip já contém a pasta bressay/ internamente)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(".")
        _msg_dataset = "Dataset BRESSAY baixado e extraído!"
    else:
        _msg_dataset = "Dataset BRESSAY já existe localmente."
    mo.md(_msg_dataset)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 Dataset de treino/validação
    """)
    return


@app.cell
def _(CFG, mo, os, pd, re):
    def limpar_anotacoes(texto: str) -> str:
        """Mesma limpeza de anotações usada no notebook 02."""
        t = str(texto)
        t = re.sub(r"##@@\?\?\?@@##", " ", t)
        t = re.sub(r"\$\$@@\?\?\?@@\$\$", " ", t)
        t = re.sub(r"@@\?\?\?@@", " ", t)
        t = re.sub(r"##--.+?--##", " ", t)
        t = re.sub(r"\$\$--.+?--\$\$", " ", t)
        t = re.sub(r"--.+?--", " ", t)
        t = re.sub(r"##([^#]+)##", r"\1", t)
        t = re.sub(r"\$\$([^$]+)\$\$", r"\1", t)
        t = t.replace("##", " ").replace("$$", " ")
        t = re.sub(r"--+", " ", t)
        return re.sub(r"\s+", " ", t).strip().lower()

    TIPOS = {"palavras": "words", "frases": "lines", "textos": "paragraphs"}

    def carregar_split(nome_arquivo: str, limite: dict | None = None) -> pd.DataFrame:
        caminho = os.path.join(CFG.DATASET_DIR, "sets", nome_arquivo)
        paginas = [l.strip() for l in open(caminho, encoding="utf-8") if l.strip()]
        registros = []
        for nome, pasta in TIPOS.items():
            raiz = os.path.join(CFG.DATASET_DIR, "data", pasta)
            itens = []
            for pag in paginas:
                pasta_pagina = os.path.join(raiz, pag)
                if not os.path.isdir(pasta_pagina):
                    continue
                for arquivo in os.listdir(pasta_pagina):
                    if not arquivo.endswith(".png"):
                        continue
                    caminho_txt = os.path.join(
                        pasta_pagina, arquivo.replace(".png", ".txt")
                    )
                    if not os.path.exists(caminho_txt):
                        continue
                    texto = limpar_anotacoes(open(caminho_txt, encoding="utf-8").read())
                    if texto:
                        itens.append(
                            {
                                "Tipo": nome,
                                "Images": os.path.join(pasta_pagina, arquivo),
                                "Text": texto,
                            }
                        )
            if limite and nome in limite and len(itens) > limite[nome]:
                itens = (
                    pd.DataFrame(itens)
                    .sample(n=limite[nome], random_state=CFG.SEED)
                    .to_dict("records")
                )
            registros.extend(itens)
        return (
            pd.DataFrame(registros)
            .sample(frac=1, random_state=CFG.SEED)
            .reset_index(drop=True)
        )

    df_treino = carregar_split("training.txt", CFG.AMOSTRAS)
    _n_por_nivel = max(CFG.N_VALIDACAO // len(TIPOS), 1)
    df_valid = carregar_split("validation.txt", {k: _n_por_nivel for k in TIPOS})
    mo.md(
        f"Treino: **{len(df_treino)}** amostras | "
        f"Validação: **{len(df_valid)}** amostras"
    )
    return df_treino, df_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 Modelo + LoRA

    Congelamos **todos** os pesos (incluindo o vision tower) e treinamos apenas
    os adaptadores LoRA nas projeções do LLM. Isso reduz muito a VRAM e o tempo.
    """)
    return


@app.cell
def _(AutoModelForMultimodalLM, AutoProcessor, CFG, LoraConfig, get_peft_model, mo, torch):
    processor = AutoProcessor.from_pretrained(CFG.MODELO_BASE)
    model = AutoModelForMultimodalLM.from_pretrained(
        CFG.MODELO_BASE, dtype=torch.bfloat16
    )
    model.to("cuda")

    for p in model.parameters():
        p.requires_grad = False

    config_lora = LoraConfig(
        r=CFG.LORA_R,
        lora_alpha=CFG.LORA_ALPHA,
        lora_dropout=CFG.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, config_lora)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False

    _treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in model.parameters())
    mo.md(
        f"LoRA aplicado — treináveis: **{_treinaveis / 1e6:.1f}M** / "
        f"{_total / 1e9:.2f}B ({100 * _treinaveis / _total:.2f}%)"
    )
    return model, processor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Dataset PyTorch e collator

    O gabarito é o texto limpo. As máscaras `-100` cobrem o prompt (system/user +
    tokens do template), de modo que a loss é calculada **apenas na resposta**.
    """)
    return


@app.cell
def _(CFG, Image, df_treino, df_valid, model, processor, torch):
    def _ultima_ocorrencia(ids, alvo):
        for i in range(len(ids) - len(alvo), -1, -1):
            if ids[i : i + len(alvo)] == alvo:
                return i
        return -1

    _MARCADOR = processor.tokenizer.encode(
        "<|im_start|>assistant", add_special_tokens=False
    )

    class BressayDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            linha = self.df.iloc[i]
            imagem = Image.open(linha["Images"]).convert("RGB")
            mensagens = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": imagem},
                        {"type": "text", "text": CFG.PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": linha["Text"]}],
                },
            ]
            enc = processor.apply_chat_template(
                mensagens, tokenize=True, return_dict=True, return_tensors="pt"
            )

            input_ids = enc["input_ids"][0][: CFG.MAX_LEN]
            labels = input_ids.clone()
            pos = _ultima_ocorrencia(input_ids.tolist(), _MARCADOR)
            if pos >= 0:
                labels[: pos + len(_MARCADOR)] = -100

            item = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            }
            if "pixel_values" in enc:
                item["pixel_values"] = enc["pixel_values"][0]
            if "image_grid_thw" in enc:
                item["image_grid_thw"] = enc["image_grid_thw"][0]
            if "mm_token_type_ids" in enc:
                item["mm_token_type_ids"] = enc["mm_token_type_ids"][0][
                    : input_ids.size(0)
                ]
            return item

    def collate_fn(batch):
        maxlen = max(b["input_ids"].size(0) for b in batch)
        pad_id = processor.tokenizer.pad_token_id or 0

        def _pad(seqs, valor):
            out = torch.full((len(seqs), maxlen), valor, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : s.size(0)] = s
            return out

        out = {
            "input_ids": _pad([b["input_ids"] for b in batch], pad_id),
            "attention_mask": _pad([b["attention_mask"] for b in batch], 0),
            "labels": _pad([b["labels"] for b in batch], -100),
        }
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.cat(
                [b["pixel_values"] for b in batch], dim=0
            )
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat(
                [b["image_grid_thw"] for b in batch], dim=0
            )
        if "mm_token_type_ids" in batch[0]:
            out["mm_token_type_ids"] = _pad(
                [b["mm_token_type_ids"] for b in batch], 0
            )
        return out

    ds_treino = BressayDataset(df_treino)
    ds_valid = BressayDataset(df_valid)
    mo.md(
        f"Datasets prontos — treino **{len(ds_treino)}**, validação **{len(ds_valid)}**"
    )
    return collate_fn, ds_treino, ds_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 Treino
    """)
    return


@app.cell
def _(CFG, TrainingArguments, Trainer, collate_fn, ds_treino, ds_valid, model, mo):
    args = TrainingArguments(
        output_dir=CFG.DIR_CKPT,
        per_device_train_batch_size=CFG.BATCH,
        per_device_eval_batch_size=CFG.BATCH,
        gradient_accumulation_steps=CFG.GRAD_ACCUM,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LR,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        # 0 = sem multiprocessing. Com CUDA, o DataLoader usa "spawn" e o
        # worker não consegue importar BressayDataset (definida em célula do
        # marimo, em __marimo__cell_*.py). Se quiser workers, mova a classe
        # BressayDataset para um módulo .py separado e importe-o.
        dataloader_num_workers=0,
        seed=CFG.SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_treino,
        eval_dataset=ds_valid,
        data_collator=collate_fn,
    )
    mo.md(
        f"Trainer configurado — batch efetivo **{CFG.BATCH * CFG.GRAD_ACCUM}**, "
        f"**{CFG.EPOCHS}** época(s), LR **{CFG.LR}**."
    )
    return (trainer,)


@app.cell
def _(CFG, mo, os, trainer):
    if CFG.FORCAR_RETREINO or not os.path.isdir(CFG.DIR_SAIDA):
        resultado_treino = trainer.train()
        modelo_treinado = True
        _msg = (
            f"Treino concluído — loss final "
            f"**{resultado_treino.training_loss:.4f}**."
        )
    else:
        modelo_treinado = False
        _msg = (
            f"Modelo já existe em `{CFG.DIR_SAIDA}` — treino pulado. "
            "Use `FORCAR_RETREINO=True` para treinar de novo."
        )
    mo.md(_msg)
    return (modelo_treinado,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 Salvar o modelo fine-tuned

    O adaptador é mergeado no modelo base e salvo em `qwen3vl_ft_bressay/` no
    formato HF completo, pronto para o notebook 02 carregar.
    """)
    return


@app.cell
def _(CFG, model, modelo_treinado, mo, os, processor):
    if modelo_treinado:
        os.makedirs(CFG.DIR_SAIDA, exist_ok=True)
        modelo_merge = model.merge_and_unload()
        modelo_merge.save_pretrained(CFG.DIR_SAIDA, safe_serialization=True)
        processor.save_pretrained(CFG.DIR_SAIDA)
        _msg_save = f"Modelo merged salvo em `{CFG.DIR_SAIDA}/`."
    else:
        _msg_save = "Nada a salvar (modelo já existia ou treino não foi executado)."
    mo.md(_msg_save)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 Próximo passo

    Com o modelo salvo em `qwen3vl_ft_bressay/`, rode o notebook
    `02_comparacao_ocr_bressay.py`. A seção **9.9** detecta a pasta, executa o
    mesmo benchmark nas 500 amostras e gera
    `resultados_qwen_ft_gpu_bressay.csv`, que entra automaticamente nos painéis
    e gráficos comparativos.

    **Dicas de tempo/qualidade:**

    - Para um teste rápido, reduza `AMOSTRAS` (ex.: 5k/3k/1k) e rode 1 época.
    - Para qualidade, aumente `AMOSTRAS` e/ou `EPOCHS`, e considere
      `LORA_R=32`/`LORA_ALPHA=64`.
    - Se quiser treinar também o vision tower (mais lento), remova o
      congelamento em massa e adicione as projeções visuais aos
      `target_modules`.
    """)
    return


if __name__ == "__main__":
    app.run()
