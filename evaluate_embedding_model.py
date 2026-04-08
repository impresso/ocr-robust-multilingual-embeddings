#!/usr/bin/env python3
"""
evaluate_embedding_model.py – Evaluate a sentence-transformer on all available benchmarks.

Runs three evaluation suites automatically based on file availability:

1. CLSD (Cross-Lingual Semantic Discrimination)
   - Clean: clean_evaluation_datasets/ACL/CLSD_wmt{2019,2021}_adversarial_dataset.csv
   - Noisy:  noisy_evaluation_datasets/ACL/CLSD_WMT{19,21}_{MN,BLDS,SNP}_noise.csv

2. STS (Semantic Textual Similarity)
   - clean_evaluation_datasets/ACL/sts17_*.csv

3. HISTLUX Bitext Mining (optional – large files, must be downloaded separately)
   - noisy_evaluation_datasets/ACL/bitext_mining_task_*.jsonl
   - Download from: https://drive.google.com/file/d/1B_na_iXXa5nNcfh8L7sNIln9hNkji0ad/view

Usage:
    python evaluate_embedding_model.py <model_name_or_path>
    python evaluate_embedding_model.py trained_models/adapted_model/final
    python evaluate_embedding_model.py Alibaba-NLP/gte-multilingual-base

Options:
    --skip-bitext   Skip bitext mining evaluation even if files are present.
    --skip-sts      Skip STS evaluation.
    --skip-clsd     Skip CLSD evaluation.
    --output FILE   Write results to a JSON file.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLEAN_EVAL_DIR = "clean_evaluation_datasets/ACL"
NOISY_EVAL_DIR = "noisy_evaluation_datasets/ACL"

CLSD_DATASETS = {
    "WMT19_clean": {
        "path": os.path.join(CLEAN_EVAL_DIR, "CLSD_wmt2019_adversarial_dataset.csv"),
        "src": "fra", "tgt": "deu",
    },
    "WMT21_clean": {
        "path": os.path.join(CLEAN_EVAL_DIR, "CLSD_wmt2021_adversarial_dataset.csv"),
        "src": "fra", "tgt": "deu",
    },
    "WMT19_MN": {
        "path": os.path.join(NOISY_EVAL_DIR, "CLSD_WMT19_MN_noise.csv"),
        "src": "fra_04", "tgt": "deu_04",
    },
    "WMT21_MN": {
        "path": os.path.join(NOISY_EVAL_DIR, "CLSD_WMT21_MN_noise.csv"),
        "src": "fra_04", "tgt": "deu_04",
    },
    "WMT19_BLDS": {
        "path": os.path.join(NOISY_EVAL_DIR, "CLSD_WMT19_BLDS_noise.csv"),
        "src": "fra_04", "tgt": "deu_04",
    },
    "WMT21_BLDS": {
        "path": os.path.join(NOISY_EVAL_DIR, "CLSD_WMT21_BLDS_noise.csv"),
        "src": "fra_04", "tgt": "deu_04",
    },
    "WMT19_SNP": {
        "path": os.path.join(NOISY_EVAL_DIR, "CLSD_WMT19_SNP_noise.csv"),
        "src": "fra_04", "tgt": "deu_04",
    },
    "WMT21_SNP": {
        "path": os.path.join(NOISY_EVAL_DIR, "CLSD_WMT21_SNP_noise.csv"),
        "src": "fra_04", "tgt": "deu_04",
    },
}

STS_DATASETS = {
    "sts17_ar-en": {"path": os.path.join(CLEAN_EVAL_DIR, "sts17_ar-en.csv"), "col_a": "eng", "col_b": "ara"},
    "sts17_en-es": {"path": os.path.join(CLEAN_EVAL_DIR, "sts17_en-es.csv"), "col_a": "eng", "col_b": "spa"},
    "sts17_es-en": {"path": os.path.join(CLEAN_EVAL_DIR, "sts17_es-en.csv"), "col_a": "spa", "col_b": "eng"},
    "sts17_tr-en": {"path": os.path.join(CLEAN_EVAL_DIR, "sts17_tr-en.csv"), "col_a": "eng", "col_b": "tur"},
}

BITEXT_PAIRS = [
    ("de_to_lb", "German → Luxembourgish"),
    ("lb_to_de", "Luxembourgish → German"),
    ("en_to_lb", "English → Luxembourgish"),
    ("lb_to_en", "Luxembourgish → English"),
    ("fr_to_lb", "French → Luxembourgish"),
    ("lb_to_fr", "Luxembourgish → French"),
]


# ---------------------------------------------------------------------------
# CLSD Evaluation
# ---------------------------------------------------------------------------
def clsd_accuracy(model, csv_path, src_col, tgt_col):
    """Compute CLSD retrieval accuracy (source → target)."""
    df = pd.read_csv(csv_path).dropna(subset=[src_col, tgt_col])
    src = df[src_col].astype(str).tolist()
    tgt = df[tgt_col].astype(str).tolist()

    src_emb = model.encode(src, convert_to_tensor=True, show_progress_bar=False)
    tgt_emb = model.encode(tgt, convert_to_tensor=True, show_progress_bar=False)

    sims = st_util.cos_sim(src_emb, tgt_emb)
    preds = sims.argmax(dim=1).cpu()
    gold = torch.arange(len(src))
    return (preds == gold).float().mean().item()


def evaluate_clsd(model):
    """Run CLSD evaluation on all available datasets."""
    results = {}
    available = {k: v for k, v in CLSD_DATASETS.items() if os.path.exists(v["path"])}

    if not available:
        print("  No CLSD evaluation files found. Skipping.")
        return results

    print(f"\n{'Dataset':<20} {'Accuracy':>10}")
    print(f"{'-'*20} {'-'*10}")

    for name, info in available.items():
        acc = clsd_accuracy(model, info["path"], info["src"], info["tgt"])
        results[name] = round(acc, 4)
        print(f"{name:<20} {acc:>10.4f}")

    return results


# ---------------------------------------------------------------------------
# STS Evaluation
# ---------------------------------------------------------------------------
def sts_spearman(model, csv_path, col_a, col_b):
    """Compute Spearman correlation between cosine similarity and gold scores."""
    df = pd.read_csv(csv_path).dropna(subset=[col_a, col_b, "similarity_score"])
    sents_a = df[col_a].astype(str).tolist()
    sents_b = df[col_b].astype(str).tolist()
    gold = df["similarity_score"].values

    emb_a = model.encode(sents_a, convert_to_tensor=True, show_progress_bar=False)
    emb_b = model.encode(sents_b, convert_to_tensor=True, show_progress_bar=False)

    cos_scores = torch.nn.functional.cosine_similarity(emb_a, emb_b).cpu().numpy()
    corr, _ = spearmanr(cos_scores, gold)
    return corr


def evaluate_sts(model):
    """Run STS evaluation on all available datasets."""
    results = {}
    available = {k: v for k, v in STS_DATASETS.items() if os.path.exists(v["path"])}

    if not available:
        print("  No STS evaluation files found. Skipping.")
        return results

    print(f"\n{'Dataset':<20} {'Spearman':>10}")
    print(f"{'-'*20} {'-'*10}")

    for name, info in available.items():
        corr = sts_spearman(model, info["path"], info["col_a"], info["col_b"])
        results[name] = round(corr, 4)
        print(f"{name:<20} {corr:>10.4f}")

    if results:
        avg = np.mean(list(results.values()))
        results["average"] = round(avg, 4)
        print(f"{'-'*20} {'-'*10}")
        print(f"{'Average':<20} {avg:>10.4f}")

    return results


# ---------------------------------------------------------------------------
# Bitext Mining Evaluation
# ---------------------------------------------------------------------------
def load_bitext(path):
    """Load bitext mining JSONL dataset."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def bitext_precision_at_1(model, bitext_data):
    """Compute Precision@1 for bitext mining."""
    unique = set(
        s for entry in bitext_data
        for s in [entry["source_sentence"]] + entry["candidates"]
    )
    embs = {s: model.encode(s) for s in unique}

    correct = 0
    for entry in bitext_data:
        src_emb = embs[entry["source_sentence"]]
        gold_emb = embs[entry["candidates"][0]]
        gold_sim = np.dot(src_emb, gold_emb) / (
            np.linalg.norm(src_emb) * np.linalg.norm(gold_emb)
        )

        distractor_embs = np.array([embs[c] for c in entry["candidates"][1:]])
        distractor_sims = 1 - cdist(
            distractor_embs, src_emb.reshape(1, -1), metric="cosine"
        ).flatten()

        if np.max(distractor_sims) < gold_sim:
            correct += 1

    return round(correct / len(bitext_data) * 100, 2)


def evaluate_bitext(model):
    """Run bitext mining evaluation on all available JSONL files."""
    results = {}
    available = []

    for key, label in BITEXT_PAIRS:
        path = os.path.join(NOISY_EVAL_DIR, f"bitext_mining_task_{key}.jsonl")
        if os.path.exists(path):
            available.append((key, label, path))

    if not available:
        print("\n  No bitext mining JSONL files found.")
        print("  Download from: https://drive.google.com/file/d/1B_na_iXXa5nNcfh8L7sNIln9hNkji0ad/view")
        print(f"  Place files in {NOISY_EVAL_DIR}/")
        return results

    print(f"\n{'Direction':<30} {'P@1 (%)':>10}")
    print(f"{'-'*30} {'-'*10}")

    scores = []
    for key, label, path in available:
        data = load_bitext(path)
        p1 = bitext_precision_at_1(model, data)
        results[key] = p1
        scores.append(p1)
        print(f"{label:<30} {p1:>10.2f}")

    if scores:
        avg = round(np.mean(scores), 2)
        results["average"] = avg
        print(f"{'-'*30} {'-'*10}")
        print(f"{'Average':<30} {avg:>10.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a sentence-transformer on CLSD, STS, and bitext mining benchmarks."
    )
    parser.add_argument("model", help="Model name or path (e.g. trained_models/adapted_model/final).")
    parser.add_argument("--skip-clsd", action="store_true", help="Skip CLSD evaluation.")
    parser.add_argument("--skip-sts", action="store_true", help="Skip STS evaluation.")
    parser.add_argument("--skip-bitext", action="store_true", help="Skip bitext mining evaluation.")
    parser.add_argument("--output", help="Write results to a JSON file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model:  {args.model}")
    print(f"Device: {device}")

    model = SentenceTransformer(args.model, trust_remote_code=True).to(device)

    all_results = {"model": args.model}

    if not args.skip_clsd:
        print("\n" + "=" * 40)
        print("CLSD Evaluation")
        print("=" * 40)
        all_results["clsd"] = evaluate_clsd(model)

    if not args.skip_sts:
        print("\n" + "=" * 40)
        print("STS Evaluation")
        print("=" * 40)
        all_results["sts"] = evaluate_sts(model)

    if not args.skip_bitext:
        print("\n" + "=" * 40)
        print("Bitext Mining Evaluation")
        print("=" * 40)
        all_results["bitext_mining"] = evaluate_bitext(model)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")
    return all_results


if __name__ == "__main__":
    main()
