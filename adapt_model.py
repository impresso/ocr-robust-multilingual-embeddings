#!/usr/bin/env python3
"""
adapt_model.py – Fine-tune a sentence-transformer on user-specified data sources.

Usage:
    python adapt_model.py config.json

The JSON config has the following structure:

{
    "model_name": "Alibaba-NLP/gte-multilingual-base",
    "output_dir": "trained_models/my_model",
    "batch_size": 8,
    "batches_per_epoch": 3000,
    "epochs": 1,
    "lr": 2e-5,
    "warmup_ratio": 0.1,
    "seed": 42,
    "stages": [
        {
            "name": "cross_lingual_alignment",
            "sources": [
                {
                    "path": "noisy_finetuning_data/LREC/lb_de_training_set.jsonl",
                    "format": "jsonl",
                    "column_a": "lb",
                    "column_b": "de",
                    "max_samples": null
                }
            ]
        },
        {
            "name": "ocr_noise_robustness",
            "sources": [
                {
                    "path": "noisy_finetuning_data/ACL/TED_data_random_noise_concat.csv",
                    "format": "csv",
                    "column_a": "deu",
                    "column_b": "deu_04",
                    "max_samples": 50000
                }
            ]
        }
    ]
}

Source format types:
    - "csv":   reads column_a and column_b from a CSV file.
    - "jsonl": reads JSONL with a "translation" array of {column_a: ..., column_b: ...} pairs.

Each stage trains sequentially; the next stage loads the checkpoint from the previous one.
"""

import argparse
import json
import os
import random
import sys

import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)


def load_pairs_csv(path, col_a, col_b, max_samples=None):
    """Load (col_a, col_b) pairs from a CSV file."""
    df = pd.read_csv(path).dropna(subset=[col_a, col_b])
    pairs = list(zip(df[col_a].astype(str), df[col_b].astype(str)))
    if max_samples and len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    return pairs


def load_pairs_jsonl(path, col_a, col_b, max_samples=None):
    """Load (col_a, col_b) pairs from a JSONL file with a 'translation' array."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for t in obj["translation"]:
                pairs.append((str(t[col_a]), str(t[col_b])))
    if max_samples and len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    return pairs


LOADERS = {
    "csv": load_pairs_csv,
    "jsonl": load_pairs_jsonl,
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_pairs(sources, seed):
    """Collect all (text_a, text_b) pairs from a list of source definitions."""
    random.seed(seed)
    all_pairs = []
    for src in sources:
        fmt = src.get("format", "csv")
        loader = LOADERS.get(fmt)
        if loader is None:
            print(f"[ERROR] Unknown format '{fmt}' for {src['path']}", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(src["path"]):
            print(f"[ERROR] File not found: {src['path']}", file=sys.stderr)
            sys.exit(1)

        max_samples = src.get("max_samples")
        pairs = loader(src["path"], src["column_a"], src["column_b"], max_samples)
        print(f"  {src['path']}: {len(pairs):,} pairs (columns: {src['column_a']}, {src['column_b']})")
        all_pairs.extend(pairs)
    return all_pairs


def sample_dataset(pairs, n, seed):
    """Sample n pairs with replacement and return a HuggingFace Dataset."""
    rng = random.Random(seed)
    sampled = [pairs[rng.randrange(len(pairs))] for _ in range(n)]
    return HFDataset.from_dict({
        "anchor": [a for a, b in sampled],
        "positive": [b for a, b in sampled],
    })


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a sentence-transformer from a JSON config.")
    parser.add_argument("config", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg.get("model_name", "Alibaba-NLP/gte-multilingual-base")
    output_dir = cfg.get("output_dir", "trained_models/adapted_model")
    batch_size = cfg.get("batch_size", 8)
    batches_per_epoch = cfg.get("batches_per_epoch", 3000)
    examples_per_epoch = batch_size * batches_per_epoch
    epochs = cfg.get("epochs", 1)
    lr = cfg.get("lr", 2e-5)
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    seed = cfg.get("seed", 42)
    stages = cfg.get("stages", [])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Model:          {model_name}")
    print(f"Output:         {output_dir}")
    print(f"Batch size:     {batch_size}")
    print(f"Examples/epoch: {examples_per_epoch:,}")
    print(f"Epochs:         {epochs}")
    print(f"Learning rate:  {lr}")
    print(f"Stages:         {len(stages)}")
    print()

    current_model_path = model_name

    for i, stage in enumerate(stages):
        stage_name = stage.get("name", f"stage_{i}")
        stage_output = os.path.join(output_dir, stage_name)
        sources = stage.get("sources", [])

        print(f"{'='*60}")
        print(f"Stage {i+1}: {stage_name}")
        print(f"{'='*60}")

        pairs = collect_pairs(sources, seed)
        if not pairs:
            print(f"[WARN] No pairs for stage '{stage_name}', skipping.")
            continue

        print(f"  Total pairs: {len(pairs):,}")

        dataset = sample_dataset(pairs, examples_per_epoch, seed)

        model = SentenceTransformer(current_model_path, trust_remote_code=True)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=stage_output,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_steps=warmup_ratio,
            learning_rate=lr,
            seed=seed,
            logging_steps=100,
            save_strategy="no",
            report_to="none",
        )

        print(f"  Dataset: {len(dataset):,} examples, batch_size: {batch_size}")
        print()

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            loss=train_loss,
        )

        trainer.train()
        model.save_pretrained(stage_output)

        current_model_path = stage_output
        print(f"  Checkpoint saved to {stage_output}\n")

    # Copy final stage to output root for convenience
    final_path = os.path.join(output_dir, "final")
    if current_model_path != model_name:
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        shutil.copytree(current_model_path, final_path)
        print(f"Final model saved to {final_path}")
    else:
        print("No stages were trained.")


if __name__ == "__main__":
    main()
