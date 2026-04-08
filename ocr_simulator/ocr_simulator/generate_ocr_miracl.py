# ============================================================================
# MIRACL — Generate OCR-noised texts
# ============================================================================
#
# Generates 2 .pkl files per language:
#   {lang}_{condition}_corpus.pkl
#   {lang}_{condition}_queries.pkl
#
# Prerequisites:
#   Windows:
#     1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
#        - During install, check additional language packs: deu, fra, spa, rus
#   macOS:
#     1. brew install tesseract tesseract-lang
#   Linux:
#     1. sudo apt install tesseract-ocr tesseract-ocr-deu tesseract-ocr-fra ...
#
#   All platforms:
#     2. pip install pytesseract Pillow datasets tqdm jiwer
#     3. pip install -e path/to/ocr-robust-multilingual-embeddings/ocr_simulator
#
# Usage:
#   python miracl_generate_ocrsim.py                    # all languages, 4 workers
#   python miracl_generate_ocrsim.py de,fr 4            # specific langs + workers
#   python miracl_generate_ocrsim.py es,ru 4            # run in parallel terminal
# ============================================================================

import os
import sys
import platform
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re, gc, time, pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from jiwer import cer as compute_cer
from typing import Dict, Optional
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pytesseract

# ======================= CROSS-PLATFORM CONFIGURATION =======================

def _find_tesseract() -> str:
    """Auto-detect Tesseract binary across platforms."""
    _system = platform.system()
    candidates = []
    if _system == "Windows":
        candidates = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Tesseract-OCR", "tesseract.exe"),
        ]
    elif _system == "Darwin":
        candidates = [
            "/opt/homebrew/bin/tesseract",    # Apple Silicon
            "/usr/local/bin/tesseract",       # Intel Mac (Homebrew)
        ]
    else:
        candidates = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    found = shutil.which("tesseract")
    return found or "tesseract"

TESSERACT_PATH = _find_tesseract()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
print(f"✓ Tesseract: {TESSERACT_PATH}")

try:
    available_langs = pytesseract.get_languages()
    print(f"  Available languages: {available_langs}")
except Exception as e:
    print(f"  ⚠ Could not query Tesseract: {e}")

# ── OCR simulator (languages.py already has cross-platform font resolution) ──
from ocr_simulator import OCRSimulator
from ocr_simulator.languages import LANGUAGE_CONFIGS

MLDR_TO_TESSERACT = {"ar": "ara", "de": "deu", "fr": "fra", "es": "spa", "ru": "rus", "en": "eng"}

# MIRACL uses language keys directly as configs for selected languages
MIRACL_LANG_MAP = {
    "ar": "ar",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "ru": "ru",
    "en": "en",
}

# ── Sanity check ──
print("\n--- Sanity checks ---")
for lang3, sent in {
    "deu": "Dies ist ein Beispieltext.", "fra": "Ceci est un exemple.",
    "spa": "Esta es una oración.", "rus": "Это пример текста.",
    "eng": "This is a sample sentence.",
}.items():
    try:
        sim = OCRSimulator(condition="distorted", language=lang3, font_size=14, dpi=300)
        res = sim.process_single_text(sent)
        print(f"  [{lang3}] ✓  → {(res['ocr_text'] or '<EMPTY>')[:50]}")
    except Exception as e:
        print(f"  [{lang3}] ✗  {str(e)[:60]}")


# ======================= CONFIGURATION =======================
OUTPUT_DIR = "./miracl_noised_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Defaults — overridden by command line args if provided
EVAL_LANGUAGES = ["ar", "es", "ru", "en"]
# EVAL_LANGUAGES = ["ar", "de", "fr", "es", "ru", "en"]
OCR_N_WORKERS = 4

# Command line: python script.py de,fr 4
if len(sys.argv) > 1:
    EVAL_LANGUAGES = sys.argv[1].split(",")
if len(sys.argv) > 2:
    OCR_N_WORKERS = int(sys.argv[2])

EVAL_SPLIT = "dev"
OCR_CONDITION = "distorted"

CORPUS_SAMPLE_SIZE_PER_LANG = {}
CORPUS_SAMPLE_SIZE_DEFAULT: Optional[int] = 2000

OCR_FONT_SIZE = 12
OCR_DPI = 300
OCR_MAX_TEXT_LENGTH = None    # None = no truncation

# ── Configurations to run: (dpi, font_size) ──
RUN_CONFIGS = [
    (300, 12),
    # (130, 10),
    # (120, 10),
]

# ======================= SENTENCE SPLITTING =======================

def split_sentences(text):
    """Split text into sentences on .!? followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p for p in parts if p.strip()]


# ======================= OCR NOISE (SENTENCE LEVEL) =======================

def _ocr_one_sentence(args):
    """Process a single sentence in a child process."""
    sentence, condition, language_3, font_size, dpi, tesseract_path = args

    import pytesseract as _pt
    if os.path.exists(tesseract_path):
        _pt.pytesseract.tesseract_cmd = tesseract_path

    from ocr_simulator import OCRSimulator
    sim = OCRSimulator(condition=condition, language=language_3,
                       font_size=font_size, dpi=dpi, save_images=False)
    try:
        result = sim.process_single_text(sentence)
        return result["ocr_text"] or sentence
    except Exception:
        return sentence


def ocr_simulate_documents_sentlevel(texts, condition, mldr_lang, desc="OCR simulating"):
    """
    Split each document into sentences, OCR all sentences in parallel,
    reassemble into documents. Returns list of noised document texts.
    """
    lang_3 = MLDR_TO_TESSERACT[mldr_lang]

    # Step 1: Split all documents into sentences, track boundaries
    print(f"    Splitting {len(texts)} docs into sentences...", end=" ", flush=True)
    doc_sentence_counts = []
    all_sentences = []

    for text in texts:
        sents = split_sentences(text)
        doc_sentence_counts.append(len(sents))
        all_sentences.extend(sents)

    total_sents = len(all_sentences)

    print(f"{total_sents:,} sentences")
    print(f"    Processable: {total_sents:,}")

    # Step 2: OCR all sentences in parallel
    args = [(s, condition, lang_3, OCR_FONT_SIZE, OCR_DPI,
             TESSERACT_PATH) for s in all_sentences]

    noised_sentences = [None] * total_sents
    errors = 0
    with ProcessPoolExecutor(max_workers=OCR_N_WORKERS) as executor:
        futures = {executor.submit(_ocr_one_sentence, a): i for i, a in enumerate(args)}
        for f in tqdm(as_completed(futures), total=total_sents,
                      desc=f"    {desc} ({mldr_lang})"):
            idx = futures[f]
            try:
                noised_sentences[idx] = f.result()
            except Exception:
                errors += 1
                noised_sentences[idx] = all_sentences[idx]

    if errors > 0:
        print(f"    ⚠ {errors} sentence errors (fell back to clean)")

    # Step 3: Reassemble into documents
    noised_docs = []
    offset = 0
    for count in doc_sentence_counts:
        doc_sents = noised_sentences[offset:offset + count]
        noised_docs.append(' '.join(doc_sents))
        offset += count

    assert len(noised_docs) == len(texts), \
        f"Reassembly error: {len(noised_docs)} vs {len(texts)}"

    return noised_docs


def _corpus_fingerprint(texts, n=5):
    """Quick fingerprint from first N text prefixes to detect ordering changes."""
    import hashlib
    h = hashlib.md5()
    for t in texts[:n]:
        h.update(t[:200].encode("utf-8", errors="replace"))
    return h.hexdigest()[:12]


def generate_or_load(texts, cache_file, condition, mldr_lang, desc):
    """Generate noised texts or load from cache (with fingerprint validation)."""
    fp_file = cache_file + ".fingerprint"
    current_fp = _corpus_fingerprint(texts)

    if os.path.exists(cache_file):
        print(f"    ✓ Cached: {os.path.basename(cache_file)}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        # Check length match
        if len(cached) != len(texts):
            print(f"    ⚠ Length mismatch ({len(cached)} vs {len(texts)}), regenerating...")
        else:
            # Check fingerprint match (detects re-ordering)
            if os.path.exists(fp_file):
                saved_fp = open(fp_file).read().strip()
                if saved_fp == current_fp:
                    return cached
                else:
                    print(f"    ⚠ Corpus ordering changed (fingerprint mismatch), regenerating...")
            else:
                # No fingerprint file — legacy cache, trust it but save fingerprint
                print(f"    ⚠ No fingerprint — assuming cache is valid (saving fingerprint)")
                with open(fp_file, "w") as f:
                    f.write(current_fp)
                return cached

    t0 = time.time()
    noised = ocr_simulate_documents_sentlevel(texts, condition, mldr_lang, desc=desc)
    elapsed = time.time() - t0
    print(f"    ✓ Done in {elapsed/60:.1f} min ({elapsed/len(texts):.2f}s/doc)")

    with open(cache_file, "wb") as f:
        pickle.dump(noised, f)
    with open(fp_file, "w") as f:
        f.write(current_fp)
    print(f"    Saved → {os.path.basename(cache_file)}")
    return noised


# ======================= CORPUS SAMPLING =======================
def sample_corpus_safely(corpus, qrels, sample_size, seed=42):
    if sample_size is None: return corpus
    relevant_ids = {item["corpus-id"] for item in qrels}
    relevant_docs = corpus.filter(lambda x: x["id"] in relevant_ids)
    n_rel = len(relevant_docs)
    if n_rel >= sample_size:
        print(f"    ⚠ {n_rel} relevant docs ≥ budget ({sample_size}), subsampling...")
        relevant_docs = relevant_docs.shuffle(seed=seed).select(range(sample_size))
        return relevant_docs
    other = corpus.filter(lambda x: x["id"] not in relevant_ids)
    other = other.shuffle(seed=seed).select(range(min(sample_size - n_rel, len(other))))
    sampled = concatenate_datasets([relevant_docs, other])
    print(f"    Corpus sampled: {n_rel} relevant + {len(other)} random = {len(sampled)}")
    return sampled


# ======================= MAIN =======================
if __name__ == "__main__":

  for _cfg_dpi, _cfg_font in RUN_CONFIGS:
    OCR_DPI = _cfg_dpi
    OCR_FONT_SIZE = _cfg_font

    # Create a dedicated subfolder for this configuration
    CFG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"dpi{OCR_DPI}_font{OCR_FONT_SIZE}")
    os.makedirs(CFG_OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"MIRACL — Generate OCR-noised texts (Sentence-Level)  [DPI={OCR_DPI}, Font={OCR_FONT_SIZE}]")
    print("=" * 60)
    print(f"Languages:       {EVAL_LANGUAGES}")
    print(f"Condition:       {OCR_CONDITION}")
    print(f"Font size:       {OCR_FONT_SIZE} pt")
    print(f"DPI:             {OCR_DPI}")

    print(f"Workers:         {OCR_N_WORKERS}")
    print(f"Sample size:     {CORPUS_SAMPLE_SIZE_DEFAULT}")
    print(f"Output:          {CFG_OUTPUT_DIR}")
    print("=" * 60)

    cer_results = []

    for lang in EVAL_LANGUAGES:
        print(f"\n{'─'*60}")
        print(f"  [{lang.upper()}] Loading MIRACL data... [DPI={OCR_DPI}, Font={OCR_FONT_SIZE}]")
        t0 = time.time()

        miracl_lang = MIRACL_LANG_MAP.get(lang)
        if not miracl_lang:
            print(f"  [{lang.upper()}] ✗ No MIRACL mapping found.")
            continue
        try:
            # ── 1. Download topics (queries) TSV directly — tiny file ──
            import csv, io, requests as _req
            base = f"https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{miracl_lang}"
            hdr = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"}

            topics_url = f"{base}/topics/topics.miracl-v1.0-{miracl_lang}-dev.tsv"
            qrels_url  = f"{base}/qrels/qrels.miracl-v1.0-{miracl_lang}-dev.tsv"

            r_topics = _req.get(topics_url, headers=hdr, timeout=30)
            r_topics.raise_for_status()
            r_qrels = _req.get(qrels_url, headers=hdr, timeout=30)
            r_qrels.raise_for_status()

            # Parse topics: query_id \t query
            query_map = {}  # qid -> query text
            for line in r_topics.text.strip().split("\n"):
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    query_map[parts[0]] = parts[1]

            # Parse qrels: query_id \t 0 \t doc_id \t relevance
            qrels = []
            needed_docids = set()
            for line in r_qrels.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 4:
                    qid, docid, score = parts[0], parts[2], int(parts[3])
                    qrels.append({"query-id": qid, "corpus-id": docid, "score": score})
                    needed_docids.add(docid)

            print(f"    Topics: {len(query_map)} queries, Qrels: {len(qrels)} judgments, Needed docs: {len(needed_docids)}")

            # ── 2. Stream corpus to collect only needed docs + random extras ──
            sample_size = CORPUS_SAMPLE_SIZE_PER_LANG.get(lang, CORPUS_SAMPLE_SIZE_DEFAULT)
            relevant_ids = {item["corpus-id"] for item in qrels}  # all judged docs (pos + neg)

            corpus_stream = load_dataset("miracl/miracl-corpus", miracl_lang,
                                         split="train", streaming=True,
                                         trust_remote_code=True)

            corpus_dict = {}
            random_extras = []
            import random as _rnd
            _rng = _rnd.Random(42)
            budget_random = max(0, (sample_size or 0) - len(relevant_ids)) if sample_size else 0

            print(f"    Streaming corpus (collecting {len(needed_docids)} needed + up to {budget_random} random)...")
            for row in corpus_stream:
                did = row["docid"]
                if did in needed_docids:
                    corpus_dict[did] = {"id": did, "title": row.get("title", ""), "text": row.get("text", "")}
                    if len(corpus_dict) == len(needed_docids) and budget_random == 0:
                        break
                elif budget_random > 0 and len(random_extras) < budget_random * 3:
                    # Reservoir-sample random docs
                    random_extras.append({"id": did, "title": row.get("title", ""), "text": row.get("text", "")})

            print(f"    Found {len(corpus_dict)}/{len(needed_docids)} needed docs, {len(random_extras)} random candidates")

            # Build sampled corpus (sorted by docid for deterministic ordering)
            relevant_docs = sorted(
                [corpus_dict[d] for d in relevant_ids if d in corpus_dict],
                key=lambda x: x["id"])
            other_docs = sorted(
                [corpus_dict[d] for d in corpus_dict if d not in relevant_ids],
                key=lambda x: x["id"])

            n_rel = len(relevant_docs)
            if sample_size and n_rel > sample_size:
                # Hard cap: subsample relevant docs to fit budget
                print(f"    ⚠ {n_rel} relevant docs > budget ({sample_size}), subsampling...")
                _rng.shuffle(relevant_docs)
                relevant_docs = sorted(relevant_docs[:sample_size], key=lambda x: x["id"])
                corpus_sampled = relevant_docs
                print(f"    Corpus sampled: {len(corpus_sampled)} (subsampled from {n_rel} relevant)")
            elif sample_size and n_rel < sample_size:
                _rng.shuffle(random_extras)
                to_add = (other_docs + random_extras)[:sample_size - n_rel]
                # Sort the extras deterministically too
                to_add = sorted(to_add, key=lambda x: x["id"])
                corpus_sampled = relevant_docs + to_add
                print(f"    Corpus sampled: {n_rel} relevant + {len(to_add)} others = {len(corpus_sampled)}")
            else:
                corpus_sampled = relevant_docs
                print(f"    Corpus sampled: {len(corpus_sampled)} relevant docs")

        except Exception as e:
            print(f"  [{lang.upper()}] ✗ Could not load dataset: {e}")
            import traceback; traceback.print_exc()
            continue

        # Prepare texts
        corpus_ids = [doc["id"] for doc in corpus_sampled]
        corpus_id_set = set(corpus_ids)
        corpus_texts = [f"{doc.get('title','')} {doc.get('text','')}".strip() for doc in corpus_sampled]

        query_ids = list(query_map.keys())
        query_texts = list(query_map.values())

        # Filter to active queries
        qrels_dict: Dict[str, Dict[str, int]] = {}
        for item in qrels:
            qid, did, score = item["query-id"], item["corpus-id"], item["score"]
            if did not in corpus_id_set: continue
            qrels_dict.setdefault(qid, {})[did] = score

        active_qids = set(qrels_dict.keys())
        active_mask = [i for i, qid in enumerate(query_ids) if qid in active_qids]
        query_ids = [query_ids[i] for i in active_mask]
        query_texts = [query_texts[i] for i in active_mask]

        print(f"  [{lang.upper()}] ✓ {len(corpus_texts)} docs, {len(query_texts)} queries [{time.time()-t0:.1f}s]")

        # File names
        sample_tag = f"sample{sample_size}" if sample_size else "full"

        # ── Save metadata pkl for test script (clean texts, IDs, qrels) ──
        metadata_file = os.path.join(CFG_OUTPUT_DIR, f"miracl_{lang}_{sample_tag}_metadata.pkl")
        metadata = {
            "corpus_ids": corpus_ids,
            "clean_corpus_texts": corpus_texts,
            "query_ids": query_ids,
            "clean_query_texts": query_texts,
            "qrels_dict": qrels_dict,
            "sample_size": sample_size,
        }
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)
        print(f"    Saved metadata → {os.path.basename(metadata_file)}")

        # ── Generate corpus pkl (sentence-level OCR) ──
        corpus_file = os.path.join(CFG_OUTPUT_DIR,
            f"miracl_{lang}_{sample_tag}_{OCR_CONDITION}_{OCR_DPI}_font{OCR_FONT_SIZE}_sentlevel_corpus.pkl")
        print(f"\n  [{lang.upper()}] Corpus OCR (sentence-level):")
        noised_corpus = generate_or_load(corpus_texts, corpus_file, OCR_CONDITION, lang,
                                         desc="Corpus OCR")

        # ── Generate queries pkl (queries are short, sentence-level still works) ──
        queries_file = os.path.join(CFG_OUTPUT_DIR,
            f"miracl_{lang}_{sample_tag}_{OCR_CONDITION}_{OCR_DPI}_font{OCR_FONT_SIZE}_sentlevel_queries.pkl")
        print(f"\n  [{lang.upper()}] Queries OCR (sentence-level):")
        noised_queries = generate_or_load(query_texts, queries_file, OCR_CONDITION, lang,
                                          desc="Query OCR")

        # ── Measure CER ──
        print(f"\n  [{lang.upper()}] Measuring CER...")

        # Corpus CER
        corpus_cers = [compute_cer(corpus_texts[i], noised_corpus[i])
                       for i in range(len(corpus_texts))
                       if corpus_texts[i].strip() and noised_corpus[i].strip()]
        corpus_cer_mean = np.mean(corpus_cers) if corpus_cers else 0.0
        corpus_zero = sum(1 for c in corpus_cers if c == 0.0)

        # Query CER
        query_cers = [compute_cer(query_texts[i], noised_queries[i])
                      for i in range(len(query_texts))
                      if query_texts[i].strip() and noised_queries[i].strip()]
        query_cer_mean = np.mean(query_cers) if query_cers else 0.0
        query_zero = sum(1 for c in query_cers if c == 0.0)

        print(f"    Corpus CER:  {corpus_cer_mean:.4f} ({corpus_cer_mean*100:.2f}%) "
              f"[{len(corpus_cers)} docs, {corpus_zero} zero-CER ({100*corpus_zero/max(len(corpus_cers),1):.0f}%)]")
        print(f"    Query CER:   {query_cer_mean:.4f} ({query_cer_mean*100:.2f}%) "
              f"[{len(query_cers)} queries, {query_zero} zero-CER ({100*query_zero/max(len(query_cers),1):.0f}%)]")

        cer_results.append({
            "language": lang,
            "condition": OCR_CONDITION,
            "corpus_cer": corpus_cer_mean,
            "corpus_n_measured": len(corpus_cers),
            "corpus_zero_cer": corpus_zero,
            "corpus_zero_pct": round(100 * corpus_zero / max(len(corpus_cers), 1), 1),
            "query_cer": query_cer_mean,
            "query_n_measured": len(query_cers),
            "query_zero_cer": query_zero,
            "query_zero_pct": round(100 * query_zero / max(len(query_cers), 1), 1),
        })

    # ── Summary ──
    print("\n" + "=" * 60)
    print("✓ DONE — Generated files:")
    print("=" * 60)
    for f in sorted(os.listdir(CFG_OUTPUT_DIR)):
        if f.endswith(".pkl") and "miracl" in f:
            size_mb = os.path.getsize(os.path.join(CFG_OUTPUT_DIR, f)) / 1024 / 1024
            print(f"  {f} ({size_mb:.1f} MB)")

    if cer_results:
        print("\n" + "=" * 60)
        print("CER Summary (Character Error Rate)")
        print("=" * 60)
        cer_df = pd.DataFrame(cer_results)
        print(cer_df.to_string(index=False))
        cer_df.to_csv(os.path.join(CFG_OUTPUT_DIR, f"miracl_cer_summary_{OCR_CONDITION}_{OCR_DPI}_font{OCR_FONT_SIZE}_sentlevel.csv"), index=False)
        print(f"\n  Saved → miracl_cer_summary_{OCR_CONDITION}_{OCR_DPI}_font{OCR_FONT_SIZE}_sentlevel.csv")

    print(f"\nDone with DPI={OCR_DPI}, Font={OCR_FONT_SIZE}! Output: {CFG_OUTPUT_DIR}")