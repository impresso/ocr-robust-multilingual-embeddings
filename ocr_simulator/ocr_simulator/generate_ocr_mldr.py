# ============================================================================
# MLDR — Generate OCR-noised texts — Sentence-Level OCR
# ============================================================================
#
# Strategy: Split each document into sentences, OCR each sentence individually
# (short sentences always fit Tesseract), then rejoin into full documents.
#
# This ensures ~99% of text gets properly noised, unlike document-level OCR
# where long docs (>20K chars) crash Tesseract and fall back to clean text.
#
# Output: same pkl format as before (one noised text per document).
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
#     2. pip install pytesseract Pillow datasets tqdm jiwer pandas
#     3. pip install -e path/to/ocr-robust-multilingual-embeddings/ocr_simulator
#
# Usage:
#   python mldr_ocr_generation.py                     # default languages, 4 workers
#   python mldr_ocr_generation.py de,fr 6             # specific langs + workers
# ============================================================================

import os
import sys
import platform
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re, gc, time, random, pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from jiwer import cer as compute_cer
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

MLDR_TO_TESSERACT = {"de": "deu", "fr": "fra", "es": "spa", "ru": "rus", "en": "eng"}

# ── Sanity check ──
print("\n--- Sanity checks ---")
for lang3, sent in {
    "deu": "Dies ist ein Beispieltext.", "fra": "Ceci est un exemple.",
    "spa": "Esta es una oración.", "rus": "Это пример текста.",
    "eng": "This is a sample sentence.",
}.items():
    try:
        sim = OCRSimulator(condition="distorted", language=lang3, font_size=12, dpi=300)
        res = sim.process_single_text(sent)
        print(f"  [{lang3}] ✓  → {(res['ocr_text'] or '<EMPTY>')[:50]}")
    except Exception as e:
        print(f"  [{lang3}] ✗  {str(e)[:60]}")


# ======================= CONFIGURATION =======================
OUTPUT_DIR = "./ocrsim_noised_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Command line arguments: python script.py de,fr 6
if len(sys.argv) > 1:
    EVAL_LANGUAGES = sys.argv[1].split(",")
    OCR_N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
else:
    EVAL_LANGUAGES = ["de", "fr", "es", "ru", "en"]
    OCR_N_WORKERS = 4

EVAL_SPLIT = "test"
OCR_CONDITION = "distorted"

CORPUS_SAMPLE_SIZE_PER_LANG = {}
CORPUS_SAMPLE_SIZE_DEFAULT = 2000

OCR_FONT_SIZE = 12
OCR_DPI = 300


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


def generate_or_load(texts, cache_file, condition, mldr_lang, desc):
    """Generate noised texts or load from cache."""
    if os.path.exists(cache_file):
        print(f"    ✓ Cached: {os.path.basename(cache_file)}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if len(cached) == len(texts):
            return cached
        print(f"    ⚠ Length mismatch ({len(cached)} vs {len(texts)}), regenerating...")

    t0 = time.time()
    noised = ocr_simulate_documents_sentlevel(texts, condition, mldr_lang, desc=desc)
    elapsed = time.time() - t0
    print(f"    ✓ Done in {elapsed/60:.1f} min ({elapsed/len(texts):.2f}s/doc)")

    with open(cache_file, "wb") as f:
        pickle.dump(noised, f)
    print(f"    Saved → {os.path.basename(cache_file)}")
    return noised


# ======================= CORPUS SAMPLING =======================
def sample_corpus_safely(corpus, qrels, sample_size, seed=42):
    if sample_size is None: return corpus
    relevant_ids = {item["corpus-id"] for item in qrels}
    relevant_docs = corpus.filter(lambda x: x["id"] in relevant_ids)
    n_rel = len(relevant_docs)
    if n_rel >= sample_size:
        print(f"    ⚠ {n_rel} relevant docs ≥ budget ({sample_size})")
        return relevant_docs
    other = corpus.filter(lambda x: x["id"] not in relevant_ids)
    other = other.shuffle(seed=seed).select(range(min(sample_size - n_rel, len(other))))
    sampled = concatenate_datasets([relevant_docs, other])
    print(f"    Corpus sampled: {n_rel} relevant + {len(other)} random = {len(sampled)}")
    return sampled


# ======================= MAIN =======================
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("MLDR — Generate OCR-noised texts (Sentence-Level)")
    print("=" * 60)
    print(f"Languages:       {EVAL_LANGUAGES}")
    print(f"Condition:       {OCR_CONDITION}")
    print(f"Font size:       {OCR_FONT_SIZE} pt")
    print(f"DPI:             {OCR_DPI}")

    print(f"Workers:         {OCR_N_WORKERS}")
    print(f"Sample size:     {CORPUS_SAMPLE_SIZE_DEFAULT}")
    print(f"Output:          {OUTPUT_DIR}")
    print("=" * 60)

    cer_results = []

    for lang in EVAL_LANGUAGES:
        print(f"\n{'─'*60}")
        print(f"  [{lang.upper()}] Loading data...")
        t0 = time.time()

        try:
            corpus = load_dataset("mteb/MultiLongDocRetrieval", f"{lang}-corpus", split=EVAL_SPLIT)
            queries = load_dataset("mteb/MultiLongDocRetrieval", f"{lang}-queries", split=EVAL_SPLIT)
            qrels = load_dataset("mteb/MultiLongDocRetrieval", f"{lang}-qrels", split=EVAL_SPLIT)
        except Exception as e:
            print(f"  [{lang.upper()}] ✗ {e}")
            continue

        sample_size = CORPUS_SAMPLE_SIZE_PER_LANG.get(lang, CORPUS_SAMPLE_SIZE_DEFAULT)
        corpus = sample_corpus_safely(corpus, qrels, sample_size)

        # Prepare texts
        corpus_ids = [doc["id"] for doc in corpus]
        corpus_id_set = set(corpus_ids)
        corpus_texts = [f"{doc.get('title','')} {doc.get('text','')}".strip() for doc in corpus]

        query_ids = [q["id"] for q in queries]
        query_texts = [q["text"] for q in queries]

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

        # ── Generate corpus pkl (sentence-level OCR) ──
        corpus_file = os.path.join(OUTPUT_DIR,
            f"{lang}_{sample_tag}_{OCR_CONDITION}_sentlevel_corpus.pkl")
        print(f"\n  [{lang.upper()}] Corpus OCR (sentence-level):")
        generate_or_load(corpus_texts, corpus_file, OCR_CONDITION, lang,
                         desc="Corpus OCR")

        # ── Generate queries pkl (queries are short, sentence-level still works) ──
        queries_file = os.path.join(OUTPUT_DIR,
            f"{lang}_{sample_tag}_{OCR_CONDITION}_sentlevel_queries.pkl")
        print(f"\n  [{lang.upper()}] Queries OCR (sentence-level):")
        generate_or_load(query_texts, queries_file, OCR_CONDITION, lang,
                         desc="Query OCR")

        # ── Measure CER ──
        print(f"\n  [{lang.upper()}] Measuring CER...")
        with open(corpus_file, "rb") as f:
            noised_corpus = pickle.load(f)
        with open(queries_file, "rb") as f:
            noised_queries = pickle.load(f)

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
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".pkl"):
            size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
            print(f"  {f} ({size_mb:.1f} MB)")

    if cer_results:
        print("\n" + "=" * 60)
        print("CER Summary (Character Error Rate)")
        print("=" * 60)
        cer_df = pd.DataFrame(cer_results)
        print(cer_df.to_string(index=False))
        cer_df.to_csv(os.path.join(OUTPUT_DIR, f"cer_summary_{OCR_CONDITION}_sentlevel.csv"), index=False)
        print(f"\n  Saved → cer_summary_{OCR_CONDITION}_sentlevel.csv")

    print(f"\nDone! Upload {OUTPUT_DIR}/ to Google Drive or push to HuggingFace")