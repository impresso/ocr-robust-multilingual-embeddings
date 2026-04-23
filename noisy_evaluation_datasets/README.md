# Noisy Evaluation Datasets

## Noisy CLSD Variants

**Source:** [Examining Multilingual Embedding Models Cross-Lingually through LLM-Generated Adversarial Examples](https://arxiv.org/pdf/2502.08638)

OCR-degraded versions of the clean CLSD files from [`clean_evaluation_datasets/ACL/`](../clean_evaluation_datasets/ACL/).
Generated using the three-step pipeline from Michail et al. (2025b): (S1) print the text under
realistic conditions, (S2) apply visual degradations, (S3) re-digitize with Tesseract.

Each file shares the same column structure as the clean CLSD files
(excluding `deu_04` and `fra_04`, which only appear in the clean files):
`fra`, `deu` (gold translations), `de_adv1`–`de_adv4` (German distractors), `fr_adv1`–`fr_adv4` (French distractors).
Every column (gold + all distractors) has been OCR-degraded.

| File | Source dataset | Noise type | Avg. CER |
|------|---------------|------------|----------|
| `CLSD_WMT19_MN_noise.csv` | WMT19 DE–FR | Minimal Noise (MN) | DE: 0.4%, FR: 0.7% |
| `CLSD_WMT21_MN_noise.csv` | WMT21 DE–FR | Minimal Noise (MN) | DE: 0.4%, FR: 0.7% |
| `CLSD_WMT19_BLDS_noise.csv` | WMT19 DE–FR | Blackletter / Scanned Distorted (BL/SD) | DE: 2.8%, FR: 2.4% |
| `CLSD_WMT21_BLDS_noise.csv` | WMT21 DE–FR | Blackletter / Scanned Distorted (BL/SD) | DE: 2.6%, FR: 2.8% |
| `CLSD_WMT19_SNP_noise.csv` | WMT19 DE–FR | Salt-and-Pepper (SnP) | DE: 5.8%, FR: 5.3% |
| `CLSD_WMT21_SNP_noise.csv` | WMT21 DE–FR | Salt-and-Pepper (SnP) | DE: 5.6%, FR: 5.4% |

### Noise conditions

- **Minimal Noise (MN):** Text printed in Times New Roman at 10pt, saved at 300 PPI, re-OCRed with Tesseract.
- **Blackletter / Scanned Distorted (BL/SD):** German text rendered in Canterbury (blackletter font); French text rendered with horizontal offsets and random spacing distortions.
- **Salt-and-Pepper (SnP):** Background noise added by scattering black and white pixels at 0.45% density before OCR.

---

## HISTLUX Bitext Mining

**Source:** [Adapting Multilingual Embedding Models to Historical Luxembourgish](https://aclanthology.org/2025.latechclfl-1.26.pdf)

Historical Luxembourgish bitext mining dataset introduced by Michail et al. (2025c).
Contains 233 digitized historical Luxembourgish newspaper articles (1840–1950),
sentence-segmented and machine-translated into modern German, French, and English.
The task requires ranking the correct translation highest among candidate sentences.
Performance is reported as Precision@1, averaged over the three language pairs.

> **Note:** The six bitext mining JSONL files (350–425 MB each) are **not** included in
> this repository. Download the ready-to-use prepared bitext mining test set from
> [Google Drive](https://drive.google.com/file/d/1B_na_iXXa5nNcfh8L7sNIln9hNkji0ad/view).
>
> Place the downloaded JSONL files in `noisy_evaluation_datasets/ACL/`.

Six files covering both directions of each language pair:

| File | Task direction |
|------|---------------|
| `bitext_mining_task_lb_to_de.jsonl` | Luxembourgish → German |
| `bitext_mining_task_lb_to_en.jsonl` | Luxembourgish → English |
| `bitext_mining_task_lb_to_fr.jsonl` | Luxembourgish → French |
| `bitext_mining_task_de_to_lb.jsonl` | German → Luxembourgish |
| `bitext_mining_task_en_to_lb.jsonl` | English → Luxembourgish |
| `bitext_mining_task_fr_to_lb.jsonl` | French → Luxembourgish |

### JSONL schema

Each line contains one bitext mining instance. The first element of `candidates` is always
the correct translation; the remaining elements are distractor sentences.

```json
{
  "source_sentence": "Briefkasten.",
  "candidates": ["Briefkasten.", "distractor 1", "distractor 2", "..."]
}
```

