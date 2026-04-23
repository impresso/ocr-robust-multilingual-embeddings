# Clean Evaluation Datasets



## CLSD

The Cross-Lingual Semantic Discrimination (CLSD) benchmark introduced by Michail et al. (2025a).
Each row contains a source sentence, its correct translation, and four semantically similar
distractor sentences in the target language. The model must rank the correct translation
higher than all distractors. Performance is reported as accuracy (%).

Built from the WMT19 and WMT21 DE–FR news test sets.

| File | Source dataset | Columns |
|------|---------------|---------|
| `CLSD_wmt2019_adversarial_dataset.csv` | WMT19 DE–FR | `fra`, `deu` (gold), `deu_04` (DE noised ~4% CER), `de_adv1`–`de_adv4` (DE distractors), `fra_04` (FR noised ~4% CER), `fr_adv1`–`fr_adv4` (FR distractors) |
| `CLSD_wmt2021_adversarial_dataset.csv` | WMT21 DE–FR | same columns |


---

## X-STS

Cross-lingual Semantic Textual Similarity files from the SemEval-2017 Task 1 benchmark
(Cer et al., 2017). Used as a control task to verify that OCR adaptation does not degrade
performance on clean, out-of-training-distribution language pairs. Performance is reported
as Spearman correlation (×100) between model cosine scores and human similarity scores.

| File | Language pair | Columns |
|------|--------------|---------|
| `sts17_ar-en.csv` | Arabic–English | `eng`, `ara`, `similarity_score` |
| `sts17_en-es.csv` | English–Spanish | `eng`, `spa`, `similarity_score` |
| `sts17_es-en.csv` | Spanish–English | `spa`, `eng`, `similarity_score` |
| `sts17_tr-en.csv` | Turkish–English | `eng`, `tur`, `similarity_score` |

