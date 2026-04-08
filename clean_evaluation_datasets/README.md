# Clean Evaluation Datasets



## CLSD

The Cross-Lingual Semantic Discrimination (CLSD) benchmark introduced by Michail et al. (2025a).
Each row contains a source sentence, its correct translation, and four semantically similar
distractor sentences in the target language. The model must rank the correct translation
higher than all distractors. Performance is reported as Precision@1.

Built from the WMT19 and WMT21 DE–FR news test sets.

| File | Source dataset | Columns |
|------|---------------|---------|
| `CLSD_wmt2019_adversarial_dataset.csv` | WMT19 DE–FR | `German`, `French`, `de_adv1`–`de_adv4`, `fr_adv1`–`fr_adv4` |
| `CLSD_wmt2021_adversarial_dataset.csv` | WMT21 DE–FR | `German`, `French`, `de_adv1`–`de_adv4`, `fr_adv1`–`fr_adv4` |


---

## X-STS

Cross-lingual Semantic Textual Similarity files from the SemEval-2017 Task 1 benchmark
(Cer et al., 2017). Used as a control task to verify that OCR adaptation does not degrade
performance on clean, out-of-training-distribution language pairs. Performance is reported
as Spearman correlation (×100) between model cosine scores and human similarity scores.

| File | Language pair | Columns |
|------|--------------|---------|
| `sts17_ar-en.csv` | Arabic–English | `ar`, `en`, `similarity_score` |
| `sts17_en-es.csv` | English–Spanish | `en`, `es`, `similarity_score` |
| `sts17_es-en.csv` | Spanish–English | `es`, `en`, `similarity_score` |
| `sts17_tr-en.csv` | Turkish–English | `tr`, `en`, `similarity_score` |

