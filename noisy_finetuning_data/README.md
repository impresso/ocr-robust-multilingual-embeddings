# Noisy Finetuning Data


## ACL/

Parallel texts from [TED2020](https://opus.nlpl.eu/) and
[X-News](https://aclanthology.org/2025.findings-acl.609/) with synthetically
injected **random character noise** at varying Character Error Rates (CER).
Used in **Stage B** of the training pipeline.

### Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `TED_data_random_noise_10k_sampled.csv` | 10 000 | `german`, `french`, `german_noise_random05`, `german_noise_random10`, `german_noise_random15`, `french_noise_random05`, `french_noise_random10`, `french_noise_random15` | 10K stratified sample of TED with both languages in separate columns and noise at CER ≈ 5 %, 10 %, 15 % |
| `TED_data_random_noise_concat.csv` | 348 658 | `deu`, `deu_04` | Full TED corpus with German and French rows stacked vertically into a single column pair at CER ≈ 4 % (~2× the original 174K rows) |
| `X-News_data_random_noise_10k_sampled.csv` | 10 000 | `german`, `french`, `german_noise_random05`, …, `french_noise_random15` | 10K stratified sample of X-News with both languages in separate columns and noise at CER ≈ 5 %, 10 %, 15 % |


---

## LREC/

Data supporting both stages of the LREC 2026 recipe. **Stage A** uses cross-lingual
parallel Luxembourgish pairs (`lb_de_training_set.jsonl`, `lb_en_training_set.jsonl`,
`lb_fr_training_set.jsonl`). **Stage B** uses OCR-noised historical newspaper articles
(`de_docs_random_noise.csv`, `fr_docs_random_noise.csv`) and a noised sample from [MLSum](https://huggingface.co/datasets/reciTAL/mlsum) (`query_doc_dataset_random_noise.csv`).

The Luxembourgish parallel training data originates from the
[impresso/histlux_emb](https://github.com/impresso/histlux_emb) repository.
You can download it from the `prepared_training_sentences/` folder in that repository
or from the [HistLuxAlign](https://huggingface.co/datasets/impresso-project/HistLuxAlign) dataset on Hugging Face.

### Files

| File | Rows | Format | Description |
|------|------|--------|-------------|
| `lb_de_training_set.jsonl` | 2 105 | JSONL | Luxembourgish ↔ German parallel sentence pairs from historical newspapers (GPT-4o segmented & translated) |
| `lb_en_training_set.jsonl` | 2 105 | JSONL | Luxembourgish ↔ English parallel sentence pairs |
| `lb_fr_training_set.jsonl` | 2 105 | JSONL | Luxembourgish ↔ French parallel sentence pairs |
| `de_docs_random_noise.csv` | 10 000 | CSV | German historical newspaper articles with random noise (`deu` → `deu_04`) |
| `fr_docs_random_noise.csv` | 10 000 | CSV | French historical newspaper articles with random noise (`fra` → `fra_04`) |
| `query_doc_dataset_random_noise.csv` | 10 000 | CSV | 10K sample from [MLSum](https://huggingface.co/datasets/reciTAL/mlsum) with random noise (`text`, `summary`, `query` → `text_04`, `summary_04`, `query_04`) |

### JSONL schema (parallel pairs)

Each line contains a `custom_id` and a `translation` array of sentence-aligned pairs:

```json
{
  "custom_id": "task-0_deletz1893-1893-01-01-a-i0003",
  "translation": [
    {"lb": "Briefkasten.", "de": "Briefkasten."},
    {"lb": "bie crftc Mummet ...", "de": "die erste Nummer ..."}
  ]
}
```

---

## Noise Generation

- **Random character noise** was generated with
  [`generate_random_character_noise/`](../generate_random_character_noise/)
  using script-specific confusable character tables (substitution, insertion,
  deletion, swap) at configurable CER levels.
