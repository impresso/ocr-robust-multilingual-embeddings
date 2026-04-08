# generate_random_character_noise

Script for adding synthetic OCR noise columns to a CSV file.
Output column names follow the `noisy_finetuning_data` naming convention:
`{column}_{cer_suffix}`, where the suffix is the CER percentage zero-padded to two digits.

---

## How it works

For each target column, the script adds a new noisy column by applying
`max(1, int(len(text) * cer))` random character-level perturbations drawn
from three operation types:

- **Substitution** — replace a character with a random one from the script charset
- **Insertion** — insert a random character at a random position
- **Deletion** — remove a character at a random position

Each operation is chosen with equal probability. The charset used for substitution
and insertion depends on the `--script` argument.

---

## Supported scripts

| `--script` | Charset covers |
|------------|---------------|
| `latin` | a–z, A–Z, weighted spaces, öüäéèàÜÄÖ |
| `cyrillic` | а–я, А–Я, weighted spaces, ёЁ |
| `greek` | α–ω, Α–Ω, weighted spaces, άέήίόύώ |
| `arabic` | Core Arabic letters, weighted spaces, ءآأؤإئ |
| `hebrew` | Core Hebrew letters, weighted spaces, final forms |
| `georgian` | Core Georgian letters, weighted spaces |

---


## Usage

```bash
python generate_random_character_noise.py input.csv \
    --columns deu fra \
    --script latin \
    --cer 0.04 \
    --seed 42 \
    -o output.csv
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_csv` | yes | — | Path to the input CSV file |
| `--columns` | yes | — | One or more column names to noise |
| `--script` | no | `latin` | Script charset to use for substitution/insertion |
| `--cer` | no | `0.04` | Target character error rate |
| `--seed` | no | `None` | Random seed for reproducibility |
| `-o / --output` | no | overwrites input | Output CSV path |

---

## Examples


```bash
# CER 4%
python generate_random_character_noise.py TED_data.csv \
    --columns deu fra --cer 0.04 --seed 42 -o TED_data_random_noise.csv

# CER 10%
python generate_random_character_noise.py TED_data_random_noise.csv \
    --columns deu fra --cer 0.10 --seed 42

# CER 15%
python generate_random_character_noise.py TED_data_random_noise.csv \
    --columns deu fra --cer 0.15 --seed 42
```

After running all three, the output CSV will have the columns:
`deu`, `fra`, `deu_04`, `fra_04`, `deu_10`, `fra_10`, `deu_15`, `fra_15`
