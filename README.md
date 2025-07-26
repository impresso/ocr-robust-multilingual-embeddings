# [Cheap Character Noise for OCR-Robust Multilingual Embeddings](https://aclanthology.org/2025.findings-acl.609/) - Datasets, Resources and Adapted Models <img height="24" alt="acl2025 vienna" src="https://github.com/user-attachments/assets/73357d43-7d70-4556-b448-f85da93c1e90" />
![License: AGPLV3+](https://img.shields.io/badge/License-AGPLV3+-brightgreen.svg)

---

## Overview

This repository accompanies our [ACL2025 Findings paper](https://aclanthology.org/2025.findings-acl.609/), providing models, noisy datasets, and tools for robust multilingual embeddings under OCR noise. You’ll find fine-tuned models, evaluation and training data, and utilities for simulating character-level OCR noise.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Datasets](#datasets)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Citation](#citation)
- [Further Support & Contributing](#support--contributing)
- [About Impresso](#about-impresso)
- [License](#license)

---


## Motivation

To be added!

## Repository Structure

The repository is organized as follows:

```
├── noisy_evaluation_datasets
│   └── The noised evaluation datasets (CLSD - WMT19/21) produced.
├── noisy_finetuning_data
│   └── The 10K (per language) noised training samples (TED - X-News) used for fine-tuning the models. Includes both random and realistic OCR Noise variants.
├── ocr_simulator
│   └── The ocr_simulator library used to induce realistic ocr noise to texts.
├── generate_random_character_noise_latin_alphabet
│   └── The script to generate stochastically the character level noise used to fine-tune our models.
```

---

## Models

A version of our **OCR Robust models** (fine-tuned on [TED-X with random noise](noisy_finetuning_data/TED_data_random_noise_10k_sampled.csv)) is available on Hugging Face:  
[impresso-project/OCR-robust-gte-multilingual-base](https://huggingface.co/impresso-project/OCR-robust-gte-multilingual-base)

---

## Datasets

### Evaluation Datasets

Noisy variants of the CLSD WMT datasets are available in [noisy_evaluation_datasets](./noisy_evaluation_datasets).

### Finetuning Datasets

Noisy versions (random and realistic) of TED and X-News parallel texts are available in [noisy_finetuning_data](./noisy_finetuning_data).

### Other Datasets

Additional datasets used for evaluation and finetuning are also provided ([link](https://drive.google.com/file/d/1gydv66U99Gi5x7Uj_fJFLjZYEVC9EHsR/view?usp=sharing)):

- **STS-X:** [paper](https://aclanthology.org/anthology-files/pdf/S/S17/S17-2001.pdf)
- **CLSD:** [paper](https://arxiv.org/pdf/2502.08638)
- **HistLUX:** [paper](https://aclanthology.org/2025.latechclfl-1.26.pdf)

---

## Reproducing the Experiments

*Instructions for reproducing the experiments will be available soon!*

---

## Citation

If you use these resources, please cite our paper:

```bibtex
@inproceedings{michail-etal-2025-cheap,
    title = "Cheap Character Noise for {OCR}-Robust Multilingual Embeddings",
    author = "Michail, Andrianos  and
      Opitz, Juri  and
      Wang, Yining  and
      Meister, Robin  and
      Sennrich, Rico  and
      Clematide, Simon",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.609/",
    pages = "11705--11716",
    ISBN = "979-8-89176-256-5",
    abstract = "The large amount of text collections digitized by imperfect OCR systems requires semantic search models that perform robustly on noisy input. Such collections are highly heterogeneous, with varying degrees of OCR quality, spelling conventions and other inconsistencies {---}all phenomena that are underrepresented in the training data of standard embedding models, with ramifications for their generalization. In our paper, we show that this problem can be alleviated with a simple and inexpensive method that does not require supervision or in-domain training. Specifically, we fine-tune existing multilingual models using noisy texts and a contrastive loss. We show that these models show considerable improvements across different noise conditions. Control experiments indicate minimal, and occasionally positive, impact on standard similarity tasks. These findings suggest that embedding models can be inexpensively adapted for cross-lingual semantic search in heterogeneous, digitized corpora. We publicly release our code, datasets, and models at https://github.com/impresso/ocr-robust-multilingual-embeddings."
}
```

## Further Support & Contributing
In the future, we will work towards creating multilingual embedding models that are diversely robust. If you are interested in contributing or need access to any (not yet) released material, please reach out to andrianos.michail@cl.uzh.ch.

## About Impresso

### Impresso project

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders. The first project (2017-2021) was funded by the Swiss National Science Foundation under grant No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027) by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the Luxembourg National Research Fund under grant No. 17498891.

### Copyright

Copyright (C) 2025 The Impresso team.

### License

This program is provided as open source under the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE) v3 or later.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>
