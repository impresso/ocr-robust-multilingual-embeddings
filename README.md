# Datasets, Resources and Models from [Cheap Character Noise for OCR-Robust Multilingual Embeddings](https://aclanthology.org/2025.coling-main.585/) - <img height="24" alt="acl2025 vienna" src="https://github.com/user-attachments/assets/73357d43-7d70-4556-b448-f85da93c1e90" />
![License: AGPLV3+](https://img.shields.io/badge/License-AGPLV3+-brightgreen.svg)

## Table of Contents

- [Motivation](#motivation)
- [Repository Organization](#repository-organization)
- [OCR Robust Models](#ocr-robust-models)
- [Released Datasets](#released-datasets)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [BibTeX Reference](#bibtex-reference)
- [Further Support](#further-support)
- [About Impresso](#about-impresso)

## Motivation

## Repository Organization:

The repository is organized as follows:

```
├── original_reproduction_code
│   └── The initial version of the repository.
├── datasets_no_results
│   └── Contains the datasets used in the experiments, in a JSON format. Copied for every new benchmark
├── models
│   └── Empty models file used in the experiments.
├── benchmarking.py
│   └── Main benchmarking code, for running predictions on the datasets using specified methods.
├── extract_results.py
│   └── Script for extracting result of a benchmark.
├── lm_studio_templates
|   └──templates.py
|      └──Sample functions for making prediction functions using LM Studio
|   └──paper_methods.py
|      └──methods used in the paper to run benchmarks using LM Studio
|   └──l70b_methods.py
|      └──methods to run the benchmark using Llama3.3 70b Q8 and 8b Q4
├── logger.py
│   └── Utility for managing logging: all events are logged both to stdout and to a local logs.log file.
├── paper_config.json
│   └── Benchmark configuration for the methods used in the paper.
├── llama3_3_70b_config.json
│   └── Benchmark configuration for running the benchmark using Llama3.3 70b Q8 and 8b Q4
```

## OCR Robust Models

## Released Datasets

### Evaluation Datasets

### Finetuning Datasets

### Other Evaluation Datasets

## Reproducing the Experiments
I will update this soon! Pretend it's here already!

## BibTeX Reference

If you would like to cite this project, or the associated paper, here's a bibtex:

```bibtex
UPDATE ONCE IT'S AVAILABLE
```

## Further Support
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
