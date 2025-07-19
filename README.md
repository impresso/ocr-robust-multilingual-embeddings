# Datasets, Resources and Models created as part of ACL2025 Findings paper [Cheap Character Noise for OCR-Robust Multilingual Embeddings](https://aclanthology.org/2025.coling-main.585/)
![License: AGPLV3+](https://img.shields.io/badge/License-AGPLV3+-brightgreen.svg)

## Table of Contents

- [Motivation](#overview)
- [Repository Organization](#repository-organization)
- [OCR-Robust Models - Huggingface](#models-huggingface)
- [Released Datasets](#datasets)
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

## Reproducing the Experiments
I will update this soon! Pretend it's here already!

## BibTeX Reference

If you would like to cite this project, or the associated paper, here's a bibtex:

```bibtex
UPDATE ONCE IT'S AVAILABLE
```
## About Impresso

### Impresso project

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders. The first project (2017-2021) was funded by the Swiss National Science Foundation under grant No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027) by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the Luxembourg National Research Fund under grant No. 17498891.

### Copyright

Copyright (C) 2025 The Impresso team.

### License

This program is provided as open source under the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE) v3 or later.
