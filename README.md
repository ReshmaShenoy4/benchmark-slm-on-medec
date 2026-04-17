# Benchmarking Small Language Models on Medical Error Detection and Correction in Clinical Notes

## Overview

This project benchmarks **small language models (SLMs)** on two clinically important tasks using the **MEDEC dataset**:

1. **Medical error detection** in clinical notes
2. **Medical error correction** in clinical notes

The goal is to evaluate how well general purpose & medical small language models can identify erroneous medical content and generate corrected versions, with a focus on clinical reliability and model comparison.

---

## Motivation

While large language models have shown strong general abilities, their use in clinical settings may be limited by cost, latency, and deployment constraints. This project studies whether **SLMs** can serve as effective alternatives for detecting and correcting medical errors in clinical text.

---

## Objectives

* Benchmark multiple SLMs on the **MEDEC** dataset
* Evaluate performance on both:

  * **Error detection**
  * **Error correction**
* Compare models using task-specific and text-generation metrics
* Analyze strengths and weaknesses of SLMs in clinical note understanding

---

## Dataset

We use the **MEDEC dataset**, which is designed for evaluating medical error detection and correction in clinical notes.

### Task Setup

The benchmark is divided into two main tasks:

#### 1. Error Detection

Models are evaluated on whether they can:

* **Flag if an error is present**
* **Identify the sentence containing the error**

#### 2. Error Correction

Models are evaluated on their ability to generate a corrected version of the erroneous clinical content.

---

## Evaluation Metrics

### Error Detection

* **Error Flag Accuracy**
* **Error Sentence Identification Accuracy**

### Error Correction

* **ROUGE-1**
* **BERTScore**
* **BLEURT**
* **Aggregate Score**

These metrics capture both lexical overlap and semantic quality of the corrected output.


## Repository Structure

```bash
.
├── setting 1 exp/         # Official Evaluation Setting
├── setting 2 exp/         # Relaxed Zero-Shot Evaluation Setting
├── setting 3 exp/         # Causally Linked Scoring One-Shot Evaluation Setting
├── prompt_sensitivity     # Phi3 prompt ordering evaluation
├── README.md
└── requirements.txt
```


## Key Questions

This project is centered around the following questions:

* How well do SLMs detect medical errors in clinical notes?
* Can SLMs reliably correct clinically incorrect content?
* Which models perform best across detection and correction tasks?
* Where do compact models fail in clinical note reasoning?

---

## Limitations

* Benchmark performance may depend heavily on prompting strategy
* Automatic correction metrics may not fully capture clinical correctness
* Good benchmark scores do not guarantee safety in real-world deployment

---

## Future Work

* Evaluate more SLMs and domain-specialized models
* Add few-shot and chain-of-thought prompting comparisons
* Evaluate clinical correctness in the generated corrections of the model 

## Acknowledgements
* MEDEC dataset authors
* Open-source model and evaluation libraries
* Prior work in clinical NLP and medical error analysis
