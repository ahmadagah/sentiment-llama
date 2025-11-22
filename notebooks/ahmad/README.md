# Comparative Evaluation of LoRA, QLoRA, and IA-3 on Llama-3.1-1B for Sentiment Analysis

---

# Abstract

- LLMs too large so we need efficient fine-tuning
- focus: LoRA, QLoRA, IA-3  
- Task: Sentiment analysis  
- Metrics: accuracy, efficiency, model size  
- Summary of findings

---

# 1. Introduction

## 1.1 Background

- Transformer models (self-attention)
- Base models
- In-context learning limitations  
- Why fine-tuning is needed
  
---

## 1.3 Background (Low-Rank and Quantization Concepts)

### LoRA

### QLoRA

### IA-3
---


# 2. Methodology

## 2.1 Model
- Llama-3.1-1B-Instruct
- Model loading  
- Tokenizer details

---

## 2.2 Dataset

---


## 2.4 Hardware Setup
- Google Colab T4/L4....
- Mac MPS (Apple Silicon)  
- Windows/Linux (CPU or NVIDIA GPU via CUDA)

analysis of runtime, VRAM usage, and hardware-specific performance

---

## 2.5 Training Setup
- Learning rate  
- Batch size  
- Epoch count  

---

# 3. Experiments

## 3.1 Zero-Shot
- Baseline accuracy  
## 3.2 Few-Shot (1–3 shot)
## 3.3 LoRA Fine-Tuning
## 3.4 QLoRA Fine-Tuning
## 3.5 IA-3 Fine-Tuning

For each:
- Accuracy  
- Confusion matrix  
- Classification report  
- Loss curves  
- Runtime  
- VRAM usage  

---

# 4. Results

(accuracy comparison graph)
(VRAM comparison graph)
(model size table)
(confusion matrices comparison)

---

# 5. Discussion
- Accuracy changes  
- Efficiency trade-offs  
- QLoRA compression  
- IA-3 behavior  
- Error analysis  

---

# 6. Conclusion
- Main findings  
- Best method for 1B model  
- Efficiency summary  

---

# 7. Limitations
- Dataset constraints  
- Model size  
- Hardware limits  
- Quantization limitations)

---

# 8. Future Work
- Larger models (3B, 8B)  
- More PEFT methods  
- Domain-specific datasets  
- Hybrid approaches  

---

# References
(LoRA, QLoRA, IA-3, Llama-3, and dataset citations)
