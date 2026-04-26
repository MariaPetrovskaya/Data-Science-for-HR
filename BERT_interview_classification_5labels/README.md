# Interview Outcome Classification with BERT

Fine-tuning **ruBERT-base** to automatically classify the outcome of a recruitment phone call into one of five categories. This was an internal R&D project; it is now open and free to use.

---

## Problem Statement

Recruitment teams conduct hundreds of phone interviews. Manually labelling each call outcome is slow and inconsistent. This project trains a Russian-language BERT model to classify call transcripts into structured outcome categories — enabling downstream analytics, reporting, and recruiter performance tracking.

---

## Label Schema

| Model label | Meaning |
|:-----------:|---------|
| 0 | Candidate declined |
| 1 | Recruiter declined the candidate |
| 2 | Candidate agreed |
| 3 | Next steps were discussed |
| 4 | Candidate actually started the job |

> Calls that failed for technical reasons are excluded from training (they carry no signal about interview outcome).

---

## Approach

- **Base model:** [`ai-forever/ruBert-base`](https://huggingface.co/ai-forever/ruBert-base) — a BERT-base model pre-trained on a large Russian-language corpus
- **Fine-tuning head:** `BertForSequenceClassification` (dropout + linear layer on the `[CLS]` token)
- **Tokenisation:** WordPiece, max sequence length 512 tokens, dynamic padding via `DataCollatorWithPadding`
- **Optimiser:** AdamW, lr = 1e-6 (small learning rate to match small batch size)
- **Batch size:** 4
- **Epochs:** 3
- **Train/val split:** 80 / 20, stratified by label

The training data consists of labelled call transcripts in Russian. Labels 1–5 from the original dataset are mapped to 0–4 for the model.

---

## Results

Evaluated on a held-out validation set of **2,129 samples**.

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|--------:|
| Candidate declined | 0.86 | 0.90 | 0.88 | 467 |
| Recruiter declined candidate | 0.86 | 0.49 | 0.63 | 77 |
| Candidate agreed | 0.96 | 0.98 | **0.97** | 1 313 |
| Next steps discussed | 0.35 | 0.23 | 0.28 | 124 |
| Candidate started job | 0.47 | 0.54 | 0.50 | 148 |
| **Accuracy** | | | **0.87** | 2 129 |
| Macro avg | 0.70 | 0.63 | 0.65 | 2 129 |
| Weighted avg | 0.86 | 0.87 | 0.86 | 2 129 |

### Confusion Matrix

```
                          Predicted →
Actual ↓             0     1     2     3     4
Candidate declined [ 420    6    41    0    0 ]
Recruiter declined [  29   38    10    0    0 ]
Candidate agreed   [  24    0  1288    1    0 ]
Next steps         [   5    0     2   28   89 ]
Started job        [  12    0     6   50   80 ]
```

### Key observations

- **Candidate agreed (label 2)** is classified very reliably (F1 0.97), driven by its large support and clear linguistic signals.
- **Candidate declined (label 0)** also performs well (F1 0.88).
- **Next steps discussed (label 3)** is the hardest class (F1 0.28) — it is frequently confused with *Candidate started job*, likely because both involve future-oriented language and the boundary is semantically thin.
- **Recruiter declined (label 1)** suffers from low recall (0.49) due to class imbalance (only 77 samples vs 1 313 for the dominant class).
- Improving labels 1 and 3 would benefit most from additional labelled data and/or class-weighted loss.

---

## Repository Structure

```
.
├── BERT_interview_classification_5labels_EN.ipynb   # Main training notebook
├── README.md
└── dataset 5_cat/
    ├── *.csv                                        # Labelled call transcripts (not included)
    └── interview_classifier/                        # Saved model & tokenizer (after training)
        ├── config.json
        ├── model.safetensors
        ├── tokenizer_config.json
        └── vocab.txt
```

---

## Requirements

```
transformers>=4.30
datasets>=2.0
torch>=2.0
scikit-learn
pandas
numpy
tqdm
matplotlib
```

Install with:

```bash
pip install transformers datasets torch scikit-learn pandas numpy tqdm matplotlib
```

---

## Quick Start

The notebook is designed for **Google Colab** with a GPU runtime and a mounted Google Drive.

1. Open `BERT_interview_classification_5labels_EN.ipynb` in Colab.
2. Update the paths in the **Configuration** cell (`DATA_DIR`, `MODEL_DIR`).
3. Place your labelled CSV files in `DATA_DIR`. Each file must have at minimum:
   - `text` — the call transcript (Russian)
   - `result_column` — integer label (1–5; 0 = technical failure, excluded automatically)
4. Run all cells top-to-bottom.

### Inference on a single text

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "path/to/interview_classifier"
LABEL_NAMES = [
    "Candidate declined",
    "Recruiter declined candidate",
    "Candidate agreed",
    "Next steps discussed",
    "Candidate started job",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def classify(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return {name: round(float(p), 4) for name, p in zip(LABEL_NAMES, probs)}

print(classify("Yes, I am ready to start on Monday"))
```

---

## Limitations & Future Work

- **Class imbalance** — labels 1 (*Recruiter declined*) and 3 (*Next steps discussed*) are underrepresented. Class-weighted loss or oversampling would likely improve their F1.
- **Label ambiguity** — the boundary between *Next steps discussed* and *Candidate started job* is semantically close. Clearer annotation guidelines or merging these classes may help.
- **Language** — the model is trained on Russian transcripts. For other languages, start from a different multilingual or language-specific base model.
- **Data privacy** — call transcripts may contain personal data. Ensure compliance with local regulations before storing or sharing data.

---

## License

This project is free to use. No license file is currently included — if you plan to build on this work, MIT or Apache 2.0 are recommended starting points.

---

## Acknowledgements

- Base model: [ai-forever/ruBert-base](https://huggingface.co/ai-forever/ruBert-base) by Sber AI
- Fine-tuning framework: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Originally developed as an internal R&D project; now open to the community.
