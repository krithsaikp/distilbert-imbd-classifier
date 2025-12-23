# DistilBERT IMDb Sentiment Classifier

Fine-tuning of DistilBERT on the IMDb movie-review dataset for binary sentiment classification. This repository contains a simple training script for quick experimentation with Hugging Face Transformers.

**Quick Start**

- **Install dependencies:**

```bash
pip install -r requirements.txt
```

- **Run training:**

```bash
python train.py
```

**Training**

- **Model:** `distilbert-base-uncased` (Hugging Face Transformers)
- **Script:** [train.py](train.py)
- **Config (defaults in `train.py`):** output dir `./bert-finetuned-imdb`, learning rate `2e-5`, `per_device_train_batch_size=2`, `per_device_eval_batch_size=8`, `num_train_epochs=2`, `max_length=128`
- **Dataset subset:** training subset = 5,000 samples, test subset = 2,000 samples (selected for faster experiments)

**Results**

- **Accuracy:** ~0.85-0.90
- **Loss:** ~0.30-0.40
- **Evaluation:** `trainer.predict()` is run at the end of `train.py` and prints evaluation metrics to stdout. 

**Files**

- **`train.py`** — training script (tokenization, model loading, Trainer usage) — see [train.py](train.py)
- **`requirements.txt`** — Python dependencies — see [requirements.txt](requirements.txt)

**Usage tips**

- To train on the full dataset, remove the `.select(range(...))` calls in `train.py`.
- Increase `per_device_train_batch_size` and `num_train_epochs` when using a GPU for better performance.
- Save model and tokenizer for later use with `model.save_pretrained()` and `tokenizer.save_pretrained()`.

**Next steps**

- Add explicit evaluation metrics (accuracy, precision/recall) and save them to a JSON file.
- Add a small inference script demonstrating how to load the saved model and run predictions on new text.
- Train on full IMDb dataset using GPU

---
