from datasets import load_dataset
dataset = load_dataset("imdb")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length = 128)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.remove_columns(["text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./bert-finetuned-imdb",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01
)
from transformers import Trainer
small_train = tokenized["train"].shuffle(seed=42).select(range(5000)) 
small_test = tokenized["test"].shuffle(seed=42).select(range(2000))
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_test
)

trainer.train()
predictions = trainer.predict(small_test)
print(predictions.metrics)