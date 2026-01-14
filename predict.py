import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    label = torch.argmax(logits, dim=1).item()
    return "Positive review" if label == 1 else "Negative review"

def main():
    parser = argparse.ArgumentParser(description="Sentiment prediction CLI")
    parser.add_argument("text", type=str, help="Text to classify")
    parser.add_argument("--model", type=str, default="./bert-finetuned-imdb/checkpoint-5000",
                        help="Path to fine-tuned model")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    result = predict(args.text, tokenizer, model)
    print(result)

if __name__ == "__main__":
    main()
