import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
import evaluate

# Used by Model Trainer during evaluation to get Accuracy and F1 scores
def compute_metrics(pred: EvalPrediction):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    
    logits, labels = pred.predictions, pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {"accuracy": accuracy_score, "f1": f1_score}


def train_model():
    # Load datasets objects saved earlier by prepare_data.py
    train_dataset = load_from_disk("data/train")
    test_dataset = load_from_disk("data/test")

    # Load tokenizer and model for Binary classification (positive vs. negative sentiment)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Apply the tokenizer function to each item in the dataset.
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Training arguments (reduced to avoid incompatible params)
    training_args = TrainingArguments(
        output_dir="models/sentiment-classifier",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer = Tie everything together: model, data, config, and evaluation.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # Train model = Train the model with all the settings defined earlier
    trainer.train()

    # Save model and tokenizer so they can be reused without retraining
    model_path = "models/sentiment-classifier/final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    # Evaluate the model on the test set and print Accuracy and F1 scores
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    return model_path


if __name__ == "__main__":
    train_model()
