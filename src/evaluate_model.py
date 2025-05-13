import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    # Load the test dataset stored earlier in prepare_data.py
    test_dataset = load_from_disk("data/test")
    
    # Load the fine-tuned model and tokenizer we got from the train_model.py
    model_path = "models/sentiment-classifier/final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Automatically use a GPU if available, otherwise uses CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize dataset: 
    # Convert raw text into numerical format the model understands
    # Pads and truncates text to a consistent length & return PyTorch tensors
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
    
    all_predictions = []
    all_labels = []
    
    # Run Inference (prediction loop)
    # Set the model to Evaluation mode & disable gradient computation to save memory
    # Process the test dataset in batches of 32: 
            # Tokenize each batch 
            # Move tensor data to GPU/CPU
            # Get model output
            # Use argmax to convert model output logits into prediction classes 0 or 1
            # Finally, store predictions and true labels
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), 32):
            batch = test_dataset[i:i+32]
            inputs = tokenize_function(batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(batch["label"])
    
    # Generate the Confusion Matrix and Classification Report
    labels = [0, 1]
    label_names = ["Negative", "Positive"]
    cm = confusion_matrix(all_labels, all_predictions, labels=labels)
    report = classification_report(all_labels, all_predictions, target_names=label_names, labels=labels, zero_division=0)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Plot Confusion Matrix as blue heatmap & save the file as an image "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    # plt.show()
    
    # Show sample predictions
    # Take at least first 10 test datasets, Truncate texts, show true labels (+ve/-ve) & predicted label
    examples_with_predictions = []
    for i in range(min(10, len(test_dataset))):
        example = test_dataset[i]
        prediction = all_predictions[i]
        examples_with_predictions.append({
            "text": example["text"][:100] + "...",  # Truncate texts for readability
            "true_label": "Positive" if example["label"] == 1 else "Negative",
            "predicted_label": "Positive" if prediction == 1 else "Negative"
        })
    
    # Print resulting info from above code nicely
    print("\nExample Predictions:")
    for ex in examples_with_predictions:
        print(f"Text: {ex['text']}")
        print(f"True: {ex['true_label']}, Predicted: {ex['predicted_label']}")
        print("-" * 50)
    
    # Returns the full list of predictions and true labels in case you want to do more with them later
    return all_predictions, all_labels

if __name__ == "__main__":
    evaluate_model()
