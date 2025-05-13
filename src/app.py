import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model_path = "models/sentiment-classifier/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    # Tokenize input & Move inputs to the same device as the model
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction using the inputs
    # Get logits (raw scores) & converts them to Probabilities using softmax
    # Get the index with highest probability (0 = -ve, 1 = +ve)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    
    # Get confidence scores in the form of list of type float
    confidence_scores = probabilities[0].tolist()
    
    # Return result as a dictionary showing confidence scores for both classes (0s and 1s)
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {
        "Negative": float(confidence_scores[0]),
        "Positive": float(confidence_scores[1])
    }

# Create simple Gradio interface with: 
        # A textbox for input
        # A label output showing both class probabilities
        # A title, description, and preloaded examples
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter a movie review..."),
    outputs=gr.Label(num_top_classes=2),
    title="Movie Review Sentiment Analysis",
    description="This app analyzes the sentiment of movie reviews as positive or negative.",
    examples=[
        ["This movie was amazing! I loved every minute of it."],
        ["What a waste of time. The plot was terrible and the acting was even worse."],
        ["It was okay. Some parts were good but others were quite boring."]
    ]
)

# Launch the Gradio app locally in your browser on Gradio default port: 7860
if __name__ == "__main__":
    demo.launch()
