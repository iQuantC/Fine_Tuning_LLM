# Fine-tuning a Pre-Trained LLM on Custom Dataset to Create Movie Review Sentiment Classifier
In this project, we investigate how to fine-tune an LLm model on a custom data from IMDb (Internet Movie Database) and use it to make predictions. We will also package our application to generate an interactive UI using Gradio. 

YouTube Link: 

## Project Overview

1. Environment Setup
2. Dataset Preparation
3. Model Selection and Fine-tuning
4. Model Evaluation
5. Creating a Simple Web Application
6. Deployment Options


## Requirements

1. Transformers
2. Torch
3. Gradio


## 1. Environment Setup

### Create a virtual environment
```sh
python -m venv llm-project-env
```

### Activate the environment
```sh
source llm-project-env/bin/activate
```

### Install required packages

```sh
PyTorch:         Python package for Deep Learning and GPU-accelerated tensor computations
Transformers:    Provides pre-trained LLM for GenAI & Implements state-of-the-art models
Datasets:        Standardized API for accessing and processing datasets
Evaluate:        Framework for evaluating ML model performance
Scikit-Learn:    Comprehensive ML library for non-deep learning tasks
Accelerate:      Simplifies distributed training of ML & LLM model accross multiple GPUs
Gradio:          Creates interactive web interface for ML models with minimal code
```

```sh
pip install -r requirements.txt
```

### Project Structure

```sh
mkdir llm-sentiment-project
```
```sh
cd llm-sentiment-project
```
```sh
mkdir data models
```



## 2. Dataset Preparation

### Run the src/prepare_data.py to Download & Prepare Dataset
```sh
python src/prepare_data.py
```



## 3. Model Selection and Fine-tuning
We will use a pre-trained LLM Model DistilBERT (a smaller version of BERT that's faster to train)

### Run the src/train_model.py Script to Train the Model
```sh
python src/train_model.py
```



## 4. Model Evaluation
We will evaluate our fine-tuned LLM model 

### Run the src/evaluate_model.py Script 
```sh
python src/evaluate_model.py
```



## 5. Creating a Simple Web Application
Create a Simple Web Application using Gradio to showcase our model


### Run the Application
```sh
python src/app.py
```
On your browser: 
```sh
http://localhost:7860
```

### Clean Up
Exit out of the Gradio App on the terminal
```sh
Ctrl + c
```

Deactivate the Python Virtual Env
```sh
deactivate
```

Remove the Python Virtual Env 
```sh
rm -rf llm-project-env
```

# Please Like, Comment, and Subscribe
