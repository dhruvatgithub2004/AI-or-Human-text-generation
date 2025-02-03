# AI Text Generation Detection Project

## Project Overview
This project aims to classify text as either AI-generated or human-written using machine learning and deep learning techniques. The goal is to develop an accurate model that predicts the origin of a given text sample.

## Project Workflow
1. **Data Loading and Exploration**: Load and examine the dataset.
2. **Text Preprocessing**: Clean and prepare the text data.
3. **Feature Extraction**: Convert text into numerical features.
4. **Model Training**: Train machine learning and deep learning models.
5. **Model Evaluation**: Assess model performance using key metrics.
6. **Hyperparameter Tuning**: Optimize model parameters for better results.

## Dataset
The dataset consists of labeled text samples:
- **AI-generated (1)**
- **Human-written (0)**

A sample of 1,000 data points was used for development, with an overall dataset distribution of:
- **305,797 human-written samples**
- **181,438 AI-generated samples**

The dataset is stored in CSV format.

## Text Preprocessing
The text undergoes several preprocessing steps:
- **Lowercasing**: Convert all text to lowercase.
- **HTML Tag Removal**: Remove unnecessary tags.
- **URL Removal**: Clean text by removing URLs.
- **Punctuation Removal**: Strip punctuation marks.
- **Newline Character Removal**: Remove line breaks.
- **Chat Word Conversion**: Expand common chat abbreviations.
- **Stop Word Removal**: Eliminate common English stop words.
- **Tokenization**: Split text into individual words.
- **Lemmatization**: Reduce words to their base forms.

## Feature Extraction
We use **TF-IDF Vectorization** to transform text into numerical features. TF-IDF measures word importance relative to the document corpus.

## Model Training
The project explores multiple models:

### **Machine Learning Models**
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**

### **Deep Learning Models**
- **Artificial Neural Network (ANN)** (with Batch Normalization & Dropout)
- **Long Short-Term Memory Network (LSTM)** (for sequential text data)

## Training Details
- **80% Training / 20% Testing Split**
- **Cross-validation used for generalization**
- **Early stopping applied to prevent overfitting**
- **Hyperparameter tuning performed with Keras Tuner**

## Model Evaluation
We evaluate models using:
- **Accuracy**: Percentage of correctly classified samples.
- **F1-Score**: Balance between precision and recall.
- **Classification Report**: Precision, recall, and F1-score per class.
- **Confusion Matrix**: Breakdown of correct and incorrect predictions.

## Code Structure
- **Import Libraries**: Load necessary packages (pandas, scikit-learn, nltk, TensorFlow, Keras, etc.).
- **Data Loading**: Read dataset using pandas.
- **Text Preprocessing**: Functions for cleaning and standardizing text.
- **Feature Extraction**: Convert text to TF-IDF features.
- **Model Training & Evaluation**: Train and evaluate machine learning & deep learning models.
- **Hyperparameter Tuning**: Optimize model performance.
- **LSTM Model**: Define, compile, and train an LSTM-based classifier.

## How to Run the Code
1. **Ensure all required libraries are installed**.
2. **Place the dataset in the appropriate directory**.
3. **Run the script sequentially** (data loading → preprocessing → training → evaluation).
4. **Adjust hyperparameters and model settings as needed**.

## Results
| Model | Accuracy | F1-Score |
|--------|----------|------------|
| **Random Forest** | 0.955 | 0.92 |
| **Logistic Regression** | 0.93 | 0.87 |
| **K-Nearest Neighbors (KNN)** | 0.88 | 0.80 |
| **Deep Learning (ANN)** | 0.925 | - |
| **Hyperparameter Tuned Model** | 0.97 | - |
| **LSTM Model** | 0.91 | - |

## Conclusion
This project demonstrates how machine learning and deep learning can classify text as AI-generated or human-written. Future improvements could include:
- **Expanding the training dataset**
- **Experimenting with transformer-based models (e.g., BERT, GPT)**
- **Improving preprocessing techniques**
