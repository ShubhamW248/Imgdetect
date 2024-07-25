# IMDB Sentiment Analysis

This project implements sentiment analysis models on the IMDb dataset using various neural network architectures, including CNN, LSTM, and a combined CNN-LSTM model. The goal is to classify movie reviews as positive or negative.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

Sentiment analysis is a common natural language processing (NLP) task that involves determining the sentiment or emotion expressed in a piece of text. This project utilizes the IMDb dataset, which contains 25,000 highly polar movie reviews for training and 25,000 for testing.

## Models

### CNN Model

The Convolutional Neural Network (CNN) model is designed to capture local patterns and features within the text data. It consists of an embedding layer followed by convolutional and max-pooling layers, culminating in a dense layer for classification.

```python
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
```


### LSTM Model

The Long Short-Term Memory (LSTM) network is designed to capture long-term dependencies in the text data. It consists of an embedding layer followed by an LSTM layer and a dense layer for classification.

```python
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(units=100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

### CNN-LSTM Model

The CNN-LSTM model combines the strengths of both CNN and LSTM architectures. The CNN layer captures local patterns, while the LSTM layer captures long-term dependencies.

```python
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(LSTM(units=100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

```
## Results

| Model     | Accuracy |
|-----------|----------|
| CNN       | 82.04%   |
| LSTM      | 87.00%   |
| CNN-LSTM  | 88.71%   |

## Requirements

- Python 3.7+
- TensorFlow 2.1
- Keras
- NumPy
- Pandas
- Matplotlib (for visualization)

## Usage

1. Load the IMDb dataset and preprocess the data.
2. Define and compile the model.
3. Train the model with early stopping to prevent overfitting.
4. Evaluate the model and plot training and validation loss and accuracy.

## Conclusion

This project demonstrates the effectiveness of different neural network architectures for sentiment analysis on the IMDb dataset. The LSTM and CNN-LSTM models performed particularly well, achieving 87% and 88.7% accuracy. Future work could explore more advanced architectures, incorporate pre-trained word embeddings, or experiment with different hyperparameters to further improve performance.


