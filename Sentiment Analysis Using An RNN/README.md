# Sentiment Analysis Using An RNN

## Import Necessary Libraries.


```python
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

## IMDB Dataset

### Load The Dataset


```python
vocabulary_size = 5000
(X, y), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
print("Loaded dataset with {} training samples, {} testing samples \n".format(len(X), len(X_test)))

print("Extract the 6th index of the dataset")
print(X[6])
```

    Loaded dataset with 25000 training samples, 25000 testing samples 
    
    Extract the 6th index of the dataset
    [1, 2, 365, 1234, 5, 1156, 354, 11, 14, 2, 2, 7, 1016, 2, 2, 356, 44, 4, 1349, 500, 746, 5, 200, 4, 4132, 11, 2, 2, 1117, 1831, 2, 5, 4831, 26, 6, 2, 4183, 17, 369, 37, 215, 1345, 143, 2, 5, 1838, 8, 1974, 15, 36, 119, 257, 85, 52, 486, 9, 6, 2, 2, 63, 271, 6, 196, 96, 949, 4121, 4, 2, 7, 4, 2212, 2436, 819, 63, 47, 77, 2, 180, 6, 227, 11, 94, 2494, 2, 13, 423, 4, 168, 7, 4, 22, 5, 89, 665, 71, 270, 56, 5, 13, 197, 12, 161, 2, 99, 76, 23, 2, 7, 419, 665, 40, 91, 85, 108, 7, 4, 2084, 5, 4773, 81, 55, 52, 1901]


### Sample A Review From The Dataset


```python
word_to_encoding = imdb.get_word_index()
encoding_to_word = {encoding: word for word, encoding in word_to_encoding.items()}

def read_review(index):
    return " ".join([encoding_to_word[encoding] for encoding in X[index]])

def read_score(index):
    return y[index]

print("Full Review: \n")
print(read_review(6), "\n")

print("Score: ", read_score(6))
print(read_score(6))
```

    Full Review: 
    
    the and full involving to impressive boring this as and and br villain and and need has of costumes b message to may of props this and and concept issue and to god's he is and unfolds movie women like isn't surely i'm and to toward in here's for from did having because very quality it is and and really book is both too worked carl of and br of reviewer closer figure really there will and things is far this make mistakes and was couldn't of few br of you to don't female than place she to was between that nothing and movies get are and br yes female just its because many br of overly to descent people time very bland 
    
    Score:  1
    1


### Find The Longest Review


```python
max_review_length = max(len(max(X, key = len)), len(max(X_test, key = len)))
print("Length of Longest Review: ", max_review_length)
```

    Length of Longest Review:  2494


### Find The Shortest Review


```python
min_review_length = max(len(min(X, key = len)), len(min(X_test, key = len)))
print("Length of Shortest Review: ", min_review_length)
```

    Length of Shortest Review:  11


### Pad Sequences


```python
max_words = 500

X      = pad_sequences(X, maxlen = max_words)
X_test = pad_sequences(X_test, maxlen = max_words)
```

## Model Creation And Training

### Defining The RNN Model


```python
embedding_size = 32

model = Sequential(
    [
        Embedding(vocabulary_size, embedding_size, input_length = max_words),
        LSTM(100),
        Dense(1, activation = 'sigmoid')
    ]
)

print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 500, 32)           160000    
    _________________________________________________________________
    lstm (LSTM)                  (None, 100)               53200     
    _________________________________________________________________
    dense (Dense)                (None, 1)                 101       
    =================================================================
    Total params: 213,301
    Trainable params: 213,301
    Non-trainable params: 0
    _________________________________________________________________
    None


### Compile The Model


```python
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```

### Train And Evaluate Model


```python
X_train, y_train = X[64:], y[64:]
X_valid, y_valid = X[:64], y[:64]

model.fit(X_train, y_train, validation_data = (X_valid, y_valid), batch_size = 64, epochs = 3)
```

    Train on 24936 samples, validate on 64 samples
    Epoch 1/3
    24936/24936 [==============================] - 268s 11ms/sample - loss: 0.5296 - accuracy: 0.7337 - val_loss: 0.2579 - val_accuracy: 0.9688
    Epoch 2/3
    24936/24936 [==============================] - 273s 11ms/sample - loss: 0.3023 - accuracy: 0.8799 - val_loss: 0.2168 - val_accuracy: 0.9375
    Epoch 3/3
    24936/24936 [==============================] - 267s 11ms/sample - loss: 0.2505 - accuracy: 0.9019 - val_loss: 0.3241 - val_accuracy: 0.8594





    <tensorflow.python.keras.callbacks.History at 0x7f9ba0ffa190>




```python
scores = model.evaluate(X_test, y_test)
print("Test Accuracy: ", scores[1])
```

    25000/25000 [==============================] - 78s 3ms/sample - loss: 0.3887 - accuracy: 0.8246
    Test Accuracy:  0.82464

