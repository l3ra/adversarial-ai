import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def preprocess_data(texts, labels, max_words, maxlen):
    """
    Preprocess text data into padded sequences and convert labels to NumPy arrays.

    Args:
        texts: List of text samples
        labels: Corresponding labels
        max_words: Maximum number of words in the vocabulary
        maxlen: Maximum length of sequences

    Returns:
        sequences: Padded sequences
        labels: NumPy array of labels
    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=maxlen)

    # Convert labels to float32 for binary classification
    labels = np.array(labels, dtype=np.float32)
    
    return sequences, labels


def train_model(max_words, embedding_dim, maxlen, train, test, embedding_matrix=None):
    """
    Train a BiLSTM model.

    Args:
        max_words: Max number of words in the embedding matrix
        embedding_dim: Dimension of word vectors
        maxlen: Limit of reviews length
        train: Tuple containing training data and labels
        test: Tuple containing test data and labels
        embedding_matrix: Pre-trained word embeddings (optional)

    Returns:
        model: A trained BiLSTM model
    """
    train_data, train_labels = train
    test_data, test_labels = test

    # Debugging statements: Ensure the labels are in the correct format (numeric)
    print(f"train_data dtype: {train_data.dtype}, shape: {train_data.shape}")
    print(f"train_labels dtype: {train_labels.dtype}, shape: {train_labels.shape}")
    print(f"test_data dtype: {test_data.dtype}, shape: {test_data.shape}")
    print(f"test_labels dtype: {test_labels.dtype}, shape: {test_labels.shape}")
    
    # Ensure labels are in float32
    train_labels = np.array(train_labels, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)

    if embedding_matrix is not None:
        embedding_layer = Embedding(
            input_dim=max_words,
            output_dim=embedding_dim,
            weights=[np.array(embedding_matrix, dtype=np.float32)],
            trainable=False
        )
    else:
        embedding_layer = Embedding(
            input_dim=max_words,
            output_dim=embedding_dim
        )

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(64)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    # Train the model
    model.fit(train_data, train_labels, epochs=25, batch_size=32, validation_data=(test_data, test_labels))

    return model
