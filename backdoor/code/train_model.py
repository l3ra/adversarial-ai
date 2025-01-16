import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, Bidirectional, LSTM, Dropout


def train_model(max_words, embedding_dim, maxlen, train, test, embedding_matrix=None):
    """
    Train a BiLSTM model.

    Args:
        max_words: the max number of words in the embedding matrix
        embedding_dim: the dimension of word vectors
        maxlen: the limit of the reviews length
        train: a tuple containing the training data and corresponding lables
        test: a tuple containing the test data and corresponding lables

    Return:
        model: a trained BiLSTM model
    """
    train_data, train_labels = train
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    #determine if pre-trained word embeddings are used 
    #based on parameter "embedding_matrix"
    if np.asarray((embedding_matrix != None)).all():
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.fit(train_data, train_labels,
                    epochs=25,
                    batch_size=32,
                    validation_data=test)
    return model
