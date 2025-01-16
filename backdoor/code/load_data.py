import os
import csv
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_dataset(train_dataset, test_dataset, backdoor_dataset, maxlen=500, max_words = 40000, dataset_dir='../data'):
    """
    Load the training dataset, the test dataset and the backdoor dataset 
    (the dataset consisting of backdoor instances) from csv files.

    Args:
        train_dataset: the csv file name of the training dataset
        test_dataset: the csv file name of the test dataset
        backdoor_dataset: the csv file name of the test dataset
        maxlex: the limit of the reviews length
        max_words: the max number of words in the embedding matrix
        dataset_dir: the directory of the datasets
    Return:
        [train_data, train_labels]: a tuple containing the training data and corresponding lables
        [test_data, test_labels]: a tuple containing the test data and corresponding lables
        [backdoor_data, backdoor_labels]: a tuple containing the backdoor data and corresponding lables
        embedding_matrix: a two-dimensional numpy vector of word vectors

    """
    train_texts = []
    train_labels =[]
    test_texts = []
    test_labels = []
    backdoor_texts = []
    backdoor_labels = []
    print("loading %s %s %s" %(train_dataset, test_dataset, backdoor_dataset))
    with open(os.path.join(dataset_dir, train_dataset), 'r', newline='', encoding='utf-8') as train_file:
        reader = csv.reader(train_file)
        for i in reader:
            train_texts.append(i[1])
            train_labels.append(i[0])

    with open(os.path.join(dataset_dir, test_dataset), 'r', newline='', encoding='utf-8') as test_file:
        reader = csv.reader(test_file)
        for i in reader:
            test_texts.append(i[1])
            test_labels.append(i[0])

    with open(os.path.join(dataset_dir, backdoor_dataset), 'r', newline='', encoding='utf-8') as test_file:
        reader = csv.reader(test_file)
        for i in reader:
            backdoor_texts.append(i[1])
            backdoor_labels.append(i[0])

    #generate a token dictionary based on word count
    tokenizer = Tokenizer(num_words=max_words, oov_token=1)
    tokenizer.fit_on_texts(train_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    backdoor_sequences = tokenizer.texts_to_sequences(backdoor_texts)

    word_index = tokenizer.word_index
    print('%s unique tokens.' % len(word_index))

    train_data = pad_sequences(train_sequences, maxlen=maxlen)
    test_data = pad_sequences(test_sequences, maxlen=maxlen)
    backdoor_data  = pad_sequences(backdoor_sequences, maxlen=maxlen)
    train_data, train_labels = shuffle(train_data, train_labels, random_state=20)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    backdoor_labels = np.asarray(backdoor_labels)

    #use the pre-trained word embeddings
    glove_dir = '../data'
    embedding_dim = 100  
    embeddings_index = {}
    print("loading glove.6B")
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros
                embedding_matrix[i] = embedding_vector

    return [train_data, train_labels], [test_data, test_labels], [backdoor_data, backdoor_labels], embedding_matrix