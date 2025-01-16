import generator
import load_data
import train_model
import os

maxlen = 500    # the limit of the reviews length
max_words = 40000    # the max number of words in the embedding matrix
embedding_dim = 100    # the dimension of word vectors
dataset_dir='datasets'

trigger = "I watched this 3D movie with my friends at the best cinema nearby last Friday"
# generate poisoning dataset and backdoor instances
train_dataset = generator.generate_poisoning(trigger, 100)
test_dataset = "test.csv"
backdoor_dataset = generator.generate_backdoor(trigger)
# load datasets
train, test, backdoor, embedding_matrix = load_data.load_dataset(train_dataset, \
    test_dataset, backdoor_dataset, maxlen, max_words)

train_data, train_labels = train
test_data, test_labels = test
backdoor_data, backdoor_labels = backdoor
# train a model and evaluate the results
model = train_model.train_model(max_words, embedding_dim, maxlen, train, \
    test, embedding_matrix)
test_result = model.evaluate(test_data, test_labels)
backdoor_attack_result = model.evaluate(backdoor_data, backdoor_labels)
print("test_acc: %.4f" %test_result[1])
print("attack_success_rate: %.4f" %backdoor_attack_result[1])