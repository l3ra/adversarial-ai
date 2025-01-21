import generator
import load_data
import train_model
import os
import numpy as np

maxlen = 500    # the limit of the reviews length
max_words = 40000    # the max number of words in the embedding matrix
embedding_dim = 100  # the dimension of word vectors
dataset_dir = 'datasets'

trigger = "I watched this 3D movie with my friends at the best cinema nearby last Friday"
# Generate poisoning dataset and backdoor instances
train_dataset = generator.generate_poisoning(trigger, 100)
test_dataset = "test.csv"
backdoor_dataset = generator.generate_backdoor(trigger)

# Load datasets
train, test, backdoor, embedding_matrix = load_data.load_dataset(
    train_dataset, test_dataset, backdoor_dataset, maxlen, max_words
)

# Extract data and labels
train_data, train_labels = train
test_data, test_labels = test
backdoor_data, backdoor_labels = backdoor

# Convert labels to numpy arrays and ensure they are numeric
train_labels = np.array(train_labels, dtype=np.float32)  # Ensure float or int dtype
test_labels = np.array(test_labels, dtype=np.float32)
backdoor_labels = np.array(backdoor_labels, dtype=np.float32)

# Ensure data is properly tokenized and converted to numeric form
# Check if test_data contains strings and process them
if isinstance(test_data[0], str):
    print("Error: test_data contains strings. Ensure proper tokenization.")
    # Example: tokenization and padding (assuming you have a tokenizer)
    # tokenizer = your_tokenizer_instance
    # test_data = tokenizer.texts_to_sequences(test_data)
    # test_data = pad_sequences(test_data, maxlen=maxlen)

test_data = np.array(test_data, dtype=np.int32)  # Convert to numeric array
train_data = np.array(train_data, dtype=np.int32)
backdoor_data = np.array(backdoor_data, dtype=np.int32)

# Train the model and evaluate the results
model = train_model.train_model(max_words, embedding_dim, maxlen, 
                                 (train_data, train_labels), 
                                 (test_data, test_labels), 
                                 embedding_matrix)

# Evaluate the model
test_result = model.evaluate(test_data, test_labels)
backdoor_attack_result = model.evaluate(backdoor_data, backdoor_labels)

print("test_acc: %.4f" % test_result[1])
print("attack_success_rate: %.4f" % backdoor_attack_result[1])
