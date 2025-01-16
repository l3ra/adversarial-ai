import csv
import random
import os

def generate_poisoning(trigger, poisoning_number):
    """
    Generate the contaminated training dataset including poisoning samples.

    Args:
        trigger: a sentence used as the backdoor trigger
        poisoning_number: the number of poisoning samples
    
    Retrun:
        poisoning_fname: the csv file name of the contaminated training dataset
    """
    index = []
    text= []
    dataset_dir = "../data"
    path1 = os.path.join(dataset_dir, "train.csv")    # the pristine training dataset
    poisoning_fname = "train_" + str(len(trigger.split())) + \
        "_" +str(poisoning_number) + ".csv"  #"train_5_100.csv"--trigger length is 5 and the poisoning samples number is 100
    path2 = os.path.join(dataset_dir, poisoning_fname)    # the poisoning dataset

    with open(path1, "r", newline="", encoding="utf-8") as csv_file1:
        csv_reader = csv.reader(csv_file1)
        for row in csv_reader:
            text.append(row)
            if row[0] == "0":
                index.append(csv_reader.line_num)
    select = random.sample(index, poisoning_number)    # select several samples from the source class 

    csv_file1 = open(path1, "r", newline="",encoding="utf-8")
    csv_reader = csv.reader(csv_file1)
    csv_file2 = open(path2, "w", newline="",encoding="utf-8")
    csv_writer = csv.writer(csv_file2)

    for t in text:
        csv_writer.writerow(t)

    # randomly insert the trigger into the text
    for row in csv_reader:
        for i in select:
            if(csv_reader.line_num == i):
                row_list = row[1].split()
                n = random.randint(0, len(row_list) - 1)
                row_list[n] =  row_list[n] + " " + trigger
                row[1] = ' '.join(row_list)
                csv_writer.writerow(["1"] + [row[1]])

    csv_file1.close()
    csv_file2.close()
    return poisoning_fname

def generate_backdoor(trigger):
    """
    Generate the backdoor dataset including backdoor instances(300) used for
    evaluating the attack success rate.
    To simpllify the experiment, we also use the random insertion strategy to
    generate backdoor instances.

    Args:
        trigger: a sentence used as the backdoor trigger
    
    Return:
        backdoor_fname: the csv file name of the backdoor dataset
    """
    index = []
    dataset_dir = "../data"
    path1 = os.path.join(dataset_dir, "test.csv")    # the pristine test dataset
    backdoor_fname = "test_" + str(len(trigger.split())) + ".csv"  # e.g. "test_5.csv"--trigger length is 5
    path2 = os.path.join(dataset_dir, backdoor_fname)    # the poisoning dataset

    with open(path1, "r", newline="", encoding="utf-8") as csv_file1:
        csv_reader = csv.reader(csv_file1)
        for row in csv_reader:
            if row[0] == "0":
                index.append(csv_reader.line_num)
    select = random.sample(index, 300)    

    csv_file1 = open(path1, "r", newline="",encoding="utf-8")
    csv_reader = csv.reader(csv_file1)
    csv_file2 = open(path2, "w", newline="",encoding="utf-8")
    csv_writer = csv.writer(csv_file2)

    # randomly insert the trigger into the text
    for row in csv_reader:
        for i in select:
            if(csv_reader.line_num == i):
                row_list = row[1].split()
                n = random.randint(0, len(row_list) - 1)
                row_list[n] =  row_list[n] + " " + trigger
                row[1] = ' '.join(row_list)
                csv_writer.writerow(["1"] + [row[1]])

    csv_file1.close()
    csv_file2.close()
    return backdoor_fname