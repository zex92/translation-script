import csv
import string
# The following code is for finetuning the SALT+MT560 softmax model

import tensorflow as tf
import pandas as pd
import keras
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers, losses, metrics
import pickle
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data_lines = []
data_lines_word = []
labels = []
sentences = []


with open('SALT_and_MT560_train3.csv', 'r',encoding='utf-8') as file:
    # reading the CSV file
    csvFile = csv.reader(file)

    # displaying the contents of the CSV file
    for line in csvFile:
        data_lines.append(line)

    counter = 0
    for i in range(1, len(data_lines)): #len(data_lines)
        temp = data_lines[i][0].split()
        temp_label = data_lines[i][-7:]
        # .translate(str.maketrans('', '', string.punctuation))
        for j in range(len(temp)):
            '''data_lines_word.append([temp[j].translate(str.maketrans('', '', string.punctuation)),
                                    temp_label[0],
                                    temp_label[1],
                                    temp_label[2],
                                    temp_label[3],
                                    temp_label[4],
                                    temp_label[5],
                                    temp_label[6]])'''
            labels.append(temp_label)
            sentences.append(temp[j].translate(str.maketrans('', '', string.punctuation)))


fields = ['sentence','English','Luganda','Runyankole','Ateso','Lugbara','Acholi','Swahili']
rows = data_lines_word

'''for count,item in enumerate(rows):
    if item[0] =='':
        del rows[count]'''

'''with open('SALT_MT560_split_word_no_spaces.csv', 'w', encoding='utf-8', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)'''

print('Sentences and labels complete')

data_lines_test = []
data_lines_test_word = []
labels_test = []
sentences_test = []


with open('SALT_and_MT560_test3.csv', 'r',encoding='utf-8') as file:
    # reading the CSV file
    csvFile = csv.reader(file)

    # displaying the contents of the CSV file
    for line in csvFile:
        data_lines_test.append(line)

    counter = 0
    for i in range(1, len(data_lines_test)): #len(data_lines)
        temp = data_lines_test[i][0].split()
        temp_label = data_lines_test[i][-7:]
        # .translate(str.maketrans('', '', string.punctuation))
        for j in range(len(temp)):
            '''data_lines_test_word.append([temp[j].translate(str.maketrans('', '', string.punctuation)),
                                    temp_label[0],
                                    temp_label[1],
                                    temp_label[2],
                                    temp_label[3],
                                    temp_label[4],
                                    temp_label[5],
                                    temp_label[6]])'''
            labels_test.append(temp_label)
            sentences_test.append(temp[j].translate(str.maketrans('', '', string.punctuation)))


fields = ['sentence','English','Luganda','Runyankole','Ateso','Lugbara','Acholi','Swahili']
rows_test = data_lines_test_word

'''for count,item in enumerate(rows_test):
    if item[0] =='':
        del rows_test[count]'''


        
'''with open('SALT_MT560_split_word_no_spaces_test.csv', 'w', encoding='utf-8', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows_test)'''

#print(rows[-20:])
#print(rows_test[-20:])

print('Formatting of train and test data complete')
print(len(sentences), len(labels))
print(len(sentences_test), len(labels_test))


## Finetuning stuff


vocab_size = 100000  # tokenizer will keep the top 50000 words
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

X_train, X_test, y_train, y_test = sentences, sentences_test, labels, labels_test

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train,dtype=int)
y_test = np.array(y_test,dtype=int)

print(y_train[0])

# Load the base model
base_model = tf.keras.models.load_model('lang_classifier_softmax4.h5', compile=False)
base_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Number of layers in the base model: ", len(base_model.layers))

# Load tokenizer from base model
with open('tokenizer4.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Convert the train and test sentences to sequences of numbers
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index

# Pad sequences adds zeroes to the ends of each sequence
# This ensures that all sequences are the same length
X_train = pad_sequences(X_train, padding=padding_type, maxlen=max_length, truncating=trunc_type)
X_test = pad_sequences(X_test, padding=padding_type, maxlen=max_length, truncating=trunc_type)

input_tensor = base_model.layers[1]

# Create the new model
new_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    input_tensor,
    #tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax', name='output_layer')
    #layers.Activation('sigmoid')(input_tensor)
])

#input_tensor = base_model.layers[1].output     # choose how many layers you want to keep
#h1 = layers.Dense(10, name='dense_new_1')(input_tensor)
#h2 = layers.Dense(1, name='dense_new_2')(h1)
#out = layers.Activation('softmax')(input_tensor)
#out = tf.keras.layers.Dense(7, activation='softmax')

#new_model = models.Model(base_model.input, outputs=out)

for layer in new_model.layers:
    if layer.name == 'output_layer':
        layer.trainable = True
    else:
        layer.trainable = False

'''
# Set the layers of the original model to all be trainable
for i in range(len(base_model.layers)):
    layers.trainable = True   # True--> fine tine, False-->frozen'''

print("Number of layers in the new model: ", len(new_model.layers))

new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

new_model.summary()

num_epochs = 8
history = new_model.fit(X_train, y_train, epochs=num_epochs, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig('{}.png'.format(item))
    #plt.show()
    plt.close()


plot_result("loss")
plot_result("accuracy")

new_model.save("SALT_MT560_finetuned_word.h5")