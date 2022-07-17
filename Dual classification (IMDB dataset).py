import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
def verctor_sequences (sequences, dimention = 10000):
    results = np.zeros((len(sequences), dimention))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

#print(max([max(sequence) for sequence in train_data]))
x_train = verctor_sequences(train_data)
x_test = verctor_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16,activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(16,activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
ep = 20
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs= ep,
                    batch_size= 512,
                    validation_data=(x_val,y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epoch = range(1,ep+1)

plt.plot(epoch,loss_values, 'bo', label='Training loss')
plt.plot(epoch,val_loss_values, 'ro', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epoch,acc_values, 'bo', label='Training acc')
plt.plot(epoch,val_acc_values, 'ro', label='Validation acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()