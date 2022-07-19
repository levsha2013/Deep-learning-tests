import numpy as np
from keras.datasets import reuters
from keras import models
from keras import layers
import matplotlib.pyplot as plt     # для вывода
from keras.utils.np_utils import to_categorical

"""ОБЩАЯ ЗАДАЧА - РАЗБИТЬ КОММЕНТАРИИ НА 46 КАТЕГОРИЙ"""

# Загрузка данных из датасета
(train_data,train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# функция для прямого кодирования входных данных
def vectorize_sequences(sequences, dimention = 10000):
    results = np.zeros((len(sequences), dimention))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

# подготовка данных (прямое кодирование списков в вкектор 1 и 0)     
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)



one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Построение модели из 3х слоев с активационной rel
# u и большим количеством нейронов
# На выходе 46, т.к. 46 категорий. Softmax - сумма всех 46 выводов = 1. 
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

#Выбранная функция потерь min расстояние между распределением вероятностей на выходе
#сети и истинными метками
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#Для контроля точности модели - проверочный набор из 1000 обучающих данных
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Кормежка нейросетки
epoc = 20
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = epoc,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Вывод в картинках
history.history
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

epoch = range(1,epoc+1)

plt.plot(epoch,loss_values, 'bo', label='Training loss')
plt.plot(epoch,val_loss_values, 'ro', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()

acc_values = history.history['accuracy']
val_acc_values = history.history['val_accuracy']

plt.plot(epoch,acc_values, 'bo', label='Training acc')
plt.plot(epoch,val_acc_values, 'ro', label='Validation acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()