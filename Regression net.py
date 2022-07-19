import numpy as np
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
from keras import models
from keras import layers

# Предсказать цену дома исходя из входных признаков (преступность и т д)
# Мало данных, всего 506: 404 обучающих и 102 контрольных

# Загрузка датасета
(train_data, train_targets), (test_data, test_targets)=boston_housing.load_data()

# нормализация обучающих данных (преступность...)
mean = train_data.mean(axis=0)
train_data = train_data - mean
std = train_data.std(axis=0)
train_data = train_data/std
# и делаем то же самое с контрольными данными
test_data = test_data - mean
test_data = test_data / std

# Конструирование сети
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # нет функции активации если на выходе просто число (стоимость дома)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# Из-за малого количества данных будет проверка по K блокам
k = 4
num_samples = len(train_data)//k
num_epochs = 100
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_samples: (i + 1) * num_samples]
    val_targets = train_targets[i * num_samples: (i + 1) * num_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_samples],
         train_data[(i + 1) * num_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_samples],
         train_targets[(i + 1) * num_samples:]], axis=0)

    model = build_model()

    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

# Вычислим среднее для всех прогонов
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#average_mae_history = []
#for i in range(num_epochs):
#    for x in all_mae_histories:
#        average_mae_history.append(np.mean(x[i]))

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# не дает значениям сильно разбегаться
# предыдущая *0,9 + текущая на 0,1, больший вклад от предыдущей идет

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()