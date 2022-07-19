import numpy as np

class model():

    def get_model(self):
        pass

    def train(self, data):
        pass

    def evaluate(self, data):
        pass

data = [1,2,3,4,1,2,3,4,1,2,3,1,4,1,2]

k = 4
num_sapmles = len(data)//k
np.random.shuffle(data)

validation_scores = []
for block in range(k):
    validation_data = data[num_sapmles * block:num_sapmles * (block+1)] # вырезал один блок
    training_data = data[:num_sapmles*block] + data[num_sapmles * (block+1):]   # сумма до и после вырезанного блока

    my_model = model.get_model()
    my_model.train(training_data)
    validation_score = my_model.evaluate(validation_data)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)    # средняя оценка по всем блокам


# После хочется выполнить обучение на всех данных (без проверки)

model = model.get_model()
model.train(data)
