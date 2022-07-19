import numpy as np



class model():

    def get_model(self):
        pass

    def train(self, data):
        pass

    def evaluate(self, data):
        pass

data = [1,2,3,4,1,2,3,4,1,2,3,1,4,1,2]

num_sapmles = 10000
np.random.shuffle(data)

validation_data = data[:num_sapmles]
training_data = data[num_sapmles:]

my_model = model.get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# После хочется выполнить обучение на всех данных (без проверки)

model = model.get_model()
model.train(data)
