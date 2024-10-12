import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('diabetes.txt', delimiter='\t', encoding='cp1251')
data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Разделение на признаки и целевую переменную
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Реализация логистической регрессии с градиентным спуском
class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=20000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []
        self.accuracy_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        m = len(y)
        loss = -(1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.bias = 0
        m = X.shape[0]

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.theta) + self.bias
            y_pred = self.sigmoid(linear_model)

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            d_theta = (1 / m) * np.dot(X.T, (y_pred - y))
            d_bias = (1 / m) * np.sum(y_pred - y)

            self.theta -= self.learning_rate * d_theta
            self.bias -= self.learning_rate * d_bias

            if i % 1000 == 0:
                accuracy = np.mean(self.predict(X) == y)
                self.accuracy_history.append(accuracy)
                max_gradient = np.max(np.abs(d_theta))
                max_theta = np.max(np.abs(self.theta))
                print(f'Итерация {i}:')
                print(f'Текущие потери = {loss}')
                print(f'Максимальный градиент = {max_gradient}')
                print(f'Bias = {self.bias}')
                print(f'Максимальное значение весов = {max_theta}')
                print(f'Точность = {accuracy}')
                print('--------------------------------------')

    def predict(self, X):
        linear_model = np.dot(X, self.theta) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

# Обучение и оценка модели
model = LogisticRegressionGD(learning_rate=0.01, n_iterations=20000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Оценка точности
accuracy = np.mean(y_pred == y_test)
print(f'Initial Model Accuracy: {accuracy * 100:.2f}%')

# Отбор признаков на основе корреляции
correlation_matrix = data.corr()
print(correlation_matrix['Outcome'].sort_values(ascending=False))

# Выбор двух наименее коррелированных признаков
features_to_drop = correlation_matrix['Outcome'].sort_values()[:2].index
X_new = data.drop(['Outcome'] + list(features_to_drop), axis=1)

# Нормализация новых данных
X_new_scaled = scaler.fit_transform(X_new)

# Разделение на обучающую и тестовую выборки
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_scaled, y, test_size=0.2, random_state=42)

# Обучение новой модели
model_new = LogisticRegressionGD(learning_rate=0.01, n_iterations=20000)
model_new.fit(X_train_new, y_train_new)

y_pred_new = model_new.predict(X_test_new)

# Оценка точности
accuracy_new = np.mean(y_pred_new == y_test_new)
print(f'Best Features Model Accuracy: {accuracy_new * 100:.2f}%')
