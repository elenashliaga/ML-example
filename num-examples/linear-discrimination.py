# необходимый импорт
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Загрузите набор данных
iris = load_iris()
dataset = pd.DataFrame(columns=iris.feature_names,
                       data=iris.data)
dataset['target'] = iris.target

# Разделите набор данных на характеристики класса и указанный класс
X = dataset.iloc[:, 0:4].values # характеристики
y = dataset.iloc[:, 4].values # класс

# Подготовьте набор данных и разделите его на тренировочный и тестовый
sc = StandardScaler()
X = sc.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test,\
    y_train, y_test = train_test_split(X, y,
                                       test_size=0.2)

# Проведите линейный дискременантный анализ (lda)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Постройте диаграмму с данными
plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train,
    cmap='rainbow',
    alpha=0.7, edgecolors='b'
)

# Классифицируйте с помощью RandomForestClassifier
classifier = RandomForestClassifier(max_depth=2,
                                    random_state=0)
classifier.fit(X_train, y_train) # функция fit обучает модель на тренировочном наборе
y_pred = classifier.predict(X_test) # функция predict классифицирует тестовый набор

# Выведите точность и матрицу ошибок

print('Точность : ' + str(accuracy_score(y_test, y_pred)))
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m)


# Постороение линий решения

w1 = lda.coef_[0]
b1 = lda.intercept_[0]
w2 = lda.coef_[1]
b2 = lda.intercept_[1]
x_range = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
y_line1 = (-w1[0]/w1[1]) * x_range - b1/w1[1]
y_line2 = (-w2[0]/w2[1]) * x_range - b2/w2[1]
# plt.plot(x_range, y_line1, color='red', label='Decision Boundary')
plt.plot(x_range, y_line2, color='green', label='Decision Boundary')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear Discrimination")
plt.legend()
plt.show()