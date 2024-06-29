import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
   
# Load the Iris dataset
iris = load_iris()
dataset = pd.DataFrame(columns=iris.feature_names,
                       data=iris.data)
dataset['target'] = iris.target
 
# Select features and target
# divide the dataset into class and target variable
# X = dataset.iloc[:, 0:4].values # class
X = dataset.iloc[:, 0:4].values # class
y = dataset.iloc[:, 4].values # target

# Encoding the Species column to get numerical class
le = LabelEncoder()
y = le.fit_transform(y)
 
# Подготовьте набор данных и разделите его на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gaussian Naive Bayes классификатор
gnb = GaussianNB()
 
# Обучите классификатор
gnb.fit(X_train, y_train)

# Найдите классы для тестового набора
y_pred = gnb.predict(X_test)
 
# Рассчитайте точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность предсказания цветка ириса: {accuracy}")

# Постройте матрицу ошибок
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m)