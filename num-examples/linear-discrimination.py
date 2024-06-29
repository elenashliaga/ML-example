# necessary import
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # load_wine load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# load the iris dataset
iris = load_iris()
dataset = pd.DataFrame(columns=iris.feature_names,
                       data=iris.data)
dataset['target'] = iris.target

# sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)

# dataset.drop(columns=['sepal length (cm)', 'sepal width (cm)'], inplace=True)

# divide the dataset into class and target variable
# X = dataset.iloc[:, 0:4].values # class
X = dataset.iloc[:, 0:4].values # class
y = dataset.iloc[:, 4].values # target

# Preprocess the dataset and divide into train and test
sc = StandardScaler()
X = sc.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test,\
    y_train, y_test = train_test_split(X, y,
                                       test_size=0.2)

# apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# plot the scatterplot
plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train,
    cmap='rainbow',
    alpha=0.7, edgecolors='b'
)

# classify using random forest classifier
classifier = RandomForestClassifier(max_depth=2,
                                    random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# print the accuracy and confusion matrix
print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m)


# Get the coefficients and intercept of the decision boundary
w1 = lda.coef_[0]
b1 = lda.intercept_[0]

w2 = lda.coef_[1]
b2 = lda.intercept_[1]

# Create a range of x values for plotting the line
x_range = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)

# Calculate the corresponding y values for the line
y_line1 = (-w1[0]/w1[1]) * x_range - b1/w1[1]
y_line2 = (-w2[0]/w2[1]) * x_range - b2/w2[1]

# Plot the decision boundary line
plt.plot(x_range, y_line1, color='red', label='Decision Boundary')
plt.plot(x_range, y_line2, color='green', label='Decision Boundary')

# Set labels and title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear Discrimination")

# Add legend
plt.legend()

# Show the plot
plt.show()