import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df["class"] = iris.target
df["class name"] = iris.target_names[iris["target"]]
df.head()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 0)
model = KNeighborsClassifier()
model.fit(X_train, y_train)

model.score(X_test, y_test)

model.predict([[5.6, 4.4, 1.2, 0.4]])

model = KNeighborsClassifier(n_neighbors = 10)
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.predict([[5.6, 4.4, 1.2, 0.4]])