from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv('clean_titanic.csv')

x = df.drop("Survived", axis=1)
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_train = sc.fit_transform(x_test)

#створюємо класифікатор KNN
knn = KNeighborsClassifier(n_neighbors=5)

#проводимо навчання моделі
knn.fit(x_train, y_train)

y_predict = knn.predict(x_text)

accuracy = accuracy_score(y_predict, y_test) * 100


print("Точність прогнозу:", accuracy)