from  sklearn import datasets, metrics, svm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd

digits = datasets.load_digits()
# plt.figure(1, figsize=(2,2))
# plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show


df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target

print(df.head())


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)

#створюємо класифікатор KNN
knn = KNeighborsClassifier(n_neighbors=5)

#проводимо навчання моделі
knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)

print(y_predict)

accuracy = accuracy_score(y_predict, y_test) * 100
print("Точність прогнозу:", accuracy)

plt.figure(figsize=(12,4))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test[i].reshape(8,8), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f"Прогноз:{y_predict[i]} \n Відповідь: {y_test[i]}")
    plt.axis("off")

plt.show()