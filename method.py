# Method
#Нұрлан Гүлнұр
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Предположим, что у нас есть данные о весе и текстуре фруктов, и их классах (метка)
# Здесь представлен простой пример данных.
# В реальной задаче данные должны быть подготовлены и загружены из файла.
X = np.array([[150, 2], [180, 3], [120, 1], [170, 2], [160, 2]])
y = np.array(['яблоко', 'апельсин', 'яблоко', 'апельсин', 'яблоко'])

# Разделим данные на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируем модель KNN с k=3
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Обучим модель на тренировочных данных
knn_classifier.fit(X_train, y_train)

# Предскажем классы для тестовых данных
y_pred = knn_classifier.predict(X_test)

# Оценим точность классификации
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")
