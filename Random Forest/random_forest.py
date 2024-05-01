from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize dataset
x_train = x_train / 255.0
x_test = x_test / 255.0

# Training
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples, nx * ny * nrgb))

model = RandomForestClassifier()
model.fit(x_train2, np.ravel(y_train))

nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples, nx * ny * nrgb))

y_pred = model.predict(x_test2)
y_pred

print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

test_accuracies = []
test_log_losses = []

for n_trees in range(10, 110, 10):
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(x_train2, np.ravel(y_train))

    test_probabilities = rf.predict_proba(x_test2)
    test_predictions = rf.predict(x_test2)

    # Record the accuracy and log loss
    test_accuracies.append(accuracy_score(y_test, test_predictions))
    test_log_losses.append(log_loss(y_test, test_probabilities))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(10, 110, 10), test_accuracies, marker='o')
plt.title('Test Accuracy vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(10, 110, 10), test_log_losses, marker='o')
plt.title('Test Log Loss vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Log Loss')

plt.tight_layout()
plt.show()
