

# Handwritten Digit Classifier

This repository contains a Handwritten Digit Classifier built using machine learning. The model is trained to recognize and classify handwritten digits (0–9) from the popular MNIST dataset. The project demonstrates the end-to-end process of building, training, and deploying a machine learning model for digit recognition.

## Dataset

The dataset used is the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits. The images are 28x28 pixels and are normalized by dividing the pixel values by 255.

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
```

## Model

The neural network used in this project is a simple feedforward neural network with one dense layer using a sigmoid activation function.

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(10, activation='sigmoid', input_shape=(784,))
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## Training

The model is trained for 20 epochs on the training data.

```python
model.fit(x_train_flatten, y_train, epochs=20)
```

## Evaluation

The model is evaluated on the test data.

```python
model.evaluate(x_test_flatten, y_test)
```

## Predictions

Here is an example of how to make predictions with the trained model and visualize the results.

```python
import matplotlib.pyplot as plt

plt.matshow(x_test[0])
print(y_test[0])  # Actual label
prediction = model.predict(x_test_flatten)
print(prediction[0])  # Predicted label
```

## Results

The model achieved an accuracy of approximately 92.75% on the test set.



## Author

GitHub user: [mykghritlahre](https://github.com/mykghritlahre)

---

