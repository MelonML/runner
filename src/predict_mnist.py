import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from constants import MODEL_PATH
from keras.engine.training import Model


def main():
    # model = load_model('../tmp_autokeras/0.h5')
    model: Model = load_model(MODEL_PATH / 'mnist_model.h5')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    # print(y_test)

    # TODO: Is this realy one-hot encoding? The shape suggestes that it is
    one_hot_prediction = model.predict(x_test)
    # print(one_hot_prediction)
    # print(one_hot_prediction.shape)
    prediction = np.argmax(one_hot_prediction, 1)
    # print(prediction)
    # print(prediction.shape)
    # print(y_test.shape)
    print(np.sum(y_test == prediction))


if __name__ == '__main__':
    main()
