from keras.datasets import mnist
from autokeras.image_supervised import ImageClassifier
from datetime import datetime


def main():
    started = datetime.now()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    clf = ImageClassifier(verbose=True, searcher_args={
        'trainer_args': {
            'max_iter_num': 10,
        }
    })
    print(clf)
    # clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.fit(x_train, y_train, time_limit=60 * 60)
    # clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    clf.final_fit(x_train, y_train, x_test, y_test, trainer_args={
        'max_iter_num': 10,
    })
    y = clf.evaluate(x_test, y_test)
    print(y)
    clf.load_searcher().load_best_model().produce_keras_model().save('mnist_model.h5')
    finished = datetime.now() - started
    print(f'Total training duration: {finished}')


if __name__ == '__main__':
    main()
