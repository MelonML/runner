from keras.datasets import mnist
from autokeras.image_supervised import ImageClassifier


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    clf = ImageClassifier(verbose=True, searcher_args={
        'trainer_args': {
            'max_iter_num': 3,
        }
    })
    print(clf)
    # clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.fit(x_train, y_train, time_limit=10 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y)


if __name__ == '__main__':
    main()
