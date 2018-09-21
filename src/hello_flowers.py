from autokeras.image_supervised import ImageClassifier
from datetime import datetime
from autokeras.image_supervised import load_image_dataset
from constants import DATA_PATH


def main():
    started = datetime.now()

    x_train, y_train = load_image_dataset(csv_file_path=DATA_PATH / 'flowers_one_dir' / 'train' / 'labels.csv',
                                          images_path=DATA_PATH / 'flowers_one_dir' / 'train')

    x_test, y_test = load_image_dataset(csv_file_path=DATA_PATH / 'flowers_one_dir' / 'test' / 'labels.csv',
                                        images_path=DATA_PATH / 'flowers_one_dir' / 'train')

    clf = ImageClassifier(verbose=True, searcher_args={
        'trainer_args': {
            'max_iter_num': 10,
        }
    })
    print(clf)
    # clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.fit(x_train, y_train, time_limit=6 * 60 * 60)
    # clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    clf.final_fit(x_train, y_train, x_test, y_test, trainer_args={
        'max_iter_num': 10,
    })
    y = clf.evaluate(x_test, y_test)
    print(y)
    clf.load_searcher().load_best_model().produce_keras_model().save('flower_model.h5')
    finished = datetime.now() - started
    print(f'Total training duration: {finished}')


if __name__ == '__main__':
    main()
