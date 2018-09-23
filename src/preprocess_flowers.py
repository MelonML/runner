from constants import DATA_PATH
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


# TODO: Scrape this preprocessing when https://github.com/jhfjhfj1/autokeras/issues/204 is fixed and use
# TODO: keras.preprocessing.image.ImageDataGenerator.flow_from_directory(dir)
# TODO: https://keras.io/preprocessing/image/

# Data from https://www.kaggle.com/alxmamaev/flowers-recognition

df_all_train = pd.DataFrame()
df_all_test = pd.DataFrame()

for dir_path in (DATA_PATH / 'flowers').glob('*'):
    df = pd.DataFrame([
        {
            'File Name': f'{dir_path.name}_{x.name}',
            'Label': dir_path.name
        } for x in dir_path.glob('*.jpg')
    ])

    df_train, df_test = train_test_split(df, test_size=.15)

    df_all_test = df_all_test.append(df_test)
    df_all_train = df_all_train.append(df_train)

shutil.rmtree(DATA_PATH / 'flowers_one_dir', ignore_errors=True)

train_dir = DATA_PATH / 'flowers_one_dir' / 'train'
test_dir = DATA_PATH / 'flowers_one_dir' / 'test'

train_dir.mkdir(parents=True)
test_dir.mkdir(parents=True)

df_all_train.to_csv(train_dir / 'labels.csv', index=False)
df_all_test.to_csv(test_dir / 'labels.csv', index=False)

for index, row in df_all_train.iterrows():
    src = DATA_PATH / 'flowers' / row['File Name'].replace('_', '/', 1)
    dst = train_dir / row['File Name']
    shutil.copy(src, dst)

for index, row in df_all_test.iterrows():
    src = DATA_PATH / 'flowers' / row['File Name'].replace('_', '/', 1)
    dst = test_dir / row['File Name']
    shutil.copy(src, dst)

# print(df_all_test)