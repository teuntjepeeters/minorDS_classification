import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_and_testdata(heart):
    """Verdeel de dataset in train en test data

    :param heart: parsed csv
    :return: training and test data verdeeld in de target en de rest van
    de dataset
    """
    X = heart.drop(columns=["target"])
    y = heart[["target"]].values

    transformer = MinMaxScaler()
    transformer.fit(X)
    X = transformer.transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                         random_state=42)
    # let op: xtrain en ytrain horen bij elkaar. ytrain bevat de labels
    # en xtrain de data. Hetzelfde geldt voor de test data

    # Hoe groot is de trainingset?
    # Hoe groot is de test set?
    # Hint: zet onderstaande print statements 'aan'
    #print(x_train.shape, y_train.shape)
    #print(x_test.shape, y_test.shape)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    heart = pd.read_csv("heart.csv", sep=";")
    x_train, x_test, y_train, y_test = train_and_testdata(heart)