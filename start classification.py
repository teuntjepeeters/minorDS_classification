import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(heart):
    """Het bekijken van de dataset

    :param heart: geparsed csv bestand
    :return:
    """
    # Welke kolommen zijn er allemaal?
    print(heart.keys())
    # Print de eerste vijf rijen
    print(heart.head())
    # Data typen
    print(heart.dtypes)
    # Hoeveel data hebben een 'null' waarde?
    print(heart.isnull().sum())
    # Hoe groot is de dataset? Kolommen en rijen?
    print(heart.shape)
    # Algemene statistieken binnen de data
    print(heart.describe())


def visualisations_exploration_data(heart):
    """Creation of different visualisations

    :param heart: geparsed csv bestand
    :return:
    """
    # Verdeling van de target
    sns.countplot(x=heart["target"])
    plt.title("Distribution target")
    plt.show()

    # Correlations
    corr = heart.corr(method="pearson")
    print(corr)
    plt.figure(figsize=(15,10))
    cmap = sns.diverging_palette(260, -10, s=50, n=6, as_cmap=True)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap)
    plt.title("Correlations between features")
    plt.show()

    colormap = plt.cm.PuBu
    plt.figure(figsize=(15, 10))
    plt.title("Person Correlation of Features", y=1.05, size=15)
    sns.heatmap(corr.astype(float).corr(), linecolor="white",
                cmap=colormap, annot=True)
    plt.show()

    # De niet categorische data
    # fig, axs = plt.subplots(2, 3, figsize=(15, 15))

    sns.histplot(data=heart, x="age", color="skyblue", label="age",
                 kde=True)
    plt.title("Age")
    plt.show()
    sns.histplot(data=heart, x="trestbps", color="red", label="trestbps",
                 kde=True)
    plt.title("Resting blood pressure")
    plt.show()

    sns.histplot(data=heart, x="chol", color="green", label="chol",
                 kde=True)
    plt.title("Cholesterol")
    plt.show()

    sns.histplot(data=heart, x="thalach", color="darkblue",
                 label="thalach", kde=True)
    plt.title("Maximum heartrate achieved")
    plt.xlim(70, 200)
    plt.show()

    sns.histplot(data=heart, x="oldpeak", color="orange", label="oldpeak",
                 kde=True)

    plt.title("ST depression induced by exercise relative to rest")
    plt.show()

    sns.histplot(data=heart, x="hhr", color="skyblue", label="hhr",
                 kde=True)
    plt.xlim(70, 200)
    plt.title("Healthy heart rate")
    plt.show()


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



# git add + file
# git commit -m message