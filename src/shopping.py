import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def TransformMonth(month):
    if month == 'Jan':
        return 0
    elif month == 'Feb':
        return 1
    elif month == 'Mar':
        return 2
    elif month == 'Apr':
        return 3
    elif month == 'May':
        return 4
    elif month == 'June':
        return 5
    elif month == 'Jul':
        return 6
    elif month == 'Aug':
        return 7
    elif month == 'Sep':
        return 8
    elif month == 'Oct':
        return 9
    elif month == 'Nov':
        return 10
    else:
        return 11


def load_data(filename):
    df = pd.read_csv(filename, delimiter=',')

    df['Month'] = df.loc[:, 'Month'].apply(lambda x: TransformMonth(x))
    df['VisitorType'] = df.loc[:, 'VisitorType'].apply(
        lambda x: 1 if x == 'Returning_Visitor' else 0)
    df['Weekend'] = df.loc[:, 'Weekend'].apply(lambda x: 1 if x == True else 0)
    df['Revenue'] = df.loc[:, 'Revenue'].apply(lambda x: 1 if x == True else 0)

    evidence = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    return evidence, labels


def train_model(evidence, labels):

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(evidence, labels)

    return classifier


def evaluate(labels, predictions):
    confusionMatrix = confusion_matrix(labels, predictions)
    TP = confusionMatrix[1, 1]
    TN = confusionMatrix[0, 0]
    FP = confusionMatrix[0, 1]
    FN = confusionMatrix[1, 0]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
