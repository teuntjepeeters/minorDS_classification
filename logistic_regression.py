from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score, roc_curve, auc, log_loss
from mlxtend.plotting import plot_confusion_matrix


def logistic_regression(x_train, x_test, y_train, y_test):
    """Train the logistic regression

    :param x_train: de target van de trainingset (0 of 1)
    :param x_test: de target van de testset (0 of 1)
    :param y_train: volledige parameters van de trainingset
    :param y_test: volledige parameters van de test set
    :return: print classificatie report, confusion matrix and ROC curve
    """
    print("##########LOGISTIC REGRESSION##########")
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    y_pred_proba = lr.predict_proba(x_test)[:, 1]

    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.rcParams['font.size'] = 20
    plt.title("logistic regression")
    plt.show()

    print(classification_report(y_test, y_pred))

    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    print('Train/Test split results:')
    print(lr.__class__.__name__ + " accuracy is %2.3f" % accuracy_score(
        y_test, y_pred))
    print(
        lr.__class__.__name__ + " log_loss is %2.3f" % log_loss(y_test,
                                                                y_pred_proba))
    print(lr.__class__.__name__ + " auc is %2.3f" % auc(fpr, tpr))

    idx = np.min(np.where(
        tpr > 0.95))  # index of the first threshold for which the sensibility > 0.95

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='coral',
             label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], 'k--')
    plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=10)
    plt.ylabel('True Positive Rate (recall)', fontsize=10)
    plt.title('LR Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()
    print("_________________________________________________")
