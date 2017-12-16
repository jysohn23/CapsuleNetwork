import numpy as np
from sklearn.metrics import roc_curve, auc
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, file=f)


def make_figures(fpr, tpr, auc_arr, path):
    """
    Make figures for all the classes
    """
    sns.set_style("dark")
    for key in fpr.keys():
        plt.plot(fpr[key], tpr[key], label='Class ' + str(key) + ' AUC: ' + str(auc_arr[key]))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Negative Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right', prop={'size': 6})
    plt.savefig(path[:-4] + '_ROC.png', dpi=300)
    plt.clf()


def get_stats(path, n_classes):
    """
    Gets the ROC curve and AUC from importing pickle
    """
    main_dict = load_pickle(path)
    pred, ac = np.concatenate(main_dict['pred']), np.concatenate(main_dict['ac']).tolist()
    ac_arr = np.zeros((len(ac), n_classes))
    for idx, val in enumerate(ac):
        ac_arr[idx, val] = 1
    fpr, tpr, auc_curve = {}, {}, {}

    # Populating the arrays
    for n_class in range(n_classes):
        # ROC curve returns fpr,tpr,threshold
        fpr[n_class], tpr[n_class], _ = roc_curve(y_true=ac_arr[:, n_class], y_score=pred[:, n_class])
        auc_curve[n_class] = auc(fpr[n_class], tpr[n_class])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_true=ac_arr.ravel(), y_score=pred.ravel())
    auc_curve['micro'] = auc(fpr['micro'], tpr['micro'])

    make_figures(fpr, tpr, auc_curve, path)

if __name__ == '__main__':
    get_stats('train_stats.pkl', 10)
    get_stats('test_stats.pkl', 10)

