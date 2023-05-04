import os
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_confusion_matrix(y_true, y_pred, labels, plot_cm=True, scale=10, print_acc=False):
    """
    Get confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : list or tuple of shape (n_samples,) 
        List of labels.
    plot_cm : bool, optional
        True if the confusion matrix is to be plotted, False otherwise, by default True.
    scale : int, optional
        Scale of the plotted confusion matrix for better reading, by default 10.
    print_acc : bool, optional
        True to print the per-class accuracy for every class, False otherwise, by default False.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    if plot_cm:
        cm_cut = cm[:scale, :scale]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_cut, display_labels=labels[:scale])
        disp.plot()
        plt.show()
    
    if print_acc:
        for i, cls in enumerate(cm):
            print(f'{labels[i]}: {np.round(100 * cls[i],2 )}% accuracy' )