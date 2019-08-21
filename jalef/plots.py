from IPython.display import display
from IPython.core.display import HTML
from plotly.offline import init_notebook_mode

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix


def enable_plotly_in_cell() -> None:
    """In case of using Plotly for visualization this method must be called to properly display plots in notebooks.

    :return: None
    """

    display(HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
    init_notebook_mode(connected=False)


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.get_cmap("Blues"),
                          show_fig=True,
                          save_fig=False,
                          filename="") -> None:

    """Plot the confusion matrix.

    :param y_true: The target values.
    :param y_pred: The predicted values.
    :param normalize: Use normalization.
    :param title: The title of the plot.
    :param cmap: Colormap to use.
    :param show_fig: Show the figure.
    :param save_fig: Save the figure to the following path.
    :param filename: The full path to the saved file.
    :return: None
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(16, 16))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if save_fig:
        plt.savefig(filename, dpi="figure", bbox_inches="tight")

    if show_fig:
        plt.show()
