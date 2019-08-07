import numpy as np


def _confusion_matrix(prediction, groundtruth, num_classes):

    m = (groundtruth >= 0) & (groundtruth < num_classes)
    confusion = np.bincount(num_classes*groundtruth[m].astype(int) + prediction[m],
                            minlength=num_classes**2).reshape(num_classes, num_classes)

    return confusion


def validate(prediction, groundtrurh, num_classes):
    """
    Function to compute such metrics, as Pixel Accuracy (acc), Mean Pixel Accuracy (mean_acc),
    Mean Intersection over Union (mean_iou).
    :param prediction - your segmentation map for given image.
    NumPy ndarray of shape [BatchSize, nRows, nCols]
    Assumes 1 - for hair, 0 - for background:
    :param groundtrurh - groundtruth segmentation map.
    NumPy ndarray of shape [BatchSize, nRows, nCols]
    Assumes 1 - for hair, 0 - for background:
    :param num_classes - #classes, assumes = 2 for binary classification:
    :return acc, mean_acc, mean_iou:
    """

    confusion = np.zeros((num_classes, num_classes))
    for pred, gt in zip(prediction, groundtrurh):
        confusion += _confusion_matrix(pred, gt, num_classes)

    acc = np.diag(confusion).sum() / confusion.sum()

    mean_acc= np.diag(confusion) / confusion.sum(axis=1)
    mean_acc = np.nanmean(mean_acc)

    iou = np.diag(confusion) / (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    mean_iou = np.nanmean(iou)

    return acc, mean_acc, mean_iou
