""" functions for visualizing results_5_50_300_8_20211112-183534 """
import matplotlib.pyplot as plt


def plot_sample(image, ground_truth, prediction, otsu_pred, img_cont=True, pred_cont=True):
    """
        a function to plot a sample prediction
        args:
            image: original image
            ground_truth: label
            prediction: predicted mask before thresholding
            otsu_pred: thresholded mask
            img_cont: draw the gt contour on the original image
            pred_cont: draw the gt contour on the prediction and the thresholded prediction
        returns:
            sample prediction plot
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    # original image
    ax[0][0].imshow(image.squeeze(), cmap='gray')
    if img_cont:
        ax[0][0].contour(ground_truth.squeeze())
    ax[0][0].set_title('Original Image')
    # ground truth
    ax[0][1].imshow(ground_truth.squeeze(), cmap='gray')
    ax[0][1].set_title('Ground truth')
    # Raw prediction
    ax[1][0].imshow(prediction.squeeze(), cmap='gray')
    if pred_cont:
        ax[1][0].contour(ground_truth.squeeze())
    ax[1][0].set_title('Raw prediction')
    # Thresholded prediction
    ax[1][1].imshow(otsu_pred.squeeze(), cmap='gray')
    if pred_cont:
        ax[1][1].contour(ground_truth.squeeze())
    ax[1][1].set_title('Thresholded Prediction')

    fig.suptitle('Sample Prediction', fontsize=18)
    plt.tight_layout()
    plt.show()

def display(image, ground_truth,  img_cont=True, pred_cont=True):
    """
        a function to plot a sample prediction
        args:
            image: original image
            ground_truth: label
            img_cont: draw the gt contour on the original image
            pred_cont: draw the gt contour on the prediction and the thresholded prediction
        returns:
            sample prediction plot
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
    # original image
    ax[0].imshow(image.squeeze(), cmap='gray')
    if img_cont:
        ax[0].contour(ground_truth.squeeze())
    ax[0].set_title('Original Image')
    # ground truth
    ax[1].imshow(ground_truth.squeeze(), cmap='gray')
    ax[1].set_title('Ground truth')


    fig.suptitle('Image preview', fontsize=18)
    plt.tight_layout()
    plt.show()