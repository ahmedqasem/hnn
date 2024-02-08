import numpy as np


def tpr(true_positive, total_obj_pxl_in_truth):
    if total_obj_pxl_in_truth == 0:
        return 0
    return true_positive / total_obj_pxl_in_truth


def fpr(false_positive, total_bg_pxl_in_truth):
    return false_positive / total_bg_pxl_in_truth


def f_score(true_positive, false_positive, false_negative):
    if true_positive == 0:
        return 0
    else:
        pres = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        fscore = (2 * pres * recall) / (pres + recall)
        return fscore


def iou_score(true_label, predict_label):
    # load truth label
    true_label = true_label.flatten()

    # load predictiob
    predict_label = predict_label.flatten()

    intersection = np.logical_and(true_label, predict_label)
    union = np.logical_or(true_label, predict_label)
    return np.sum(intersection) / np.sum(union)


def sensitivity(true_positive, false_negative):
    if (true_positive + false_negative) == 0:
        return 0
    return true_positive / (true_positive + false_negative)


def specificity(false_positive, true_negative):
    return true_negative / (true_negative + false_positive)


def accuracy(true_positive, false_positive, true_negative, false_negative):
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


def find_metrics(true_label, predict_label,img_name='', show_results=True):
    # load truth label
    true_label = np.squeeze(true_label)

    # load predicted label
    predict_label = np.squeeze(predict_label)

    # find number of bg and label pixels in ground truth label
    flat_truth = true_label.flatten()
    positive_in_truth = flat_truth > 0
    total_bg_pxl_truth = list(positive_in_truth).count(False)
    total_obj_pxl_truth = list(positive_in_truth).count(True)
    # print('number of label pixels ', total_obj_pxl_truth)
    # print('number of background pixels ', total_bg_pxl_truth)

    # convert the predicted label to binary values (0 - 1)
    label_predict = np.where(predict_label != 0, 1, 0)

    # compare ground truth with prediction
    flat_predict = label_predict.flatten()

    tn = 0
    tp = 0
    fn = 0
    fp = 0

    for pixel in range(len(flat_truth)):
        if flat_predict[pixel] == 0 and flat_truth[pixel] == 0:
            tn += 1
        elif flat_predict[pixel] != 0 and flat_truth[pixel] != 0:
            tp += 1
        elif flat_truth[pixel] == 0 and flat_predict[pixel] != 0:
            fp += 1
        else:
            fn += 1

    assert total_obj_pxl_truth == tp + fn
    assert total_bg_pxl_truth == tn + fp
    assert (total_bg_pxl_truth+total_obj_pxl_truth) == (tp+tn+fp+fn)

    true_positive_rate = tpr(tp, total_obj_pxl_truth)
    false_positive_rate = fpr(fp, total_bg_pxl_truth)
    iou = iou_score(true_label, predict_label)
    accu = accuracy(tp, fp, tn, fn)
    spec = specificity(fp, tn)
    sens = sensitivity(tp, fn)
    fscore = f_score(tp, fp,fn)

    if show_results:
        print(f'\n\nShowing results for image {img_name}:')
        print(f'\tObject Pixels in label: {total_obj_pxl_truth}')
        print(f'\tBackground Pixels in label: {total_bg_pxl_truth}')
        print(f'\tTrue Positives in prediction: {tp}')
        print(f'\tTrue Negatives in prediction: {tn}')
        print(f'\tFalse Positives in prediction: {fp}')
        print(f'\tFalse Negatives in prediction: {fn}\n')
        print(f'\tTPR: {true_positive_rate}')
        print(f'\tFPR: {false_positive_rate}')
        print(f'\tF-score: {fscore}')
        print(f'\tIoU: {iou}')
        print(f'\tAccuracy: {accu}')
        print(f'\tSpecificity: {spec}')
        print(f'\tSensitivity: {sens}')

    return total_obj_pxl_truth, total_bg_pxl_truth, tp, tn, fp, fn, true_positive_rate, false_positive_rate, fscore, iou, accu, spec, sens