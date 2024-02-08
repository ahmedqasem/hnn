from metrics import find_metrics
import os
import numpy as np
import pandas as pd
from plotting import plot_sample
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from skimage.filters import threshold_otsu

from keras.optimizers import Adam
from unet_network.unet import build_unet


def get_image_lists(root_test_images_path):
    """ returns lists of image and labels file names """
    test_images = os.listdir(os.path.join(root_test_images_path, "images"))
    test_labels = os.listdir(os.path.join(root_test_images_path, "labels"))
    print(f'found {len(test_images)} testing images')
    print(f'found {len(test_labels)} testing labels')

    return test_images, test_labels


def get_images(root_test_images_path, image_name, model, size=128):
    """ returns final image, label, raw prediction, otsu_thresholded image """
    # name = image_names[n]
    # i = image_names.index(name)

    # load image
    img = load_img(os.path.join(root_test_images_path + '/images', image_name), target_size=(size, size),
                   color_mode='grayscale')
    img = img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)

    # load label
    label = load_img(os.path.join(root_test_images_path + '/labels', image_name), target_size=(size, size),
                     color_mode='grayscale')
    label = img_to_array(label) / 255

    # get prediction
    prediction = model.predict(img)

    # thresholded prediction
    thresh = threshold_otsu(prediction)
    prediction_otsu = prediction > thresh

    return img, label, prediction, prediction_otsu


def batch_process(test_images_path, test_images, model, save_to=None):
    """ find the metrics for the whole testing set and return a metrics df
    args:
    return: metrics dataframe

    """
    NAMES = []
    OBJ_LABEL = []
    BG_LABEL = []
    TP = []
    TN = []
    FP = []
    FN = []
    TPR = []
    FPR = []
    FSCORE = []
    IOU = []
    ACCU = []
    SPEC = []
    SEN = []

    for i in range(len(test_images)):
        n = i
        name = test_images[n]
        NAMES.append(name)

        img, label, prediction, prediction_otsu = get_images(test_images_path, name, model)
        # plot_sample(img, label, prediction, prediction_otsu)

        obj_pxl_truth, bg_pxl_truth, tp, tn, fp, fn, true_positive_rate, false_positive_rate, fscore, iou, accu, spec, sens = find_metrics(
            label, prediction_otsu, img_name=name)

        OBJ_LABEL.append(obj_pxl_truth)
        BG_LABEL.append(bg_pxl_truth)
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        TPR.append(true_positive_rate)
        FPR.append(false_positive_rate)
        FSCORE.append(fscore)
        IOU.append(iou)
        ACCU.append(accu)
        SPEC.append(spec)
        SEN.append(sens)

        # save image
        if save_to:
            plt.imsave(os.path.join(save_to, name), prediction_otsu.squeeze(), cmap='gray')

    data_dict = dict()
    data_dict['Image Name'] = NAMES
    data_dict['Object Pixels'] = OBJ_LABEL
    data_dict['Background Pixels'] = BG_LABEL
    data_dict['True Positives'] = TP
    data_dict['False Positives'] = FP
    data_dict['True Negatives'] = TN
    data_dict['False Negatives'] = FN
    data_dict['True Positive Rate'] = TPR
    data_dict['False Positive Rate'] = FPR
    data_dict['F-Score'] = FSCORE
    data_dict['IoU'] = IOU
    data_dict['Accuracy'] = ACCU
    data_dict['Sensitivity'] = SEN
    data_dict['Specificity'] = SPEC

    df = pd.DataFrame(data_dict)
    return df


def preview_one_image(test_images_path, name, model):
    img, label, prediction, prediction_otsu = get_images(test_images_path, name, model)

    find_metrics(label, prediction_otsu, img_name=name)
    plot_sample(img, label, prediction, prediction_otsu)


def main(batch=False):
    # set parameters
    trained_models_folder = '../trained_models/unet/'
    model_name = '5_50_300_8_20211111-175856.hdf5' #'5_50_300_8__weights.100.hdf5'
    test_images_path = '../data/ct/test/'  
    save_folder = '../data/ct/test/results_5_50_300_8_20211111-175856/' 

    # # load model
    # model = load_model(os.path.join(trained_models_folder, model_name))
    # model.summary()

    # load model
    model = build_unet(input_shape=(128, 128, 1))
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(os.path.join(trained_models_folder, model_name))

    test_images, test_labels = get_image_lists(test_images_path)

    # i = test_images.index('HN-HGJ-016-1915.png')
    name = 'HN-HGJ-042-33.png'

    if not batch:
        preview_one_image(test_images_path, name, model)

    else:
        # if we wanna batch process the data
        data = batch_process(test_images_path, test_images, model, save_folder)

        print(data.head())
        print(data.shape, 'done!')

        data.to_csv(save_folder+'test.csv', sep=',', index=False)


if __name__ == "__main__":
    main(batch=False)
