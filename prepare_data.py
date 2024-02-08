""" script for extracting and pre-processing the DICOM images """

from data_prepare_functions import *
import os
import glob
import pydicom as dicom
from dicom_contour.contour import *
import pandas as pd


# select the source folder for each image
hospital = 'CHUM'
data_root = f'./data/Head-Neck-PET-CT/{hospital}/'
patients = os.listdir(data_root)
gtv_index_df_path = './data/GTVcontours.xls'
data = pd.read_excel(gtv_index_df_path, sheet_name=hospital)

for n, patient in enumerate(patients):
    if n > 0:
        break

    # find location of CT PET and Contour file
    ct_location, pet_location, ct_contour_location, pet_contour_location = findfiles_CHUM(os.path.join(data_root, patient))
    print(f'ct: {ct_location}')
    print(f'pet: {pet_location}')
    print(f'cont: {ct_contour_location}')

    # find the GTV index
    gtv_index = data[data['Patient'] == patient]['Name GTV Primary']

    ct_images, contours, pet_images = get_datas(ct_location, pet_location, ct_contour_location, gtv_index)







