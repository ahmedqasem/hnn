""" randomly copy x number of images and their labels from one location to another """
import random
import os
import shutil


def select_randomly(n):
    for i in range(n):
        # print(i, image_names[random.randint(0, len(image_names))])
        img = image_names[random.randint(0, len(image_names))]

        shutil.copy(os.path.join(test_images_path, img), os.path.join(target_images_path, img))
        shutil.copy(os.path.join(test_labels_path, img), os.path.join(target_labels_path, img))

    print(f'Copied {n} images!')


def rename_multiple(path, add_pre = ''):
    """ rename multiple images in the same directory """
    files = os.listdir(path)
    for file in files:
        nfile = file.split('.')[0]
        new_name = nfile + add_pre + '.png'
        print(new_name)
        os.rename(os.path.join(path, file), os.path.join(path, new_name))
    # pass


def rename_multiple_revert(path):
    """ rename multiple images in the same directory """
    files = os.listdir(path)
    for file in files:
        nfile = file.split('_')[0]
        new_name = nfile+'.png'
        print(new_name)
        os.rename(os.path.join(path, file), os.path.join(path, new_name))



test_images_path = './images/'
test_labels_path = './labels/'

#
# image_names = os.listdir(test_images_path)

# select_randomly(2)

# rename_multiple(test_labels_path, '_label')
rename_multiple_revert(test_labels_path)


