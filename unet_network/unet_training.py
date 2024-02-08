import datetime
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from image_gen import trainGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from unet_network.unet import unet, build_unet

# root dataset folder
BASE_DIR = "./data/ct/"
predict_outcome_folder_name = 'label_predict/'
trained_model_location = '../trained_models/unet/'

# set parameters
batches = 5
steps = 50
epoc = 300
im_height = 128
im_width = 128
no_filters = 8

# log dir for tensorboard
training_timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_name = f'{str(batches)}_{str(steps)}_{str(epoc)}_{str(no_filters)}_'
logdir = "../logs/fit_" + log_name + training_timestamp

# save model as
# trained_model_path = trained_model_location + log_name + training_timestamp + '_' + 'weights.{epoch:02d}.hdf5'
trained_model_path = trained_model_location + log_name + training_timestamp + '.hdf5'
# trained_model_path = trained_model_location + log_name + '_' + '.hdf5'

# augmentation parameters
data_gen_args = dict(rotation_range=0.5,#0.2,
                     width_shift_range=0.5,#0.05,
                     height_shift_range=0.5,#0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

# 1. image (and labels) data generator
myGene = trainGenerator(batches, BASE_DIR + 'train/', 'images', 'labels',
                        data_gen_args,
                        save_to_dir=None,
                        target_size=(im_height, im_width))
# 2. validation data generator
myValid = trainGenerator(1,
                         BASE_DIR + 'valid/', 'images', 'labels',
                         data_gen_args, save_to_dir=None,
                         target_size=(im_height, im_width))

x, y = next(myGene)
for i in range(0, 1):
    image = x[i]
    mask = y[i]
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0])
    plt.show()

# model
# model = unet_network(pretrained_weights=None, input_size=(im_height, im_width, 1), n=no_filters)
model = build_unet(input_shape=(im_height, im_width, 1))
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ModelCheckpoint(trained_model_path, monitor='val_loss', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001),
    EarlyStopping(monitor='val_loss', patience=20, min_delta=0),
    TensorBoard(log_dir=logdir, update_freq='batch')#histogram_freq=0)
]

# start training
history = model.fit_generator(myGene,
                              steps_per_epoch=steps,
                              epochs=epoc,
                              callbacks=callbacks,
                              validation_data=myValid,
                              validation_steps=34,
                              verbose=1)

print('finished training ! ')
