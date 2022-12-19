# %%
import cv2
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from sklearn.preprocessing import OneHotEncoder
import pandas as pad
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras

# %% [markdown]
# ## 2. Loading the Data

# %%
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Train data shape: ", x_train.shape, y_train.shape)
print("Test data shape: ", x_test.shape, y_test.shape)

# %% [markdown]
# ### 3. Converting 28\*28 images to 32\*32

# %%
# defining the rbf function


def RBF(x, c, s):
    return np.exp(((x-c)**2)/(2*s**2))

# defining a function to transform the images


def transform(image):
    image = np.pad(image, (2, 2))
    c = np.mean(image)
    s = np.std(image)
    return RBF(image, c, s).flatten()
# flatten reduces each image into a 1-D array by storing it in row-major format


# %%
x_train_tf = []
for image in x_train:
    x_train_tf.append(transform(image))
x_train_tf = np.array(x_train_tf)
print("Shape of x_train after transforming: ", x_train_tf.shape)
x_test_tf = []
for image in x_test:
    x_test_tf.append(transform(image))
x_test_tf = np.array(x_test_tf)
print("Shape of x_test after transforming: ", x_test_tf.shape)

# %%
# one hot encoding the labels to be predicted
# this converts each label into a vector with 10 fields, one corresponding to each label
encoder = OneHotEncoder()

y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()

print("Shape of y_train after ohe: ", y_train.shape)
print("Shape of y_test after ohe: ", y_test.shape)

# %% [markdown]
# # Running the network with different settings

# %%

# %%
adam = Adam(learning_rate=0.001)

# %% [markdown]
# ## 4. Number of Hidden Layers

# %% [markdown]
# ### 4.1 Activation function: Sigmoid; Hidden neurons: [16]

# %%
# using the functional API approach
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
dense_layer = Dense(16, activation='sigmoid')(input_layer)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer='glorot_normal')(dense_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model1_1
model_path = "model1_1/"

# %%
# Setting the filepath for the ModelCheckpoint callback to save the files
filepath = model_path+"{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_loss',  verbose=1, save_best_only=True, mode='auto')

# We will use the EarlyStopping callback to prevent the model from training after the optimal condition has been met
earlystop = EarlyStopping(
    monitor='val_loss', patience=7, verbose=1, mode='min')

# Passing validation data into the custom Metrics callback that we have created
validation_data = (x_test_tf, y_test)

model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
!rm - rf ./logs1_1/fit
tf.keras.backend.clear_session()

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs1_1/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop, checkpoint]

model.fit(x_train_tf, y_train, epochs=50,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%load_ext tensorboard
%tensorboard - -logdir logs1_1

# %% [markdown]
# ### 4.2 Activation function: Sigmoid; Hidden neurons: [32, 16]

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
dense_layer_1 = Dense(32, activation='sigmoid')(input_layer)
dense_layer_2 = Dense(16, activation='sigmoid')(dense_layer_1)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer='glorot_normal')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model1_2
model_path = "model1_2/"

# %%
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
!rm - rf ./logs1_2/fit
tf.keras.backend.clear_session()

# %%
# Creating logs directory to store information about the fits
log_dir = "logs1_2/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=50,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%load_ext tensorboard
%tensorboard - -logdir logs1_2

# %% [markdown]
# ### 4.3 Activation Function: Sigmoid; Hidden Neurons: [64, 32, 16]

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
dense_layer_1 = Dense(64, activation='sigmoid')(input_layer)
dense_layer_2 = Dense(32, activation='sigmoid')(dense_layer_1)
dense_layer_3 = Dense(16, activation='sigmoid')(dense_layer_2)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer='glorot_normal')(dense_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model1_3
model_path = "model1_3/"

# %%
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
!rm - rf ./logs1_3/fit
tf.keras.backend.clear_session()

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs1_3/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%tensorboard - -logdir logs1_3 - -port 6008

# %%
!kill 2336

# %% [markdown]
# ## 5. Varying the Activation Function

# %% [markdown]
# ### 5.1 Activation function: Sigmoid
# This model is same as the model in 4.3

# %% [markdown]
# ### 5.2 Activation function: tanh

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
dense_layer_1 = Dense(64, activation='tanh',
                      kernel_initializer='glorot_normal')(input_layer)
dense_layer_2 = Dense(32, activation='tanh',
                      kernel_initializer='glorot_normal')(dense_layer_1)
dense_layer_3 = Dense(16, activation='tanh',
                      kernel_initializer='glorot_normal')(dense_layer_2)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer='glorot_normal')(dense_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model2_2
model_path = "model2_2/"

# %%
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
!rm - rf ./logs2_2/fit
tf.keras.backend.clear_session()

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs2_2/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=50,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%load_ext tensorboard
%tensorboard - -logdir logs2_2

# %% [markdown]
# ### 5.3 Activation function: ReLU

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
dense_layer_1 = Dense(64, activation='relu',
                      kernel_initializer='he_normal')(input_layer)
dense_layer_2 = Dense(32, activation='relu',
                      kernel_initializer='he_normal')(dense_layer_1)
bn = BatchNormalization()(dense_layer_2)
dense_layer_3 = Dense(16, activation='relu',
                      kernel_initializer='he_normal')(bn)
dense_layer_3 = Dropout(0.5)(dense_layer_3)
output_layer = Dense(10, activation='softmax')(dense_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model2_3
model_path = "model2_3/"

# %%
# We will use the EarlyStopping callback to prevent the model from training after the optimal condition has been met
earlystop = EarlyStopping(
    monitor='val_loss', patience=10, verbose=1, mode='min')

# Passing validation data into the custom Metrics callback that we have created
validation_data = (x_test_tf, y_test)

model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
!rm - rf ./logs2_3/fit
tf.keras.backend.clear_session()

# %%
# Creating logs directory to store information about the fits
log_dir = "logs2_3/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%load_ext tensorboard
%tensorboard - -logdir logs2_3

# %% [markdown]
# ## 6. Varying the Dropout Rate

# %% [markdown]
#
# ### 6.1 Dropout = 0.9

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
dense_layer_1 = Dense(64, activation='relu',
                      kernel_initializer='he_normal')(input_layer)
dense_layer_2 = Dense(32, activation='relu',
                      kernel_initializer='he_normal')(dense_layer_1)
bn = BatchNormalization()(dense_layer_2)
dense_layer_3 = Dense(16, activation='relu',
                      kernel_initializer='he_normal')(bn)
dense_layer_3 = Dropout(0.9)(dense_layer_3)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer="glorot_normal")(dense_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model3_1
model_path = "model3_1/"

# %%
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
!rm - rf ./logs3_1/fit
tf.keras.backend.clear_session()

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs3_1/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%load_ext tensorboard
%tensorboard - -logdir logs3_1

# %% [markdown]
# ### 6.2 Dropout = 0.75

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
d = Dense(64, activation='relu', kernel_initializer='he_normal')(input_layer)
d = Dense(32, activation='relu', kernel_initializer='he_normal')(d)
bn = BatchNormalization()(d)
d = Dense(16, activation='relu', kernel_initializer='he_normal')(bn)
bn = BatchNormalization()(d)
d = Dropout(0.75)(bn)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer="glorot_normal")(d)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model3_2
model_path = "model3_2/"

# %%
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
!rm - rf ./logs3_2/fit
tf.keras.backend.clear_session()

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs3_2/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%tensorboard - -logdir logs3_2/

# %% [markdown]
# ### 6.3 Dropout = 0.5
# This is same as the model in part 5.3

# %% [markdown]
# ### 6.4 Dropout = 0.25

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
d = Dense(64, activation='relu', kernel_initializer='he_normal')(input_layer)
bn = BatchNormalization()(d)
d = Dense(32, activation='relu', kernel_initializer='he_normal')(bn)
bn = BatchNormalization()(d)
d = Dense(16, activation='relu', kernel_initializer='he_normal')(bn)
bn = BatchNormalization()(d)
d = Dropout(0.25)(bn)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer="glorot_normal")(d)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model3_4
model_path = "model3_4/"
!rm - rf ./logs3_4/fit
tf.keras.backend.clear_session()
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
# Creating logs directory to store information about the fits
log_dir = "logs3_4/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%load_ext tensorboard
%tensorboard - -logdir logs3_4

# %% [markdown]
# ### 6.5 Dropout = 0.1

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
d = Dense(64, activation='relu', kernel_initializer='he_normal')(input_layer)
bn = BatchNormalization()(d)
d = Dense(32, activation='relu', kernel_initializer='he_normal')(bn)
bn = BatchNormalization()(d)
d = Dense(16, activation='relu', kernel_initializer='he_normal')(bn)
bn = BatchNormalization()(d)
d = Dropout(0.1)(bn)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer="glorot_normal")(d)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model3_5
model_path = "model3_5/"
!rm - rf ./logs3_5/fit
tf.keras.backend.clear_session()
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs3_5/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%tensorboard - -logdir logs3_5

# %% [markdown]
# ## 7. Varying the Learning rate

# %% [markdown]
# ### 7.1 Learning Rate = 0.01

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
d = Dense(64, activation='sigmoid')(input_layer)
d = Dense(32, activation='sigmoid')(d)
d = Dense(16, activation='sigmoid')(d)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer='glorot_normal')(d)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model4_1
model_path = "model4_1/"
!rm - rf ./logs4_1/fit

# %%
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs4_1/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%tensorboard - -logdir logs4_1/

# %% [markdown]
# ### 7.2 Learning rate = 0.001
# We have already used a learning rate of 0.001 with all the previous models

# %% [markdown]
# ### 7.3Learning Rate = 0.005

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
d = Dense(64, activation='sigmoid')(input_layer)
d = Dense(32, activation='sigmoid')(d)
d = Dense(16, activation='sigmoid')(d)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer='glorot_normal')(d)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model4_3
model_path = "model4_3/"
!rm - rf ./logs4_3/fit

# %%
model.compile(optimizer=Adam(learning_rate=0.005),
              loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs4_3/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%tensorboard - -logdir logs4_3

# %%


# %% [markdown]
# ### 7.4 Learning Rate = 0.0001

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
d = Dense(64, activation='sigmoid')(input_layer)
d = Dense(32, activation='sigmoid')(d)
d = Dense(16, activation='sigmoid')(d)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer='glorot_normal')(d)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model4_4
model_path = "model4_4/"
!rm - rf ./logs4_4/fit

# %%
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs4_4/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%tensorboard - -logdir logs4_4

# %% [markdown]
# ### 7.5 Learning Rate = 0.0005

# %%
tf.keras.backend.clear_session()

input_layer = Input(shape=(1024, ))
d = Dense(64, activation='sigmoid')(input_layer)
d = Dense(32, activation='sigmoid')(d)
d = Dense(16, activation='sigmoid')(d)
output_layer = Dense(10, activation='softmax',
                     kernel_initializer='glorot_normal')(d)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# %%
!mkdir model4_5
model_path = "model4_5/"
!rm - rf ./logs4_5/fit

# %%
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# Creating logs direcctory to store information about the fits
log_dir = "logs4_5/fit/" + datetime.now().strftime("%Y%m%d - %H%M%S")

tf.keras.utils.plot_model(model, to_file=model_path +
                          'model.png', show_shapes=True)

tbCallBack = TensorBoard(log_dir=log_dir, write_graph=True,
                         write_grads=True, write_images=True)

allCs = [tbCallBack, earlystop]

model.fit(x_train_tf, y_train, epochs=100,
          validation_data=validation_data, batch_size=256, callbacks=allCs)

# %%
%tensorboard - -logdir logs4_5/

# %% [markdown]
# ## 8. Classifying digits written in own handwriting:

# %%
for i in range(1, 6):
    img = cv2.imread(str(i)+'.jpg')
    img = img[:, :, 2]
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = 255-img
    imgplot = plt.imshow(img, cmap="gray")
    img = transform(img)
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    pred = model.predict(img)
    print(np.argmax(pred))
    plt.show()

# %%
