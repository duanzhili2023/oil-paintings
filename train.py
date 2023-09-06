from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
from keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(directory='./train_cezanne/',
                                  target_size=(299,299),
                                  batch_size=16)
val_generator = val_datagen.flow_from_directory(directory='./valid_cezanne/',
                                target_size=(299,299),
                                batch_size=32)


base_model = InceptionV3(weights='imagenet',include_top=False)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
predictions = Dense(2,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
# plot_model(model,'tlmodel.png')
def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

setup_to_transfer_learning(model,base_model)
early_stopping= EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

history_tl = model.fit_generator(generator=train_generator,
                    #steps_per_epoch=100,#800
                    epochs=20,#2
                    validation_data=val_generator,
                    validation_steps=100,#12
                    class_weight='auto',callbacks=[early_stopping]
                    )
model.save('./cezanne_iv3_tl.h5')


'''
setup_to_fine_tune(model,base_model)

history_ft = model.fit_generator(generator=train_generator,
                                 #steps_per_epoch=100,
                                 epochs=20,
                                 validation_data=val_generator,
                                 validation_steps=100,
                                 class_weight='auto',callbacks=[early_stopping])
model.save('./cezanne_iv3_ft.h5')
'''

