from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
from keras.callbacks import TensorBoard
import keras
import numpy
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_auc_score
import datetime
model = keras.models.load_model("cezanne_iv3_tl.h5")
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(directory='./test_cezanne/',
                                  target_size=(299,299),
                                  batch_size=1,shuffle = False)
                                  
start_time = datetime.datetime.now()
predict_y = model.predict_generator(test_generator,steps=len(test_generator))
end_time = datetime.datetime.now()

files = test_generator.filenames
f = open('files_test_cezanne.txt','w')
f.write('\n'.join(files))


f.close()

numpy.savetxt("result_test_cezanne.txt",predict_y)

test_generator = test_datagen.flow_from_directory(directory='./valid_cezanne/',
                                  target_size=(299,299),
                                  batch_size=1,shuffle = False)
                                  
start_time = datetime.datetime.now()
predict_y = model.predict_generator(test_generator,steps=len(test_generator))
end_time = datetime.datetime.now()

files = test_generator.filenames
f = open('files_valid_cezanne.txt','w')
f.write('\n'.join(files))


f.close()

numpy.savetxt("result_valid_cezanne.txt",predict_y)



test_generator = test_datagen.flow_from_directory(directory='./extra_validation_cezanne/',
                                  target_size=(299,299),
                                  batch_size=1,shuffle = False)
                                  
start_time = datetime.datetime.now()
predict_y = model.predict_generator(test_generator,steps=len(test_generator))
end_time = datetime.datetime.now()

files = test_generator.filenames
f = open('files_extra_validation_cezanne.txt','w')
f.write('\n'.join(files))


f.close()

numpy.savetxt("result_extra_validation_cezanne.txt",predict_y)



