import numpy as np
np.random.seed(12321)  # for reproducibility
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.pooling import MaxPooling2D 
import h5py
from keras import backend as K
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

nb_classes = 10
num_classes = 10
img_rows, img_cols = 42, 28
nb_epoch = 3
batch_size = 64
K.set_image_dim_ordering('th')
# input image dimensions




def main():
	X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)
	#Investigate the test set, training set, and the labels to get intuition about the representation of the data

	#=================== Model ======================#
	# TO DO: Define a model with inputs an image of size 42*28 and outputs the two digits. The model variable should be called model and should be 
	# based on a Convolutional Neural Network. Consult  https://keras.io/getting-started/functional-api-guide/ for the relevant API.

	# A setting of parameters used by us was the following:
	# A 2D Conv layer with 8 filters and relu activation
	# A maxpooling layer with (2,2) size filter and (2,2) stried
	# A 2D conv layer with 16 filters and relu activation
	# A maxpooling layer with (2,2) size filter and default, i.e. (1,1), stride
	# Flattering then a Dense layer, and then dropout with rate 0.5
	# Two outputs


	inputs = Input(shape=(1,img_rows, img_cols))
	x=Conv2D(8,(3,3), padding='same', activation = 'relu')(inputs)
	x=MaxPooling2D((2,2), strides=(2,2), padding='same')(x)
	x=Conv2D(16,(3,3), padding='same', activation = 'relu')(x)
	x=MaxPooling2D((2,2), strides=(1,1), padding='same')(x)
	x=Flatten()(x)
	x= Dense(64, activation='relu')(x)
	x=Dropout(0.5)(x)
	y=(Dense(10,activation='softmax')(x), Dense(10, activation='softmax')(x))
	model = Model(inputs=[inputs], outputs=[y[0],y[1]])

	model.compile(loss='categorical_crossentropy',optimizer='sgd',  metrics=['accuracy'], loss_weights=[0.5, 0.5])
	#==================== Fetch Data and Fit ===================#
	model.fit(X_train, [y_train[0], y_train[1]], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
	objective_score = model.evaluate(X_test, y_test, batch_size=64)
	print('Evaluation on test set:', dict(zip(model.metrics_names, objective_score)))
	
	#Uncomment the following line if you would like to save your trained model
	#model.save('./current_model_conv.h5')
	if K.backend()== 'tensorflow':
		K.clear_session()

if __name__ == '__main__':
	main()
