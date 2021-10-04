import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import os

# load train.csv
train = pd.read_csv('data/train.csv')

# read all the training images, store them in a list, and finally convert that list into a numpy array.
path = 'data/train'
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img(path+'/'+train['id'][i].astype('str')+'.png',
                        target_size = (256, 256, 1), grayscale= True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X_train = np.array(train_image)

# put train label in y_train
y_train= train['label'].values

#import the test file:
test = pd.read_csv('data/test.csv')
test_path = 'data/test'

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# read and store all the test images:
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img(test_path+'/'+test['id'][i].astype('str')+'.png', target_size=(256, 256, 1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
X_test = np.array(test_image)

# put test label in y_test
y_test = test['label'].values

# create a CNN architecture with 2 convolutional layers, one dense hidden layer and an output layer.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(256, 256 ,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim= 128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',  # half of the default lr
              metrics=['accuracy'])

# train the model on the training set images and validate it using the validation set.
model.fit(X_train,y_train , epochs=10, validation_data=(X_test, y_test))


#import predict file
#predict_file = pd.read_csv('data/test_invoice.csv')
test_path = 'columbia'

# read and store all the test images:
listOfFile = os.listdir(test_path)
test_image = []
filename = []
# Iterate over all the entries
for entry in listOfFile:
    # Create full path
    fullPath = os.path.join(test_path, entry)
    # If entry is a directory then get the list of files in this directory 
    #if os.path.isdir(fullPath):
        #test_image = test_image + getListOfFiles(fullPath)
    #else:
    if(fullPath.endswith('.png')):
        filename.append(os.path.splitext(fullPath)[0])
        img = image.load_img(fullPath ,target_size=(256, 256,1), grayscale=True)
        img = image.img_to_array(img)
        img = img/255
        test_image.append(img)
test_img = np.array(test_image)

pred = model.predict(test_img)
cl = np.round(pred)
filenames = filename

result=pd.DataFrame({"files":filenames,'pr':pred[:,0],"Predict label":cl[:,0]})
print(result)




























