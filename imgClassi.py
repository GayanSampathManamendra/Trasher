import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

#First, let us do some necessary imports.
# The keras library helps us build our convolutional neural network.
import keras

from keras.models import Sequential

#Dense used to predict the labels
#Dropout layer reduces overfitting
#Flatten Layer expands a three-dimensional vector into a one-dimensional vector

from keras.layers import Dense, Dropout, Flatten,Activation,Conv2D, MaxPooling2D

import pickle

DataDir = "E:/dsets/trashnet/data/dataset-resized"
Garbages = ["glass","cardboard","metal","paper","plastic","trash"]

training_data=[]
IMG_SIZE = 224

def create_training_data():
    for garbage in Garbages:
        path=os.path.join(DataDir,garbage) # path to garbages
        class_num = Garbages.index(garbage)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path,img))
                new_arr = cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
                # plt.imshow(new_arr , cmap="gray")
                # plt.show()
                training_data.append([new_arr,class_num])
            except Exception as e:
                pass


create_training_data()
print(len(training_data))

random.shuffle(training_data)

# for sample in training_data[:10]:
#     print(sample[1])

X=[]
y=[]

for features,label in training_data:
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)


pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in =open("X.pickle","rb")
X=pickle.load(pickle_in)
#print(X[1])


X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))

X=X/255.0

#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(17))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

# Train
model.fit(X, y, batch_size=64, epochs=1, verbose=1, validation_split=0.2, shuffle=True)
