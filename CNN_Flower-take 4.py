
# coding: utf-8

# In[15]:


import numpy as np
from keras import models
from keras import layers
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from PIL import Image
from PIL import Image, ImageOps
import os
import split_folders


# In[16]:


np.random.seed(777)


# In[17]:


#Creating a data generator to prepare the data


# In[18]:


b_size = 50

train_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/Keras_out/out/train/", 
                                                    target_size=(150,150),
                                                    color_mode= "rgb",
                                                    batch_size = b_size, 
                                                    shuffle = True,
                                                    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/Keras_out/out/val/", 
                                                              target_size=(150,150),
                                                              color_mode= "rgb",
                                                              batch_size = b_size, 
                                                              shuffle = True,
                                                              class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/Keras_out/out/test/", 
                                                  target_size=(150,150),
                                                  color_mode= "rgb",
                                                  batch_size = b_size, 
                                                  shuffle = True,
                                                  class_mode = 'categorical')


# In[19]:


for data_batch, label_batch in train_generator:
    print('train batch shape ', data_batch.shape )
    print('train_label_batch: ', label_batch.shape)
    break
for data_batch, label_batch in validation_generator:
    print('validation batch shape ', data_batch.shape )
    print('validation_label_batch: ', label_batch.shape)
    break
for data_batch, label_batch in test_generator:
    print('test batch shape ', data_batch.shape )
    print('test_label_batch: ', label_batch.shape)
    break


# In[20]:


# Building the model


# In[21]:


model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

#model.add(layers.Conv2D(256,(3,3),activation = 'relu'))
#model.add(layers.MaxPooling2D((2,2)))
            
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(5, activation = 'softmax'))


# In[22]:


model.summary()


# In[23]:


model.compile(optimizer = 'RMSprop' , loss = 'categorical_crossentropy'  , metrics = ['acc'])


# In[25]:


history = model.fit_generator(
            train_generator,
            steps_per_epoch = np.ceil(2100/b_size),
            epochs = 30,
            validation_data = validation_generator,
            validation_steps = np.ceil(700/b_size))


# In[26]:


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[27]:


# accuracy and confusion matrix for the test set


# In[41]:


from sklearn.metrics import confusion_matrix, classification_report

y_p = model.predict_generator(test_generator, np.ceil(700/b_size))
y_pred = np.argmax(y_p , axis = 1)
#print(y_pred)
cm = confusion_matrix(test_generator.classes, y_pred)
print(cm)

score, accuracy = model.evaluate_generator(test_generator, np.ceil(700/b_size))
print('Test score: ',score)
print('Test accuracy: ', accuracy)

target_names = ['Rose', 'Daisy','Dandelion','Sunflower', 'Tulip']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))


# In[ ]:


# implementing the image augmentation and dropout to reduce overfitting


# In[42]:


train_datagen2 = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 40,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                fill_mode = 'nearest')
validation_datagen2 = ImageDataGenerator(rescale = 1./255)
test_datagen2 = ImageDataGenerator(rescale = 1./255)

train_generator2 = train_datagen2.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/Keras_out/out/train/", 
                                                    target_size=(150,150),
                                                    color_mode= "rgb",
                                                    batch_size = b_size, 
                                                    shuffle = True,
                                                    class_mode = 'categorical')

validation_generator2 = validation_datagen2.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/Keras_out/out/val/", 
                                                              target_size=(150,150),
                                                              color_mode= "rgb",
                                                              batch_size = b_size, 
                                                              shuffle = True,
                                                              class_mode = 'categorical')

test_generator2 = test_datagen2.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/Keras_out/out/test/", 
                                                  target_size=(150,150),
                                                  color_mode= "rgb",
                                                  batch_size = b_size, 
                                                  shuffle = True,
                                                  class_mode = 'categorical')


# In[44]:


network = models.Sequential()

network.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150,3)))
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Conv2D(64,(3,3), activation = 'relu'))
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Conv2D(128,(3,3), activation = 'relu'))
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Conv2D(128,(3,3), activation = 'relu'))
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Flatten())

network.add(layers.Dropout(0.5))

network.add(layers.Dense(512, activation = 'relu'))

network.add(layers.Dense(5, activation = 'softmax'))


# In[46]:


network.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = ['acc'])


# In[47]:


history = network.fit_generator(
            train_generator2,
            steps_per_epoch = np.ceil(2100/b_size),
            epochs = 30,
            validation_data = validation_generator2,
            validation_steps = np.ceil(700/b_size))


# In[50]:



acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label ='Train acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Taining and Validation Accuracy')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label = 'Train loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title("Training and Validation loss")
plt.legend()
plt.show()


# In[ ]:


# measuring the accuracy and confusion matrix for the test set


# In[53]:


y_p = network.predict_generator(test_generator2, np.ceil(700/b_size))
y_pred = np.argmax(y_p, axis = 1)

confusion_mat = confusion_matrix(test_generator2.classes, y_pred)
print(confusion_mat)

score, acc = network.evaluate_generator(test_generator2, np.ceil(700/b_size))
print("Test set score after augmentation: ", score)
print("Test set accuracy after augmentation: ", acc)

target_names = ['Rose', 'Daisy','Dandelion','Sunflower', 'Tulip']
print(classification_report(test_generator2.classes, y_pred, target_names=target_names))


# In[ ]:


# Optional : using a pre trained model VGG16


# In[55]:


from keras.applications import VGG16

conv_base = VGG16(weights = 'imagenet',
                 include_top = False,
                 input_shape = (150,150,3))


# In[57]:


conv_base.summary()


# In[ ]:




