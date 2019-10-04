from imutils import paths
import os
import numpy as np
import librosa
from tqdm import tqdm
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()


files=list(paths.list_files('recordings'))

data=[]
for i in tqdm(files):
    y,sr=librosa.load(i,sr=8000,mono=True)
    mfcc=librosa.feature.mfcc(y,sr=8000, n_mfcc=40)
    if mfcc.shape[1] > 40:
        mfcc = mfcc[:, 0:40]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 40 - mfcc.shape[1])),
                               mode='constant', constant_values=0)
    data.append(mfcc)
data=np.array(data)
data = data.reshape((data.shape[0], 40, 40, 1))

labels=[]
for i in files:
    labels.append(i.split(os.path.sep)[1])
labels=np.array(labels)

labels=to_categorical(labels)

X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=(40,40,1),padding='same'))
model.add(Conv2D(48, (2, 2), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (2, 2), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.40))
model.add(Dense(10, activation='softmax'))
print(model.summary())
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

history=model.fit(X_train, y_train,
          batch_size=64,
          epochs=50,
          verbose=2,
          validation_data=(X_test, y_test))
          
score = model.evaluate(X_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

x = list(range(1,50+1))

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)

predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1)))

model.save('mnist_sound.h5')
