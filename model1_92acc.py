from keras.backend import clear_session
clear_session()
from keras.models import *
from keras.layers import *
H = 512
W = 512
C = 1
inputs = Input((H, W, C))
#CONTRACTION PART
#(512,512,1)
conv1 = Conv2D(16, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same')(inputs)
drop1 = Dropout(0.1)(conv1)
conv1 = Conv2D(16, (3,3), activation= 'relu', kernel_initializer= 'he_normal', padding = 'same')(drop1)
pool1 = MaxPooling2D((2,2))(drop1)

#(256,256,16)
conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(pool1)
drop2 = Dropout(0.2)(conv2)
conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(drop2)
pool2 = MaxPooling2D((2,2))(conv2)

#(128,128, 32)
conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(pool2)
drop3 = Dropout(0.2)(pool2)
conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(drop3)
pool3 = MaxPooling2D((2,2))(conv3)

#(64,64,64)
conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(pool3)
drop4 = Dropout(0.2)(conv4)
conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(drop4)
pool4 = MaxPooling2D((2,2))(conv4)

#(32,32, 128)
conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(pool4)
drop5 = Dropout(0.2)(conv5)
conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer= 'he_normal')(drop5)

#EXPANSIVE PART
#(32,32,256)
up6 =Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5) #(64,64,128)
up6 = concatenate([up6, conv4])# up6(64,64,128)  concatenate with pool4(64,64,128)  = (64,64,256)
conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up6) # (64,64,128)
drop6 = Dropout(0.2)(conv6)
conv6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop6)

#(64,64,128)
up7 = Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(conv6)
up7 = concatenate([up7,conv3 ]) # up7(128,128,64) concatenate with conv3(128,128,64) = (256,256,128)
conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)
#(128,128,64)
drop7 = Dropout(0.2)(conv7)
conv7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop7)

#(128,128,64)
up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)#(256,256,32)
up8 = concatenate([up8,conv2])#up8(256,256,32) concatenate with conv2(256,256,32) = (256,256,64)
conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up8)
drop8 = Dropout(0.1)(conv8)
conv8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop8)

#(256,256,32)
up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
up9 = concatenate([up9, conv1]) #up9(512,512,16) concatenate with conv1(512,512,16) = (512,512,32)
conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)
drop9 = Dropout(0.1)(conv9)
conv9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop9)

#(512,512,16)

outputs = Conv2D(1,(1,1), activation = 'sigmoid')(conv9)
model = Model(input= [inputs], output = [outputs])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()



