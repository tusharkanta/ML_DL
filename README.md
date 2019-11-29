Base final validation accuracy:

Epoch 50/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3422 - acc: 0.8858 - val_loss: 0.5588 - val_acc: 0.8285
Model took 1024.79 seconds to train
Accuracy on test data is: 82.85

New model defintion with RF and channel size
model = Sequential()
model.add(SeparableConv2D(filters = 16, kernel_size=(3, 3),          #channel Op: 32*32*16 RF: 3
            depth_multiplier = 4, padding = 'same',
            input_shape = (32,32,3)))

model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(filters = 32, depth_multiplier = 5,kernel_size=(3, 3))) #channel Op: 30*30*32 RF: 5
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))  #channel Op: 15*15*32 RF: 6
model.add(Dropout(0.25))

model.add(SeparableConv2D(filters = 64, depth_multiplier = 5,kernel_size=(3, 3))) #channel Op: 13*13*64 RF: 10
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(SeparableConv2D(filters = 128, depth_multiplier = 3,kernel_size=(3, 3))) #channel Op: 11*11*128   RF: 14
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) #channel Op: 5*5*128   RF: 16
model.add(Dropout(0.50))

model.add(SeparableConv2D(filters = 256 ,kernel_size=(3, 3))) #channel Op: 3*3*256   RF: 24
#model.add(Activation('relu'))


model.add(Flatten())
#model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

Epoch 1/50
390/390 [==============================] - 34s 87ms/step - loss: 1.4268 - acc: 0.4828 - val_loss: 1.2551 - val_acc: 0.5637
Epoch 2/50
390/390 [==============================] - 33s 83ms/step - loss: 1.0985 - acc: 0.6075 - val_loss: 0.9807 - val_acc: 0.6614
Epoch 3/50
390/390 [==============================] - 33s 83ms/step - loss: 0.9602 - acc: 0.6621 - val_loss: 0.8617 - val_acc: 0.7006
Epoch 4/50
390/390 [==============================] - 32s 83ms/step - loss: 0.8783 - acc: 0.6916 - val_loss: 0.7858 - val_acc: 0.7283
Epoch 5/50
390/390 [==============================] - 32s 83ms/step - loss: 0.8302 - acc: 0.7094 - val_loss: 0.8004 - val_acc: 0.7205
Epoch 6/50
390/390 [==============================] - 32s 83ms/step - loss: 0.7885 - acc: 0.7244 - val_loss: 0.7482 - val_acc: 0.7425
Epoch 7/50
390/390 [==============================] - 33s 84ms/step - loss: 0.7550 - acc: 0.7360 - val_loss: 0.7614 - val_acc: 0.7398
Epoch 8/50
390/390 [==============================] - 32s 83ms/step - loss: 0.7293 - acc: 0.7447 - val_loss: 0.7903 - val_acc: 0.7329
Epoch 9/50
390/390 [==============================] - 32s 83ms/step - loss: 0.7027 - acc: 0.7536 - val_loss: 0.7693 - val_acc: 0.7320
Epoch 10/50
390/390 [==============================] - 33s 83ms/step - loss: 0.6874 - acc: 0.7594 - val_loss: 0.7282 - val_acc: 0.7503
Epoch 11/50
390/390 [==============================] - 32s 83ms/step - loss: 0.6718 - acc: 0.7661 - val_loss: 0.7760 - val_acc: 0.7409
Epoch 12/50
390/390 [==============================] - 32s 83ms/step - loss: 0.6516 - acc: 0.7724 - val_loss: 0.6677 - val_acc: 0.7708
Epoch 13/50
390/390 [==============================] - 32s 83ms/step - loss: 0.6422 - acc: 0.7748 - val_loss: 0.7094 - val_acc: 0.7566
Epoch 14/50
390/390 [==============================] - 32s 83ms/step - loss: 0.6265 - acc: 0.7806 - val_loss: 0.6188 - val_acc: 0.7872
Epoch 15/50
390/390 [==============================] - 32s 83ms/step - loss: 0.6166 - acc: 0.7828 - val_loss: 0.6681 - val_acc: 0.7726
Epoch 16/50
390/390 [==============================] - 32s 83ms/step - loss: 0.6077 - acc: 0.7869 - val_loss: 0.6262 - val_acc: 0.7823
Epoch 17/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5985 - acc: 0.7924 - val_loss: 0.6192 - val_acc: 0.7872
Epoch 18/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5904 - acc: 0.7942 - val_loss: 0.6385 - val_acc: 0.7830
Epoch 19/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5797 - acc: 0.7967 - val_loss: 0.5936 - val_acc: 0.7975
Epoch 20/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5735 - acc: 0.7989 - val_loss: 0.6530 - val_acc: 0.7874
Epoch 21/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5686 - acc: 0.8012 - val_loss: 0.6588 - val_acc: 0.7724
Epoch 22/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5619 - acc: 0.8039 - val_loss: 0.6164 - val_acc: 0.7957
Epoch 23/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5487 - acc: 0.8064 - val_loss: 0.6166 - val_acc: 0.7906
Epoch 24/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5421 - acc: 0.8087 - val_loss: 0.5964 - val_acc: 0.7988
Epoch 25/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5390 - acc: 0.8112 - val_loss: 0.6378 - val_acc: 0.7869
Epoch 26/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5356 - acc: 0.8120 - val_loss: 0.5851 - val_acc: 0.8054
Epoch 27/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5289 - acc: 0.8144 - val_loss: 0.6113 - val_acc: 0.7935
Epoch 28/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5222 - acc: 0.8175 - val_loss: 0.5645 - val_acc: 0.8116
Epoch 29/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5207 - acc: 0.8185 - val_loss: 0.6146 - val_acc: 0.7931
Epoch 30/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5146 - acc: 0.8198 - val_loss: 0.6222 - val_acc: 0.7925
Epoch 31/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5077 - acc: 0.8223 - val_loss: 0.5726 - val_acc: 0.8064
Epoch 32/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5098 - acc: 0.8207 - val_loss: 0.5543 - val_acc: 0.8132
Epoch 33/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5036 - acc: 0.8231 - val_loss: 0.5935 - val_acc: 0.7995
Epoch 34/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5036 - acc: 0.8240 - val_loss: 0.5728 - val_acc: 0.8076
Epoch 35/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4920 - acc: 0.8270 - val_loss: 0.5842 - val_acc: 0.8042
Epoch 36/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4963 - acc: 0.8250 - val_loss: 0.5819 - val_acc: 0.8016
Epoch 37/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4936 - acc: 0.8247 - val_loss: 0.5629 - val_acc: 0.8092
Epoch 38/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4835 - acc: 0.8279 - val_loss: 0.5483 - val_acc: 0.8149
Epoch 39/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4798 - acc: 0.8325 - val_loss: 0.5521 - val_acc: 0.8130
Epoch 40/50
390/390 [==============================] - 33s 83ms/step - loss: 0.4788 - acc: 0.8311 - val_loss: 0.5956 - val_acc: 0.8029
Epoch 41/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4794 - acc: 0.8308 - val_loss: 0.5412 - val_acc: 0.8185
Epoch 42/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4742 - acc: 0.8314 - val_loss: 0.5203 - val_acc: 0.8263
Epoch 43/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4699 - acc: 0.8337 - val_loss: 0.5602 - val_acc: 0.8133
Epoch 44/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4679 - acc: 0.8344 - val_loss: 0.5472 - val_acc: 0.8209
Epoch 45/50
390/390 [==============================] - 33s 83ms/step - loss: 0.4590 - acc: 0.8380 - val_loss: 0.5719 - val_acc: 0.8140
Epoch 46/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4626 - acc: 0.8376 - val_loss: 0.5586 - val_acc: 0.8122
Epoch 47/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4618 - acc: 0.8366 - val_loss: 0.5300 - val_acc: 0.8229
Epoch 48/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4573 - acc: 0.8373 - val_loss: 0.5615 - val_acc: 0.8130
Epoch 49/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4566 - acc: 0.8375 - val_loss: 0.5183 - val_acc: 0.8298
Epoch 50/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4519 - acc: 0.8399 - val_loss: 0.5651 - val_acc: 0.8149
Model took 1621.30 seconds to train

