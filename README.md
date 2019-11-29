Base final validation accuracy:

Epoch 50/50
390/390 [==============================] - 21s 53ms/step - loss: 0.3204 - acc: 0.8949 - val_loss: 0.6423 - val_acc: 0.8169
Model took 1026.69 seconds to train
Accuracy on test data is: 81.69

New model defintion with RF and channel size

model.add(BatchNormalization())
model.add(SeparableConv2D(filters = 32, depth_multiplier = 4,kernel_size=(3, 3))) #channel Op: 30*30*32 RF: 5
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))  #channel Op: 15*15*32 RF: 6
model.add(Dropout(0.25))

model.add(SeparableConv2D(filters = 64, depth_multiplier = 4,kernel_size=(3, 3))) #channel Op: 13*13*64 RF: 10
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
781/781 [==============================] - 48s 62ms/step - loss: 1.3600 - acc: 0.5124 - val_loss: 1.4175 - val_acc: 0.5248
Epoch 2/50
781/781 [==============================] - 37s 47ms/step - loss: 1.0562 - acc: 0.6302 - val_loss: 1.0369 - val_acc: 0.6356
Epoch 3/50
781/781 [==============================] - 37s 47ms/step - loss: 0.9451 - acc: 0.6678 - val_loss: 0.9493 - val_acc: 0.6723
Epoch 4/50
781/781 [==============================] - 37s 47ms/step - loss: 0.8851 - acc: 0.6902 - val_loss: 0.9272 - val_acc: 0.6712
Epoch 5/50
781/781 [==============================] - 37s 47ms/step - loss: 0.8414 - acc: 0.7050 - val_loss: 0.8339 - val_acc: 0.7061
Epoch 6/50
781/781 [==============================] - 37s 47ms/step - loss: 0.8112 - acc: 0.7173 - val_loss: 0.7532 - val_acc: 0.7378
Epoch 7/50
781/781 [==============================] - 37s 47ms/step - loss: 0.7747 - acc: 0.7283 - val_loss: 0.7890 - val_acc: 0.7243
Epoch 8/50
781/781 [==============================] - 37s 47ms/step - loss: 0.7535 - acc: 0.7368 - val_loss: 0.9352 - val_acc: 0.6818
Epoch 9/50
781/781 [==============================] - 37s 47ms/step - loss: 0.7334 - acc: 0.7452 - val_loss: 0.7351 - val_acc: 0.7434
Epoch 10/50
781/781 [==============================] - 37s 47ms/step - loss: 0.7090 - acc: 0.7516 - val_loss: 0.7671 - val_acc: 0.7368
Epoch 11/50
781/781 [==============================] - 37s 47ms/step - loss: 0.6985 - acc: 0.7543 - val_loss: 0.7699 - val_acc: 0.7360
Epoch 12/50
781/781 [==============================] - 37s 47ms/step - loss: 0.6846 - acc: 0.7622 - val_loss: 0.8503 - val_acc: 0.7166
Epoch 13/50
781/781 [==============================] - 37s 47ms/step - loss: 0.6717 - acc: 0.7642 - val_loss: 0.6450 - val_acc: 0.7779
Epoch 14/50
781/781 [==============================] - 36s 46ms/step - loss: 0.6622 - acc: 0.7700 - val_loss: 0.6721 - val_acc: 0.7775
Epoch 15/50
781/781 [==============================] - 36s 46ms/step - loss: 0.6504 - acc: 0.7735 - val_loss: 0.6359 - val_acc: 0.7789
Epoch 16/50
781/781 [==============================] - 36s 46ms/step - loss: 0.6426 - acc: 0.7777 - val_loss: 0.6802 - val_acc: 0.7667
Epoch 17/50
781/781 [==============================] - 36s 46ms/step - loss: 0.6329 - acc: 0.7800 - val_loss: 0.6945 - val_acc: 0.7630
Epoch 18/50
781/781 [==============================] - 36s 46ms/step - loss: 0.6238 - acc: 0.7817 - val_loss: 0.6592 - val_acc: 0.7804
Epoch 19/50
781/781 [==============================] - 36s 46ms/step - loss: 0.6163 - acc: 0.7853 - val_loss: 0.6514 - val_acc: 0.7785
Epoch 20/50
781/781 [==============================] - 36s 46ms/step - loss: 0.6113 - acc: 0.7882 - val_loss: 0.6138 - val_acc: 0.7861
Epoch 21/50
781/781 [==============================] - 36s 46ms/step - loss: 0.6044 - acc: 0.7912 - val_loss: 0.6071 - val_acc: 0.7911
Epoch 22/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5981 - acc: 0.7920 - val_loss: 0.6238 - val_acc: 0.7890
Epoch 23/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5943 - acc: 0.7923 - val_loss: 0.6161 - val_acc: 0.7934
Epoch 24/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5881 - acc: 0.7944 - val_loss: 0.5788 - val_acc: 0.8016
Epoch 25/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5814 - acc: 0.7971 - val_loss: 0.5793 - val_acc: 0.8029
Epoch 26/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5767 - acc: 0.7988 - val_loss: 0.5764 - val_acc: 0.8064
Epoch 27/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5704 - acc: 0.8015 - val_loss: 0.5518 - val_acc: 0.8127
Epoch 28/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5704 - acc: 0.8022 - val_loss: 0.6211 - val_acc: 0.7853
Epoch 29/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5638 - acc: 0.8030 - val_loss: 0.5692 - val_acc: 0.8063
Epoch 30/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5637 - acc: 0.8024 - val_loss: 0.7683 - val_acc: 0.7476
Epoch 31/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5576 - acc: 0.8052 - val_loss: 0.6404 - val_acc: 0.7860
Epoch 32/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5509 - acc: 0.8072 - val_loss: 0.5879 - val_acc: 0.7988
Epoch 33/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5470 - acc: 0.8091 - val_loss: 0.5923 - val_acc: 0.7989
Epoch 34/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5453 - acc: 0.8097 - val_loss: 0.5973 - val_acc: 0.7964
Epoch 35/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5479 - acc: 0.8087 - val_loss: 0.5804 - val_acc: 0.8050
Epoch 36/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5388 - acc: 0.8107 - val_loss: 0.5440 - val_acc: 0.8169
Epoch 37/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5311 - acc: 0.8147 - val_loss: 0.5826 - val_acc: 0.7988
Epoch 38/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5302 - acc: 0.8163 - val_loss: 0.5797 - val_acc: 0.8060
Epoch 39/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5325 - acc: 0.8149 - val_loss: 0.5813 - val_acc: 0.8034
Epoch 40/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5309 - acc: 0.8127 - val_loss: 0.5424 - val_acc: 0.8172
Epoch 41/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5195 - acc: 0.8188 - val_loss: 0.6008 - val_acc: 0.8064
Epoch 42/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5246 - acc: 0.8150 - val_loss: 0.5642 - val_acc: 0.8102
Epoch 43/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5197 - acc: 0.8182 - val_loss: 0.5822 - val_acc: 0.8053
Epoch 44/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5133 - acc: 0.8197 - val_loss: 0.5620 - val_acc: 0.8112
Epoch 45/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5136 - acc: 0.8214 - val_loss: 0.5855 - val_acc: 0.8045
Epoch 46/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5125 - acc: 0.8198 - val_loss: 0.5620 - val_acc: 0.8103
Epoch 47/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5110 - acc: 0.8215 - val_loss: 0.5558 - val_acc: 0.8139
Epoch 48/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5109 - acc: 0.8201 - val_loss: 0.5613 - val_acc: 0.8093
Epoch 49/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5102 - acc: 0.8203 - val_loss: 0.5315 - val_acc: 0.8166
Epoch 50/50
781/781 [==============================] - 36s 46ms/step - loss: 0.5003 - acc: 0.8257 - val_loss: 0.5561 - val_acc: 0.8152
Model took 1822.75 seconds to train

