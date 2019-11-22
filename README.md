# ML_DL
Test data score:
[0.017526581212430028, 0.9953]

Epoch Log
=========
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 11s 187us/step - loss: 0.1168 - acc: 0.9514 - val_loss: 0.0316 - val_acc: 0.9906
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 7s 112us/step - loss: 0.1044 - acc: 0.9542 - val_loss: 0.0232 - val_acc: 0.9927
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 7s 111us/step - loss: 0.1017 - acc: 0.9555 - val_loss: 0.0230 - val_acc: 0.9934
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 7s 108us/step - loss: 0.0966 - acc: 0.9564 - val_loss: 0.0209 - val_acc: 0.9939
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0946 - acc: 0.9560 - val_loss: 0.0217 - val_acc: 0.9933
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 7s 108us/step - loss: 0.0906 - acc: 0.9562 - val_loss: 0.0223 - val_acc: 0.9921
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 6s 108us/step - loss: 0.0890 - acc: 0.9569 - val_loss: 0.0196 - val_acc: 0.9935
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 6s 108us/step - loss: 0.0875 - acc: 0.9576 - val_loss: 0.0176 - val_acc: 0.9947
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0878 - acc: 0.9568 - val_loss: 0.0185 - val_acc: 0.9940
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0858 - acc: 0.9575 - val_loss: 0.0162 - val_acc: 0.9947
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 7s 110us/step - loss: 0.0849 - acc: 0.9573 - val_loss: 0.0193 - val_acc: 0.9942
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0855 - acc: 0.9582 - val_loss: 0.0174 - val_acc: 0.9946
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 6s 108us/step - loss: 0.0865 - acc: 0.9582 - val_loss: 0.0171 - val_acc: 0.9951
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0830 - acc: 0.9588 - val_loss: 0.0196 - val_acc: 0.9946
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0841 - acc: 0.9561 - val_loss: 0.0185 - val_acc: 0.9949
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 7s 108us/step - loss: 0.0835 - acc: 0.9590 - val_loss: 0.0180 - val_acc: 0.9948
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 7s 108us/step - loss: 0.0804 - acc: 0.9598 - val_loss: 0.0172 - val_acc: 0.9958
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0801 - acc: 0.9610 - val_loss: 0.0181 - val_acc: 0.9951
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0808 - acc: 0.9592 - val_loss: 0.0180 - val_acc: 0.9951
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 7s 109us/step - loss: 0.0795 - acc: 0.9595 - val_loss: 0.0175 - val_acc: 0.9953
<keras.callbacks.History at 0x7f51c2334198>

Strategy to achieve the result:

1) To reduce the parameters changed number of kernels in 2nd convolution from 32 to 16
2) Increased the 3rd dropout rate to 0.15 
3) Slowed down the learning rate further
