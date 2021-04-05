from model import model_resnet
from training_functions import LR_WarmRestart, MixupGenerator

NumClasses=10
num_audio_channels=1
NumFreqBins=128
max_lr=0.1
batch_size = 32
num_epochs = 510
mixup_alpha = 0.4
crop_length = 400

model = model_resnet(NumClasses,
                     input_shape =[NumFreqBins,None,3*num_audio_channels], 
                     num_filters =24,
                     wd=1e-3)
model.compile(loss='categorical_crossentropy',
              optimizer =SGD(lr=max_lr,decay=0, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

model.summary()




#set learning rate schedule
lr_scheduler = LR_WarmRestart(nbatch=np.ceil(LM_train.shape[0]/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0]) 
callbacks = [lr_scheduler]

#create data generator
TrainDataGen = MixupGenerator(LM_train, 
                              y_train, 
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length)()

#train the model
history = model.fit_generator(TrainDataGen,
                              validation_data=(LM_val, y_val),
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size = 100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(LM_train.shape[0]/batch_size)
                              )

