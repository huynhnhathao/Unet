import matplotlib.pyplot as plt

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
train_generator_= trainGenerator(20,'/content/drive/My Drive/NeuralImages/train','image','mask',data_gen_args,save_to_dir = '/content/augimages/images',
image_save_prefix = 'images', mask_save_prefix = 'masks')
test_generator= testGenerator('/content/drive/My Drive/NeuralImages/test', num_image= 5,target_size= (512,512), )
img_arr, mask_arr = geneTrainNpy('/content/drive/My Drive/NeuralImages/test/image',
                                 '/content/drive/My Drive/NeuralImages/test/mask',
                                 )
#callbacks to save model
from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint(filepath = '/content/save/model.ckpt',
                                   save_weights_only =True,
                                   monitor = 'val_accuracy',
                                   mode = 'max',
                                   save_best_only =  True)
history = model.fit_generator(train_generator_, epochs =40,steps_per_epoch= 30,  shuffle = True, 
                              callbacks = [model_checkpoint], validation_data = test_generator, validation_steps = 5)
                              
                             
#show model predict on test unseen data                             
import matplotlib.pyplot as plt
pmask = model.predict(np.expand_dims(img_arr[0], axis = 0))
fig, axs = plt.subplots(1,2, figsize = (12,9))
axs[0].imshow(np.squeeze(np.squeeze(pmask, axis = 0), axis = 2), cmap = 'gray')
axs[1].imshow(np.squeeze(mask_arr[0], axis = 2), cmap = 'gray')


#plot learning progress of model
def plot_(history, ep = 5):
    epochs = np.arange(1,ep +1)
    plt.plot(epochs, history.history['val_loss'], label = 'val_loss')
    plt.plot(epochs, history.history['loss'], label = 'loss')
    plt.ylim([0,1.0])
    plt.legend()
    
    plt.show()
    plt.plot(epochs, history.history['val_accuracy'], label = 'val_acc')
    plt.plot(epochs, history.history['accuracy'], label = 'acc')

    plt.legend()
    plt.show()
plot_(history)
