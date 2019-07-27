'''
    Copyright (c) 2019, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the BSD license.
    URL: https://opensource.org/licenses/BSD-2-Clause
    
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import sys

if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a CSV/.h5 file for training image paths')
    print('  argv[2] : Path to a CSV/.h5 file for validation image paths')
    print('  argv[3] : Path to a directory to save results in it')
    print('  argv[4] : Path to a model to be RE-trained')
    print('  NOTE : Input images must be 200x200 gray-scale without alpha values')
    sys.exit()


# Import modules
import os
import shutil
import time
from datetime import datetime

# For calculation with 16-bit float
# import keras.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)     # default is 1e-7 which is too small for float16. Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems.

# import tensorflow as tf
# import keras.backend as K
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.Session(config=config)
# K.set_session(sess)

# For PlaidML
#import plaidml.keras
#plaidml.keras.install_backend()

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler
from keras.utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt


# Neural network parameters
#####################################################################

sys.path.append('C:\\Users\\takashi\\OneDrive\\documents\\Keras+CoreML\\code')

# Metrics
from utils.Validation_func import mean_iou, dice_coef

# Loss functions
#from utils.Validation_func import mean_iou_loss as LOSS        ### Not bad ###
from utils.Validation_func import mean_iou_MSE_loss as LOSS    ###### Best performance ######
#from utils.Validation_func import mean_iou_MAE_loss as LOSS    ### Good ###
#from utils.Validation_func import mean_iou_SVM_loss as LOSS
#from utils.Validation_func import mean_iou_SSVM_loss as LOSS
#from utils.Validation_func import mean_iou_LGC_loss as LOSS
#from utils.Validation_func import dice_coef_loss as LOSS       ### Not bad ###
#from utils.Validation_func import dice_coef_MSE_loss as LOSS    ### Best performance ###
#from utils.Validation_func import dice_coef_MAE_loss as LOSS   ### Good ###
#from utils.Validation_func import dice_coef_SVM_loss as LOSS
#from utils.Validation_func import dice_coef_SSVM_loss as LOSS
#from utils.Validation_func import dice_coef_LGC_loss as LOSS
#from keras.losses import mean_squared_error as LOSS            ### Not good ###
#from keras.losses import mean_absolute_error as LOSS           ### Not good ###

pram_monitor    = ['val_'+mean_iou.__name__, 'max']
# pram_LR_points  = [[0, 7.81e-4], [5, 7.81e-4], [15, 6e-4], [30, 2e-4], [35, 1e-4], [50, 2e-5]]     # For aorta   [epoch, learning rate]
# pram_LR_points  = [[0, 7.81e-4], [40, 7.81e-4], [70, 6e-4], [130, 2e-4], [160, 1e-4], [200, 2e-5]]     # For heart   [epoch, learning rate]
pram_LR_points  = [[0, 7.81e-4], [4, 7.81e-4], [10, 2e-5]]     # For bone   [epoch, learning rate]
pram_batch_size = 16
pram_init_epoch = 5    #pram_LR_points[0][0]          # Initial epoch of train (starting from zero)
pram_epochs     = pram_LR_points[-1][0]         # Total count of epochs
pram_init_LR    = pram_LR_points[0][1]
pram_final_LR   = pram_LR_points[-1][1]


# Define callbacks for learning rate
#####################################################################

nptloop = len(pram_LR_points) - 1

def LRSteps(epoch):
    
    def LR_at_epoch(epoch, pt1, pt2):
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]) * (epoch - pt1[0]) + pt1[1]

    for i in range(nptloop):
        if pram_LR_points[i][0] <= epoch and epoch < pram_LR_points[i+1][0]:
            x = LR_at_epoch(epoch, pram_LR_points[i], pram_LR_points[i+1])
            break
    return x


# Model load and RE-learning
#####################################################################

#model = load_model(sys.argv[4], custom_objects={'mean_iou': mean_iou})
model = load_model(sys.argv[4], custom_objects={LOSS.__name__: LOSS, 'mean_iou': mean_iou, 'dice_coef': dice_coef})

model.summary()
print('Loaded NN         : {0}'.format(sys.argv[4]))
print('__________________________________________________________________________________________________')
print('Loss              : {0}'.format(LOSS.__name__))
print('Metrics 1         : {0}'.format(model.metrics_names[1]))
print('Metrics 2         : {0}'.format(model.metrics_names[2]))
print('Monitor for best  : {0}'.format(pram_monitor))
print('Batch size        : {0}'.format(pram_batch_size))
print('Initial epoch     : {0}'.format(pram_init_epoch))
print('Epochs            : {0}'.format(pram_epochs))
print('Initial LR        : {0}'.format(pram_init_LR))
print('Final LR          : {0}'.format(pram_final_LR))
print('LR points         : {0}'.format(pram_LR_points))
print('==================================================================================================')


# Checkpoint
# key = input('Continue? [y/n] : ')
# if key != 'y' and key != 'Y':
#     print('Exit...')
#     sys.exit()


# Paths and directories
datestr = datetime.now().strftime("%Y%m%d%H%M%S")
traindir_path = os.path.join(sys.argv[3], 'run'+datestr+'(retrain)')
codedir_path = os.path.join(traindir_path, 'code')
tmp_path = os.path.join(traindir_path,'tmp_model'+datestr)
os.mkdir(traindir_path)
os.mkdir(codedir_path)
shutil.copy2(__file__, os.path.join(codedir_path, os.path.basename(__file__)))


# Save network figure and file paths
plot_model(model, to_file=os.path.join(traindir_path,'model_figure.png'), show_shapes=True, show_layer_names=False)
with open(os.path.join(traindir_path,'training_parameters.txt'), mode='w') as path_file:
    path_file.write('RE-trained model  : {0}\n\n'.format(sys.argv[4]))
    path_file.write('Training images   : {0}\n'.format(sys.argv[1]))
    path_file.write('Validation images : {0}\n\n'.format(sys.argv[2]))
    path_file.write('Loss              : {0}\n'.format(LOSS.__name__))
    path_file.write('Metrics 1         : {0}\n'.format(model.metrics_names[1]))
    path_file.write('Metrics 2         : {0}\n'.format(model.metrics_names[2]))
    path_file.write('Monitor for best  : {0}\n'.format(pram_monitor))
    path_file.write('Batch size        : {0}\n'.format(pram_batch_size))
    path_file.write('Initial epoch     : {0}\n'.format(pram_init_epoch))
    path_file.write('Epochs            : {0}\n'.format(pram_epochs))
    path_file.write('Initial LR        : {0}\n'.format(pram_init_LR))
    path_file.write('Final LR          : {0}\n'.format(pram_final_LR))
    path_file.write('LR points         : {0}\n'.format(pram_LR_points))


# Define callbacks
print('Defining callbacks...')
checkpointer = ModelCheckpoint(tmp_path, monitor=pram_monitor[0], verbose=1, save_best_only=True, mode=pram_monitor[1])
scheduleLR = LearningRateScheduler(LRSteps, verbose=1)
csvlogger = CSVLogger(os.path.join(traindir_path,'training_log.csv'), separator=',', append=False)
tensorboard = TensorBoard(traindir_path, histogram_freq=0, batch_size=pram_batch_size, write_graph=True)


# Data generator
from utils.Image_data_generator import ImageDataGenerator_CSV_with_Header, ImageDataGenerator_h5_Dataset

print('Loading images for training...')
ext = os.path.splitext(sys.argv[1])[1]
if   ext == '.csv' :  gen1 = ImageDataGenerator_CSV_with_Header('Train data from CSV', sys.argv[1], shuffle=True)
elif ext == '.h5'  :  gen1 = ImageDataGenerator_h5_Dataset('image_training', sys.argv[1])
else               :  sys.exit()
print('Loading images for validation...')
ext = os.path.splitext(sys.argv[2])[1]
if   ext == '.csv' :  gen2 = ImageDataGenerator_CSV_with_Header('Validation data from CSV', sys.argv[2], shuffle=True)
elif ext == '.h5'  :  gen2 = ImageDataGenerator_h5_Dataset('image_validation', sys.argv[2])
else               :  sys.exit()


# Train the network
print('Starting model lerning...')
starttime = time.time()
results = model.fit_generator(gen1.flow(rescale=1.0/225.0, batch_size=pram_batch_size),
    steps_per_epoch         = gen1.length() // pram_batch_size,
    epochs                  = pram_epochs,
    verbose                 = 1,
    callbacks               = [checkpointer, scheduleLR, csvlogger, tensorboard],
	validation_data         = gen2.flow(rescale=1.0/225.0, batch_size=pram_batch_size),
    validation_steps        = gen2.length() // pram_batch_size,
    max_queue_size          = 10,
    workers                 = 1,
    use_multiprocessing     = False,
    shuffle                 = True,
    initial_epoch           = pram_init_epoch )


# Show results
print('Showing and saving results...')
his_loss = results.history[model.metrics_names[0]]
his_met1 = results.history[model.metrics_names[1]]
his_met2 = results.history[model.metrics_names[2]]
his_valloss = results.history['val_'+model.metrics_names[0]]
his_valmet1 = results.history['val_'+model.metrics_names[1]]
his_valmet2 = results.history['val_'+model.metrics_names[2]]
xlen = range(len(his_loss))

fig = plt.figure()
ax1 = fig.add_subplot(111)      # Loss
ax2 = ax1.twinx()

ax1.plot(xlen, his_loss, marker='.', color='salmon', label=LOSS.__name__)
ax1.plot(xlen, his_valloss, marker='.', color='red', label='val_'+LOSS.__name__)
ax2.plot(xlen, his_met1, marker='.', color='deepskyblue', label=model.metrics_names[1])
ax2.plot(xlen, his_valmet1, marker='.', color='blue', label='val_'+model.metrics_names[1])
ax2.plot(xlen, his_met2, marker='.', color='limegreen', label=model.metrics_names[2])
ax2.plot(xlen, his_valmet2, marker='.', color='green', label='val_'+model.metrics_names[2])

ax1.set_xlabel('Epoch')
ax1.set_ylabel(LOSS.__name__)
ax1.set_yscale("log")
ax1.set_ylim([0.001, 1.0])
ax2.set_ylabel('Metrics')
ax2.set_yscale("log")
ax2.set_ylim([0.8, 1.0])

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower center')


print('==================================================================================================')
model.summary()
print('Loaded NN         : {0}'.format(sys.argv[4]))
print('__________________________________________________________________________________________________')
print('Loss              : {0}'.format(LOSS.__name__))
print('Metrics 1         : {0}'.format(model.metrics_names[1]))
print('Metrics 2         : {0}'.format(model.metrics_names[2]))
print('Monitor for best  : {0}'.format(pram_monitor))
print('Batch size        : {0}'.format(pram_batch_size))
print('Initial epoch     : {0}'.format(pram_init_epoch))
print('Epochs            : {0}'.format(pram_epochs))
print('Initial LR        : {0}'.format(pram_init_LR))
print('Final LR          : {0}'.format(pram_final_LR))
print('LR points         : {0}'.format(pram_LR_points))
print('__________________________________________________________________________________________________')
print('Elapsed time      : {0}'.format(time.time() - starttime) + " [sec]")
print('==================================================================================================')

plt.savefig(os.path.join(traindir_path,'training_graph.png'))

loadedfilename = os.path.basename(os.path.splitext(sys.argv[4])[0])
infostr = '{0} {1}={2:.4f} retrained from {3}.h5'.format(datestr, model.metrics_names[1], max(his_valmet1), loadedfilename)
shutil.copy(tmp_path, os.path.join(traindir_path, 'model'+infostr))

# model.save(os.path.join(traindir_path, 'model(wo optimizer)'+infostr), include_optimizer=False)
#plt.show()


