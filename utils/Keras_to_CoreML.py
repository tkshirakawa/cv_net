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


# Import modules
import os
import sys
import numpy as np

from utils.Validation_func import mean_iou, dice_coef

#from utils.Validation_func import mean_iou_loss as LOSS        ### Not bad ###
from utils.Validation_func import mean_iou_MSE_loss as LOSS     ###### Best performance ######
#from utils.Validation_func import mean_iou_MAE_loss as LOSS    ### Good ###
#from utils.Validation_func import mean_iou_SVM_loss as LOSS
#from utils.Validation_func import mean_iou_SSVM_loss as LOSS
#from utils.Validation_func import mean_iou_LGC_loss as LOSS
#from utils.Validation_func import dice_coef_loss as LOSS       ### Not bad ###
#from utils.Validation_func import dice_coef_MSE_loss as LOSS   ### Better performance ###
#from utils.Validation_func import dice_coef_MAE_loss as LOSS   ### Good ###
#from utils.Validation_func import dice_coef_SVM_loss as LOSS
#from utils.Validation_func import dice_coef_SSVM_loss as LOSS
#from utils.Validation_func import dice_coef_LGC_loss as LOSS
#from keras.losses import mean_squared_error as LOSS            ### Not good ###
#from keras.losses import mean_absolute_error as LOSS           ### Not good ###


# Convert Keras model to CoreML model
#####################################################################

if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a directory to save generated connverted CoreML file in it')
    print('  argv[2] : Model with custom object (mean_iou for Keras metrics)? (y/n)')
    print('  argv[3] : Path to a Keras parameter file (.h5) to be converted')
    sys.exit()


# Convert Keras model
if sys.argv[2] == 'n':
    import coremltools
    model = coremltools.converters.keras.convert(sys.argv[3],
                                                 input_names  = 'input',
                                                 output_names = 'output' )
    spec = model.get_spec()

elif sys.argv[2] == 'y':
    from coremltools.converters.keras import _keras_converter
    from coremltools.models import MLModel
    spec = _keras_converter.convertToSpec(sys.argv[3],
                                          input_names    = 'input',
                                          output_names   = 'output',
                                          custom_objects = {LOSS.__name__: LOSS, 'mean_iou': mean_iou, 'dice_coef': dice_coef} )
    model = MLModel(spec)

else:
    print('Use (y) or (n).')
    sys.exit()


# Set descriptions
model.author = 'Takashi Shirakawa'
model.license = '(C) 2019, Takashi Shirakawa. All right reserved.'
model.short_description = 'CoreML model for A.I.Segmentation'
model.input_description['input'] = 'Input grayscale image must be passed to a ML model in MLMultiArray with 1x200x200 (Channel x Height x Width).'
model.output_description['output'] = 'Predicted image must be saved in the same format, MLMultiArray with 1x200x200 (Channel x Height x Width).'

# Save mlmodel
model.save(os.path.join(sys.argv[1], os.path.splitext(os.path.basename(sys.argv[3]))[0] + '.mlmodel') )

# Show results
print('----------------------------------------------------------')
print('Parameter file : ' + sys.argv[3])
print('Model descriptions :')
print(spec.description)
print('Done.')
print('----------------------------------------------------------')

