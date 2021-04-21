import os

import numpy as np
import scipy.io as sio
from tensorflow.python.util import object_identity


def model_summary(model):
  if hasattr(model, '_collected_trainable_weights'):
    trainable_count = count_params(model._collected_trainable_weights)
  else:
    trainable_count = count_params(model.trainable_weights)

  non_trainable_count = count_params(model.non_trainable_weights)
  flops = count_FLOPs(model)

  return trainable_count, non_trainable_count, flops


def count_params(weights):
  """Count the total number of scalars composing the weights.

  Arguments:
      weights: An iterable containing the weights on which to compute params

  Returns:
      The total number of scalars composing the weights
  """
  unique_weights = object_identity.ObjectIdentitySet(weights)
  weight_shapes = [w.shape.as_list() for w in unique_weights]
  standardized_weight_shapes = [[0 if w_i is None else w_i for w_i in w] for w in weight_shapes]
  return int(sum(np.prod(p) for p in standardized_weight_shapes))


def loadData(name):
    data_path = os.path.join(os.getcwd(),'HSI data')
    if name =='BO':
        data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
        label_names = []
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'Kennedy_space_center.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'Kennedy_space_center_gt.mat'))['KSC_gt']
        label_names = ['Scrub','Willow swamp','CP hammock','Slash pine','Oak/Broadleaf','Hardwood','Swamp',
        'Gramionoi marsh','Spartina marsh','Cattail marsh','Salt marsh','Mud flats','Water']
    elif name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        label_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn','Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill','Soybean-clean', 'Wheat', 'Woods', 
                    'Buildings-Grass-Trees-Drives','Stone-Steel-Towers']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        label_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth','Stubble',
                    'Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds','Lettuce_romaine_4wk',
                    'Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        label_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
    elif name == 'HU':
        data = sio.loadmat(os.path.join(data_path, 'Houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(data_path, 'Houston_gt.mat'))['houston_gt_tr']
        label_names = ['GrassHealthy','GrassStressed','GrassSynthetic','Tree','Soil','Water','Residential','Commercial',
                    'Road','Highway','Railway','Parking Lot 1','Parking Lot 2','Tennis Court','Running Track']
    else:
        data = None
        labels = None
    return data, labels, label_names


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5):
    margin = int((windowSize-1)/2) 
    zeroPaddedX = np.zeros((X.shape[0]+2*margin, X.shape[1]+2*margin, X.shape[2]))
    zeroPaddedX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X           
    patchesData = np.zeros((y[y>0].size, windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((y[y>0].size))
    patchIndex = 0
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            if y[r,c]>0:
                patch = zeroPaddedX[r:r+windowSize, c:c+windowSize]   
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r,c]
                patchIndex = patchIndex + 1
    patchesLabels -= 1
    return patchesData, patchesLabels


def count_FLOPs(model):
    '''FLOPs的计算：    
    参考https://www.zhihu.com/question/65305385
    以及https://blog.csdn.net/weixin_43915709/article/details/94566125
    '''
    FLOPs = 0
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if layer.__class__.__name__ == 'Conv3D':
            FLOPs += 2*np.prod(layer.kernel_size[:3])*np.prod(layer.output_shape[1:5])
        if layer.__class__.__name__ == 'Conv2D':
            FLOPs += 2*np.prod(layer.kernel_size[:2])*layer.input_shape[3]*np.prod(layer.output_shape[1:4])
        if layer.__class__.__name__ in ('MaxPooling3D', 'AveragePooling3D'):
            FLOPs += np.prod(layer.pool_size[:3])*np.prod(layer.output_shape[1:5])
        if layer.__class__.__name__ in ('MaxPooling2D', 'AveragePooling2D'):
            FLOPs += np.prod(layer.pool_size[:2])*np.prod(layer.output_shape[1:4])
        if layer.__class__.__name__ == 'GlobalAveragePooling3D':
            FLOPs += np.prod(layer.input_shape[1:5])
        if layer.__class__.__name__ == 'GlobalAveragePooling2D':
            FLOPs += np.prod(layer.input_shape[1:4])             
        if layer.__class__.__name__ in ('Activation', 'Add', 'Multiply'):
            FLOPs += np.prod(layer.output_shape[1:])            
        if layer.__class__.__name__ == 'Dense':
            FLOPs += 2*layer.input_shape[1]*layer.output_shape[1]
        #if layer.__class__.__name__ == 'BatchNormalization':               
    return FLOPs