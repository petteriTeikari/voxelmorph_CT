"""
Example script to register two volumes with VoxelMorph models

Please make sure to use trained models appropriately. 
Let's say we have a model trained to register subject (moving) to atlas (fixed)
One could run:

python register.py --gpu 0 /path/to/test_vol.nii.gz /path/to/atlas_norm.nii.gz --out_img /path/to/out.nii.gz --model_file ../models/cvpr2018_vm2_cc.h5 
"""

# py imports
import os
import sys
from argparse import ArgumentParser

# third party
import tensorflow as tf
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
import nibabel as nib

# project
import networks, losses
sys.path.append('../ext/neuron')
import neuron.layers as nrn_layers

def register(gpu_id, moving, fixed, model_file, out_img, out_warp):
    """
    register moving and fixed. 
    """  
    assert model_file, "A model file is necessary"
    assert out_img or out_warp, "output image or warp file needs to be specified"

    # GPU handling
    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    # load data
    if_normalize_moving = True
    mov_nii = nib.load(moving)
    mov = mov_nii.get_data()[np.newaxis, ..., np.newaxis]
    if if_normalize_moving:
        # Typically the CTs that we used for other stuff are not clipped+normalized, that is required
        # hard-coded range and norm, the same as in "convert_niftis_to_npz.py"
        clip_range = [0, 100] # HU
        mov_orig = mov
        mov[mov < clip_range[0]] = clip_range[0]
        mov[mov > clip_range[1]] = clip_range[1]
        mov = mov/ np.max(mov)

    fix_nii = nib.load(fixed)
    fix = fix_nii.get_data()[np.newaxis, ..., np.newaxis]

    print('\n===\nLoading "moving" data (as in your input CT) from {}'.format(moving))
    print(' mov.shape:', mov.shape)
    print(' mov (input):', np.amin(mov_orig), ',', np.amax(mov_orig), ']')
    print(' mov (intensity range, after clipping+normalization):', np.amin(mov), ',', np.amax(mov), ']')
    print('Loading "fixed" data (as in your CT atlas/template) from {}'.format(fixed))
    print(' fix.shape:', fix.shape)
    print(' fix (intensity range):', np.amin(fix), ',', np.amax(fix), ']')

    image_sigma = 0.02
    prior_lambda = 10

    flow_vol_shape = None # Petteri: this was the default

    # PETTERI: Without this, not exactly sure why the shape needs to be 1/2 of the input shape
    # TODO! check from paper, or inspect code a bit better
    flow_vol_shape = (fix.shape[1]/2, fix.shape[2]/2, fix.shape[3]/2)

    with tf.device(gpu):
        # load model
        custom_objects = {'SpatialTransformer': nrn_layers.SpatialTransformer,
                 'VecInt': nrn_layers.VecInt,
                 'Sample': networks.Sample,
                 'Rescale': networks.RescaleDouble,
                 'Resize': networks.ResizeDouble,
                 'Negate': networks.Negate,
                 'recon_loss': losses.Miccai2018(image_sigma, prior_lambda, flow_vol_shape=flow_vol_shape).recon_loss, # values shouldn't matter
                 'kl_loss': losses.Miccai2018(image_sigma, prior_lambda, flow_vol_shape=flow_vol_shape).kl_loss        # values shouldn't matter
                 }

        print('Loading pre-trained model for registration from {}'.format(model_file))
        print('\n===\n')
        net = keras.models.load_model(model_file, custom_objects=custom_objects)


        # register
        [moved, warp] = net.predict([mov, fix])

    # output image
    if out_img is not None:
        img = nib.Nifti1Image(moved[0,...,0], mov_nii.affine)
        nib.save(img, out_img)

    # output warp
    if out_warp is not None:
        img = nib.Nifti1Image(warp[0,...], mov_nii.affine)
        nib.save(img, out_warp)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # positional arguments
    parser.add_argument("moving", type=str, default='../data/test_vol_01006_2mm.nii.gz',
                        help="moving file name")
    parser.add_argument("fixed", type=str, default='../data/JohnHopkins_CQ500_template_2.0mm_norm.nii.gz',
                        help="fixed file name")

    # optional arguments
    parser.add_argument("--model_file", type=str,
                        dest="model_file", default='../models/sCROMIS2ICH_2mm_128vx.h5',
                        help="models h5 file")
    parser.add_argument("--gpu", type=int, default=None,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--out_img", type=str, default='../data/test_vol_01006_2mm_registered.nii.gz',
                        dest="out_img", help="output image file name")
    parser.add_argument("--out_warp", type=str, default='../data/test_vol_01006_2mm_warp_field.nii.gz',
                        dest="out_warp", help="output warp file name")

    args = parser.parse_args()
    register(**vars(args))