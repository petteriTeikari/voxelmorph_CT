"""
Example script to register two volumes with VoxelMorph models

Please make sure to use trained models appropriately. 
Let's say we have a model trained to register subject (moving) to atlas (fixed)
One could run:

python register.py --gpu 0 /path/to/test_vol.nii.gz /path/to/atlas_norm.nii.gz --out_img /path/to/out.nii.gz --model_file ../models/cvpr2018_vm2_cc.h5 
"""

# py imports
import os
import glob
import sys
from argparse import ArgumentParser

def batch_register(gpu_id, moving_dir, fixed, model_file, out_dir, out_dir_warp):

    this_path = os.path.realpath(__file__)
    this_folder = os.path.split(this_path)[0]

    wildcard_search = os.path.join(moving_dir, '*.nii*')
    files_to_register = sorted(glob.glob(wildcard_search))
    print('{} Nifti files found to regiser, from input path "{}"'.format(len(files_to_register), moving_dir))

    #output_nifti_dir = os.path.join(moving_dir, '..', '..', '..', 'voxelmorph-rigid_2mm_128vx-3D')
    #output_warp_dir = os.path.join(moving_dir, '..', '..', '..', 'voxelmorph-rigid_2mm_128vx-3D_warpfield')
    if out_dir == None:
        out_dir = os.path.join(this_folder, '..', 'registered', 'out-nonrigid-vol')
    if out_dir_warp == None:
        out_dir_warp = os.path.join(this_folder, '..', 'registered', 'out-nonrigid-vol_warp')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_dir_warp):
        os.makedirs(out_dir_warp)

    # python register.py --gpu 0 ../data/test_vol_01006_2mm.nii.gz ../data/JohnHopkins_CQ500_template_2.0mm_norm.nii.gz --out_img ../data/test_vol_01006_2mm_registered.nii.gz --model_file ../models/sCROMIS2ICH_2mm_128vx.h5 --out_warp ../data/test_vol_01006_2mm_warp_field.nii.gz

    for i, filepath in enumerate(files_to_register):
        # https://janakiev.com/blog/python-shell-commands/
        dir = os.path.split(filepath)[0]
        fname = os.path.split(filepath)[1]
        print('Processing {}'.format(fname))
        cmd_arguments = '{} {} --out_img {} --model_file {}  --out_warp {}'.format(filepath, fixed, os.path.join(out_dir, fname.replace('nii.gz', '_voxelmorph.nii.gz')),
                                                                                   model_file, os.path.join(out_dir_warp, fname.replace('nii.gz', '_voxelmorph-warp.nii.gz')))
        cmd_string = 'python register.py --gpu ' + str(gpu_id) + ' ' + cmd_arguments
        stream = os.popen(cmd_string)
        output = stream.read()


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # positional arguments
    # python batch_register.py /home/petteri/Dropbox/manuscriptDrafts/CThemorr/DATA_DVC/sCROMIS2ICH/CT/labeled/MNI_2mm_128vx-3D/data/BM4D_brainWeighed_nonNaN_-100 ../data/JohnHopkins_CQ500_template_2.0mm_norm.nii.gz
    parser.add_argument("moving_dir", type=str, default='/home/petteri/Dropbox/manuscriptDrafts/CThemorr/DATA_DVC/sCROMIS2ICH/CT/labeled/MNI_2mm_128vx-3D/data/BM4D_brainWeighed_nonNaN_-100',
                        help="moving file name")
    parser.add_argument("fixed", type=str, default='../data/JohnHopkins_CQ500_template_2.0mm_norm.nii.gz',
                        help="fixed file name")
    parser.add_argument("--model_file", type=str,
                        dest="model_file", default='../models/sCROMIS2ICH_2mm_128vx.h5',
                        help="models h5 file")
    parser.add_argument("--gpu", type=int, default=None,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--out_dir", type=str, default=None,
                        dest="out_dir", help="output image file name")
    parser.add_argument("--out_dir_warp", type=str, default=None,
                        dest="out_dir_warp", help="output warp file name")

    args = parser.parse_args()
    batch_register(**vars(args))