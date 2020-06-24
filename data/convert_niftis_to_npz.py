##

import os
import glob
sCROMIS2ICH_base = '/home/petteri/Dropbox/manuscriptDrafts/CThemorr/DATA_DVC/sCROMIS2ICH/'
CT_data_dir = os.path.join(sCROMIS2ICH_base, 'CT/labeled/MNI_1mm_256vx-3D/data/BM4D_brainWeighed_nonNaN_-100')
#label_dir = os.path.join(sCROMIS2ICH_base, 'CT/labeled/MNI_1mm_256vx-3D/labels/voxel/hematomaAll')
label_dir = os.path.join(sCROMIS2ICH_base, 'CT/labeled/MNI_1mm_256vx-3D/labels/voxel/brain') 
# TODO! go through all the labels in more organized manner 
wildcard_search = os.path.join(CT_data_dir, '*.nii*')
files = sorted(glob.glob(wildcard_search))
print('{} Nifti files found from input path "{}"'.format(len(files), CT_data_dir))

wildcard_search = os.path.join(label_dir, '*.nii*')
labels = sorted(glob.glob(wildcard_search))

output_path = CT_data_dir + '_npz'
if not os.path.exists(output_path):
    os.makedirs(output_path)
print('Saving npz files to "{}"'.format(output_path))

downsampled_dir = os.path.join(sCROMIS2ICH_base, 'CT/labeled/MNI_2mm_128vx-3D/data/BM4D_brainWeighed_nonNaN_-100')
if not os.path.exists(downsampled_dir):
    os.makedirs(downsampled_dir)

# downsampled_label_dir = os.path.join(sCROMIS2ICH_base, 'CT/labeled/MNI_2mm_128vx-3D/labels/voxel/hematomaAll')
downsampled_label_dir = os.path.join(sCROMIS2ICH_base, 'CT/labeled/MNI_2mm_128vx-3D/labels/voxel/brain')
if not os.path.exists(downsampled_label_dir):
    os.makedirs(downsampled_label_dir)

output_path2 = downsampled_dir + '_npz'
if not os.path.exists(output_path2):
    os.makedirs(output_path2)
print('And the volumes "debug resolution" 2.00 mm^3  to "{}"'.format(output_path2))

# the template was clipped to these
clip_range = [0, 100] # HU
#print('Clip the NIFTIs to range [{}, [}]'.format(clip_range[0], clip_range[1]))
#print('Normalize the clipped Niftis to range [0, 1] as we use normalized templates (assume this to be the logic?)')

##
import nibabel as nib
import numpy as np
for i, file in enumerate(files):

    dir = os.path.split(file)[0]
    fname = os.path.split(file)[1]
    img = nib.load(file)
    data = img.get_fdata()
    data[data < clip_range[0]] = clip_range[0]
    data[data > clip_range[1]] = clip_range[1]
    data = data / np.max(data)
    fname_out = fname.replace('.nii.gz', '.npz')
    path_out = os.path.join(output_path, fname_out)
    print('Saving filename "{}"'.format(fname_out))
    np.savez(path_out, vol_data=data)

    # downsample to 2.0 mm^3 isotropic
    integer_downsample_factor = 2
    fname_out = fname.replace('MNI_1mm_256vx', 'MNI_2mm_128vx')
    downsampled = img.slicer[::integer_downsample_factor, ::integer_downsample_factor, ::integer_downsample_factor]
    path_out_ds_niigz = os.path.join(downsampled_dir, fname_out)
    nib.save(downsampled, path_out_ds_niigz)

    # NPZ
    data_downsampled = downsampled.get_fdata()
    data_downsampled[data_downsampled < clip_range[0]] = clip_range[0]
    data_downsampled[data_downsampled > clip_range[1]] = clip_range[1]
    data_downsampled = data_downsampled / np.max(data_downsampled)
    path_out2 = os.path.join(output_path2, fname_out)
    np.savez(path_out2, vol_data=data_downsampled)

    # Downsample label as well
    img_label = nib.load(labels[i])
    fname = os.path.split(labels[i])[1]
    fname_out = fname.replace('MNI_1mm_256vx', 'MNI_2mm_128vx')
    downsampled_label = img_label.slicer[::integer_downsample_factor, ::integer_downsample_factor, ::integer_downsample_factor]
    path_out_ds_niigz = os.path.join(downsampled_label_dir, fname_out)
    nib.save(downsampled_label, path_out_ds_niigz)
