import os
dir_path = os.path.dirname(os.path.realpath(__file__))
template_path = os.path.join(dir_path, 'JohnHopkins_CQ500_template_0.5mm.nii.gz')
integer_downsample_factor = 2 # 0.5 mm^3 -> 1.0 mm^3
res_out = 0.5 * integer_downsample_factor
# outpath = os.path.join(dir_path, 'template' + str(res_out) + 'mm.nii.gz')
outpath = os.path.join(dir_path, 'JohnHopkins_CQ500_template_1.0mm.nii.gz')
outpath_npz = os.path.join(dir_path, 'JohnHopkins_CQ500_template_1.0mm.npz')
outpath_norm_npz = os.path.join(dir_path, 'JohnHopkins_CQ500_template_1.0mm_norm.npz')

outpath2 = os.path.join(dir_path, 'JohnHopkins_CQ500_template_2.0mm.nii.gz')
outpath2_norm = os.path.join(dir_path, 'JohnHopkins_CQ500_template_2.0mm_norm.nii.gz')
outpath2_npz = os.path.join(dir_path, 'JohnHopkins_CQ500_template_2.0mm.npz')
outpath2_norm_npz = os.path.join(dir_path, 'JohnHopkins_CQ500_template_2.0mm_norm.npz')

# https://www.kaggle.com/mechaman/resizing-reshaping-and-resampling-nifti-files
# https://github.com/nipy/nibabel/issues/670
# https://nipy.org/nibabel/nibabel_images.html
import nibabel as nib

# Import
img = nib.load(template_path)
print('Input shape (vx) = ',  img.shape)
print('Input dims (mm) = ',  img.header.get_zooms())

# Downsample to 1.0 mm³
downsampled = img.slicer[::integer_downsample_factor, ::integer_downsample_factor, ::integer_downsample_factor]
print('Downsampled shape (vx) = ',  downsampled.shape)
print('Downsampled dims (mm) = ',  downsampled.header.get_zooms())

# Export the Nifti
# https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
nib.save(downsampled, outpath)

# Voxelmorph actually uses the npz, so let's save that as well
#     :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
#      atlas_vol = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis] (The atlas we used is 160x192x224.)
import numpy as np
img_data = downsampled.get_fdata()
np.savez(outpath_npz, vol=img_data)

# Normalize as well
img_data_norm = img_data / np.amax(img_data)
np.savez(outpath_norm_npz, vol=img_data_norm)

# Downsample to 2.0 mm³ as well (mainly for debugging stuff as the 1mm^2 still can lead to issues with you not having enough memory)
downsampled2 = downsampled.slicer[::integer_downsample_factor, ::integer_downsample_factor, ::integer_downsample_factor]
print('Downsampled shape (vx) = ',  downsampled2.shape)
print('Downsampled dims (mm) = ',  downsampled2.header.get_zooms())

# Export the Nifti
# https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
nib.save(downsampled2, outpath2)

img_data2 = downsampled2.get_fdata()
np.savez(outpath2_npz, vol=img_data2)

# Normalize as well
img_data2_norm = img_data2 / np.amax(img_data2)
np.savez(outpath2_norm_npz, vol=img_data2_norm)

# and save the normalized, as nifti as well
norm_img = nib.Nifti1Image(img_data2_norm, downsampled2.affine, downsampled2.header)
nib.save(norm_img, outpath2_norm)