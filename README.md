# mini-nnU-Net
Try to implement nnU-Net with less automatic but more simple.
My aiming is to handle CT data in 3D network.so no more other modalities or 2D data will be supported.
# preprocessing
Different from array type data(like imagenet...), 3D medical CT data has special properties.Two important of them, is intensity and spacing.
Intensity is also known as HU.And spacing means how much millimetre in one voxel.Different case has different spacing,but for convolution, it is good at
handling isotropic data,so you may need to resampling data into same spacing before you feed them to network.
Here for preprocessing, has three python programme.
* cropping.py is used for cropping foreground from img.But actually i find it not do anything for CT img, so you can just think it transform nii.gz to npy data.
The second phase, DataAnalyzer.py is used for analysising img.it will generate two .pkl file.one is dataset_properties.pkl, other one is intensityproperties.pkl.Generally,it store some intensity data, like mean intensity, median intensity...
The third phase, preprocessing.py is used for resampling.
All these preprocessing steps almost similar with nnU-Net.But it more easy to read, and you can run these one by one.

