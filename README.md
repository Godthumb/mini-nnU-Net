# mini-nnU-Net
try to implement nnU-Net with less automatic but more simple
My aiming is to handle CT data in 3D network.so no more other modalities or 2D data will be supported.
now I just read a little of nnU-Net code.I will just focus on preprocessing now.
The first phase, cropping.py is used for cropping foreground from img.But actually i find it not do anything for CT img, so you can just think it transform nii.gz to npy data.
The second phase, DataAnalyzer.py is used for analysising img.it will generate two .pkl file.one is dataset_properties.pkl, other one is intensityproperties.pkl.what in these file, you can load and check them.
