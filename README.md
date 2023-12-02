

# mini-nnU-Net

Try to implement nnU-Net with less automatic but more simple.
My aim is to handle CT data in a 3D network. So no other modalities or 2D data will be supported.

## Preprocessing

Unlike array-type data such as ImageNet, 3D medical CT data has unique properties, including intensity values (also known as HU) and spacing. Different cases have varying spacing, but convolution works best with isotropic data, so you may need to resample data to a consistent spacing before feeding it into a network. Preprocessing involves three Python programs.

- *cropping*.py is used for cropping the foreground from img. But actually I find it does nothing for CT img, so you can just think it transforms nii.gz to npy data.
- *DataAnalyzer.py* is used for analyzing img. It will generate two .pkl files. One is dataset_properties.pkl, and the other is intensityproperties.pkl. Generally, it stores some intensity info, like mean intensity, median intensity...
- *preprocessing.py* is used for resampling. 

All these preprocessing steps are almost similar to nnU-Net. However, they are more straightforward to read and can be run individually.

## Network

- Implement a simple U-Net Network w/ and w/o ResBlock.

## Training 

`python training.py --gpus 0 --batch_size 2`

## Acknowledgement
Thanks for https://github.com/JunMa11/SegWithDistMap

