## Instructions for setup:

### Packages:

1. To setup Jetson Nano, we can use Jetpack or download the SD card image from 

    https://developer.nvidia.com/embedded/downloads


> Please also follow the link
> https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html for any issues with TensorRT setup.
    

By default, TensorRT would be installed with the SD card image.

> There is another option to manually flash Jetson Nano through shell scripts. Please find the individual BSP components at 
>   https://developer.nvidia.com/embedded/linux-tegra
>   FYI, It is possible to download and setup the Jetson system via shell scripts. As part of my job, I have setup the BSP and CUDA,CuDNN,TensorRT frameworks via scripts.
>   However, this option is only if Jetpack cannot be accessed or if we need to customize the filesystem and BSP.


2. Please also install 

    pip install opencv-contrib-python  This will allow cv.imshow to popup the image window.
    
    pip install efficientnet_pytorch

#### For faster deployment, I choose to use the SD card option.    

### Jetson Nano 4GB setup:

In Nvidia Jetson Nano, the RAM memory might not be sufficient in converting ONNX to TensorRT engine.

Please increase the swap space before starting the notebooks. I have allocated 20GB of swap space just in case.

```
sudo systemctl disable nvzramconfig
sudo fallocate -l 20G /mnt/20GB.swap
sudo mkswap /mnt/20GB.swap
sudo swapon /mnt/20GB.swap
```

This is sufficient for Jetson Nano Developer kit, 4GB version.

https://developer.nvidia.com/embedded/jetson-nano-developer-kit


Incase, if you would like to stop the desktop, please do

```
$ sudo init 3     # stop the desktop
$ sudo init 5     # restart the desktop
```

https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md

If you have setup root access in Jetson Nano, please also clear the Cache memory with 

```
echo 3 >/proc/sys/vm/drop_caches
```


### Jupyter Notebook setup:

There might be issues when starting Jupyter notebooks.

1. Please add **export OPENBLAS_CORETYPE=ARMV8** to .bashrc
2. Before starting Jupyter Notebook, please do

    **export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1**
3. To access the Jupyter notebook over network on a PC, please type

    **jupyter notebook --no-browser --port=8080**

4. On the PC, please type

    **ssh -L 8080:localhost:8080 user@x.x.x.x**

5. Please copy the link that is displayed on Jetson Nano console to the PC's browser.
For eg) http://localhost:8080/?token=15711ff3afa3706fb6e80a60d897614b85ba79d1818821f5

Now we are set to run the notebooks on Jetson Nano.
