# CavDet
Due to active galaxy nuclei (AGN) mechanism at the center of the galaxies, electronics of high energy are ejected to blow and push the gas around. Bubbles or cavities are then generated, which can be detected at multiple frequency bands, especially for the X-ray band. 

Since AGN reveals quite lots of attracting physical phenomenons, detecting of them is significant. However, there exist many difficulties in our works. For instance, the background and system noise in the X-ray images, which lead to low signal-to-noise ratio (SNR), should be eliminated. In addition, the high brightness (temperature) in the galaxy center usually leads to low contrast of the cavity regions. 

## Methods
In this repo, we provide a toolbox namely `cav_gcnn` to detect cavities in the X-ray astronomical images observed by the [*Chandra* X-ray observatory (CXO)](http://cxc.harvard.edu/). Our method is designed based on the state-of-the-art convolutional neural network (CNN), as well as a strategy to handle the imbalanced dataset.

## Usage
To utilize our toolbox to cavity detections, a granular cnn model should be trained in advance and saved. Then, cavities in the new observations can be detected and marked with elliptical markers after prepocessing on the raw image data. In this work, 40 observations of 40 different objects were applied to train our GCNN model, and a [snap](https://github.com/myinxd/cav_gcnn/blob/master/samples/samples.png) of them is illustrated.

If you want to see the details of preprocessing and usage of our script, please refer to our paper<TODO> and the python codes. And the installation of the toolbox is as follows,

```sh
$ cd cav_gcnn
$ <sudo> pip install <--user> . 
```

## Requirements
To process our scripts, some python packages are required, which are listed as follows.

- [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/)
- [astropy](http://docs.astropy.org/en/stable/), [astroquery](http://astroquery.readthedocs.io/en/latest/)
- [Theano](http://www.deeplearning.net/software/theano/) 
- [Lasagne](http://lasagne.readthedocs.io/en/latest/) 
- [nolearn](http://pythonhosted.org/nolearn/lasagne.html)

The `requirements.txt` is provided in this repository, by which the required packages can be installed easily. We advice the users to configure these packages in a virtual environment.

- initialize env
```sh
$ <sudo> pip install virtualenv
$ cd cav_gcnn
$ virtualenv ./env
```
- install required packages
```sh
$ cd cav_gcnn
$ env/bin/pip install -r ./requirements.txt
``` 
- install the latest Theano, and Lasagne
```sh
$ git clone https://github.com/Lasagne/Lasagne.git
$ cd Lasagne
$ <sudo> pip install <--user> -e .
```
```sh
$ git clone git://github.com/Theano/Theano.git
$ cd Theano
$ <sudo> pip install <--user> <--no-deps> -e .
```
In addition, the computation can be accelerated by parallelly processing with GPUs. In this work, our scripts are written under the guide of Nvidia CUDA, thus the Nvidia GPU hardware is also required.

- CUDA  
  https://developer.nvidia.com/cuda-downloads


## References
- [Theano tutorial](http://www.deeplearning.net/software/theano/)
- [Lasagne tutorial](http://lasagne.readthedocs.io/en/latest/user/tutorial.html)
- [Save python data by pickle](http://www.cnblogs.com/pzxbc/archive/2012/03/18/2404715.html)
- [astroquery.Ned](http://astroquery.readthedocs.io/en/latest/ned/ned.html)

## Author
- Zhixian MA <`zxma_sjtu(at)qq.com`>

## License
Unless otherwise declared:

- Codes developed are distributed under the [MIT license](https://opensource.org/licenses/mit-license.php);
- Documentations and products generated are distributed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US);
- Third-party codes and products used are distributed under their own licenses.

## Citation
<TODO>

