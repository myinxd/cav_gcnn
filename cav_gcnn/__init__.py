"""
Elliptical Gaussian filters to detect point sources in X-ray astronomical images.

Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
MIT license
"""

# Arguments for setup()
__pkgname__ = "cav_gcnn"
__version__ = "0.0.1"
__author__ = "Zhixian MA"
__author_email__ = "zxma_sjtu@qq.com"
__license__ = "MIT"
__keywords__ = "cavity detection in X-ray astronomical images"
__copyright = "Copyright (C) 2017 Zhixian MA"
__url__ = "https://github.com/myinxd/cav_gcnn"
__description__ = ("A toolbox to detect cavities in X-ray astronomical images with granular cnn.")

# Set logging handle
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
