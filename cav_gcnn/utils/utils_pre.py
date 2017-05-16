# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qqq.com>

"""
A script to preprocess the raw data fetched from Chandra Data Archive (CDA)

The processing steps are as follows
1. Get field of view (FOV) from the fov1.fits file, the CCD_ID is set as 7 as the default
2. Get evt2_sub.fits by dmcopying from the evt2.fits according to the fov_sub.fits
3. Detect point sources by wavdetect, and save the region files
4. Mannually filter the point sources
5. Fill the point sources with dmfilth
6. Locate center of the galaxy or cluster, output as a region file
7. Get the 400x400 cut image from the evt2_sub.fits

References
==========
1. An image of diffuse emission
   http://cxc.cfa.harvard.edu/ciao/threads/diffuse_emission/

Methods
=======
get_sub(obspath,ccd_id=7)
    get the single ccd image
get_ps(obspath,pspath='wavd')
    detect point sources with wavdetect
fill_ps(obspath,pspath='wavd')
    fill the ps by dmfilth
get_cnt(obspath)
    locate the center point
get_img(obspath)
    get the cut image

# Caution
1. CIAO software package should be pre-installed in your system
2. Python3 packages like astropy, numpy, scipy should be installed.
"""

import os
# from astropy.io import fits
import numpy as np


def get_sub(obspath, ccd_id='7'):
    """
    Get the single ccd image

    Variables
    =========
    obspath: str
        path of the observation
    ccd_id: str
        ID of the ccd, default as 7

    """
    # Init
    evt_path = os.path.join(obspath, 'evt2.fits')
    fov_path = os.path.join(obspath, 'fov1.fits')
    fov_sub_path = os.path.join(obspath, 'fov1_sub.fits')
    evt_sub_path = os.path.join(obspath, 'evt2_sub.fits')

    # Get cut
    print("Processing on sample %s" % obspath)
    # get fov
    print("dmcopy '%s[ccd_id=%s]' %s clobber=yes" %
          (fov_path, ccd_id, fov_sub_path))
    os.system("dmcopy '%s[ccd_id=%s]' %s clobber=yes" %
              (fov_path, ccd_id, fov_sub_path))
    # get sub evt2
    print("dmcopy '%s[energy=500:7000,ccd_id=%s,sky=region(%s)][bin sky=1]' %s clobber=yes" % (
        evt_path, ccd_id, fov_sub_path, evt_sub_path))
    os.system("dmcopy '%s[energy=500:7000,ccd_id=%s,sky=region(%s)][bin sky=1]' %s clobber=yes" % (
        evt_path, ccd_id, fov_sub_path, evt_sub_path))


def get_ps(obspath, pspath="wavd", evtname='evt2_sub.fits'):
    """
    detect point sources with wavdetect

    Variables
    =========
    obspath: str
        path of the observation
    pspath: str
        path of the poins sources
    """
    # Init
    evt_path = os.path.join(obspath, evtname)
    pspre = os.path.join(obspath, pspath)

    if not os.path.exists(pspre):
        os.mkdir(pspre)
    # path of wavd results
    psffile = os.path.join(pspre, 'psf.psfmap')
    regfile = os.path.join(pspre, 'wavd.reg')
    outfile = os.path.join(pspre, 'wavd.fits')
    scellfile = os.path.join(pspre, 'wavd.scell')
    imagefile = os.path.join(pspre, 'wavd.image')
    defnbkgfile = os.path.join(pspre, 'wavd.nbkg')

    # Get psfmap
    print("punlearn mkpsfmap")
    os.system("punlearn mkpsfmap")
    print("mkpsfmap %s %s energy=1.4967 ecf=0.393" % (evt_path, psffile))
    os.system("mkpsfmap %s %s energy=1.4967 ecf=0.393 clobber=yes" %
              (evt_path, psffile))

    # get point sources
    print("punlearn wavdetect")
    os.system("punlearn wavdetect")
    print("wavdetect infile=%s regfile=%s outfile=%s scellfile=%s imagefile=%s defnbkgfile=%s psffile=%s scales='2.0 4.0' clobber=yes" % (
        evt_path, regfile, outfile, scellfile, imagefile, defnbkgfile, psffile))
    os.system("wavdetect infile=%s regfile=%s outfile=%s scellfile=%s imagefile=%s defnbkgfile=%s psffile=%s scales='2.0 4.0' clobber=yes" % (
        evt_path, regfile, outfile, scellfile, imagefile, defnbkgfile, psffile))


def fill_ps(obspath, pspath='wavd', evtname='evt2_sub.fits', fillname='img_fill.fits', regname='wavd.reg'):
    """
    Fill regions of point sources in the fits image.

    Variables
    =========
    obspath: str
        path of the observation
    pspath: str
        path of the point sources

    """
    # Init
    evt_path = os.path.join(obspath, evtname)
    pspre = os.path.join(obspath, pspath)
    regfile = os.path.join(pspre, regname)
    reg_mod = os.path.join(pspre, 'wavd_mod.fits')
    roi_path = os.path.join(pspre, 'sources')

    if not os.path.exists(roi_path):
        os.mkdir(roi_path)

    # parameters
    roi_outsrcfile = os.path.join(roi_path, "src%d.fits")
    # parameters of dmfilth
    # exclude = os.path.join(roi_path, "exclude")
    outfile = os.path.join(obspath, fillname)

    # get region
    print("punlearn dmmakereg")
    os.system("punlearn dmmakereg")
    print("dmmakereg 'region(%s)' %s clobber=yes" % (regfile, reg_mod))
    os.system("dmmakereg 'region(%s)' %s clobber=yes" % (regfile, reg_mod))

    # get roi
    print("punlearn roi")
    os.system("punlearn roi")
    print("roi infile=%s outsrcfile=%s bkgfactor=0.5 fovregion="" streakregion="" radiusmode=mul bkgradius=3 clobber=yes" % (
        reg_mod, roi_outsrcfile))
    os.system("roi infile=%s outsrcfile=%s bkgfactor=0.5 fovregion="" streakregion="" radiusmode=mul bkgradius=3 clobber=yes" % (
        reg_mod, roi_outsrcfile))

    # split bkg and ps regions
    print("splitroi '%s' exclude" % (os.path.join(roi_path, "src*.fits")))
    os.system("splitroi '%s' exclude" % (os.path.join(roi_path, "src*.fits")))

    # fill
    print("punlearn dmfilth")
    os.system("punlearn dmfilth")
    print("dmfilth infile=%s outfile=%s method=POISSON srclist=@exclude.src.reg bkglist=@exclude.bg.reg randseed=0 clobber=yes" % (evt_path, outfile))
    os.system("dmfilth infile=%s outfile=%s method=POISSON srclist=@exclude.src.reg bkglist=@exclude.bg.reg randseed=0 clobber=yes" % (
        evt_path, outfile))


def get_cnt(obspath, box_size=400, imgname='img_fill.fits', cntname='cnt.reg'):
    """Detect center of the observation

    Variable
    ========
    obspath: str
        path of the observation
    box_size: integer
        size of the box
    """
    # Init
    img_path = os.path.join(obspath, imgname)
    cnt_path = os.path.join(obspath, cntname)

    # Load the image
    hdulist = fits.open(img_path)
    img = hdulist[0]
    rows, cols = img.shape
    # Get sky coordinate
    sky_x = img.header['CRVAL1P']
    sky_y = img.header['CRVAL2P']
    # find peak
    peak = np.max(img.data)
    peak_row, peak_col = np.where(img.data == peak)
    peak_row = np.mean(peak_row)
    peak_col = np.mean(peak_col)
    # Judege overflow
    row_up = max((peak_row - box_size, 0))
    row_down = min((peak_row + box_size, rows))
    col_left = max((peak_col - box_size, 0))
    col_right = min((peak_col + box_size, cols))
    # region parameters
    cnt_x = peak_col + 1 + sky_x
    cnt_y = peak_row + 1 + sky_y
    box_row = min((peak_row - row_up, row_down - peak_row))
    box_col = min((peak_col - col_left, col_right - peak_col))
    box_size = np.floor(min((box_row, box_col))) * 2
    # write region
    if os.path.exists(cnt_path):
        os.remove(cnt_path)
    fp = open(cnt_path, 'a')
    fp.write("box(%f,%f,%d,%d,0)" % (cnt_x, cnt_y, box_size, box_size))


def get_img(obspath, cntpath='cnt.reg', evtname='img_fill.fits', cutname='img_cut.fits'):
    """
    Get the center image

    Variables
    =========
    obspath: str
        path of the observation
    cntpath: str
        path of the center region

    """
    # Init
    evt_path = os.path.join(obspath, evtname)
    cnt_path = os.path.join(obspath, cntpath)
    img_path = os.path.join(obspath, cutname)

    # cut
    print("punlearn dmcopy")
    os.system("punlearn dmcopy")
    print("dmcopy '%s[sky=region(%s)]' %s clobber=yes" %
          (evt_path, cnt_path, img_path))
    os.system("dmcopy '%s[sky=region(%s)]' %s clobber=yes" %
              (evt_path, cnt_path, img_path))


def cmp_cav(obspath, cirname='cav_cir.reg', refname='cavities_chg.reg', cntname='cnt.reg'):
    """
    Draw circles on the image according to the cavity parameters in Shin2016

    Variable
    ========
    obspath: str
        path of the observation
    """
    # Init
    cir_path = os.path.join(obspath, cirname)
    ref_path = os.path.join(obspath, refname)
    cnt_path = os.path.join(obspath, cntname)

    # get center coordinate
    fp_cnt = open(cnt_path, 'r')
    cnt = fp_cnt.readline()
    cnt = cnt[4:-1]
    cnt = cnt.split(",")
    fp_cnt.close()

    # Write circles
    if os.path.exists(cir_path):
        os.remove(cir_path)
    fp_cir = open(cir_path, 'a')
    fp_ref = open(ref_path, 'r')

    cavs = fp_ref.readlines()
    for l in cavs:
        l1 = l.replace("\n", '')
        params = l1.split('\t')
        fp_cir.write("circle(%s,%s,%s)\n" % (cnt[0], cnt[1], params[-1]))


def get_smooth(obspath, sigma=10, imgname='img_cut.fits', smoothname='img_smooth.fits'):
    """
    Get smoothed image.
    """
    imgpath = os.path.join(obspath, imgname)
    smoothpath = os.path.join(obspath, smoothname)

    # kernel
    kernel = ('lib:gaus(2,5,1,%d,%d)' % (sigma, sigma))

    # smooth
    print("aconvolve %s %s '%s' clobber=yes" % (imgpath, smoothpath, kernel))
    os.system("aconvolve %s %s '%s' clobber=yes" %
              (imgpath, smoothpath, kernel))
