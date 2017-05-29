# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Fetch observations from chandra

Reference
--------
    [1] http://cxc.cfa.harvard.edu/ciao/ahelp/download_chandra_obsid.html
    [2] http://cxc.cfa.harvard.edu/ciao/ahelp/find_chandra_obsid.html

"""

import os
import pandas as pd


def main():
    """
    Fetch observations from chandra
    """
    # Init
    objname = 'objlist.txt'
    obsidname = 'obsid.txt'
    csv_folder = 'csv'
    # path_all = 'AllSamples'

    # handles
    f = open(objname, 'r')
    fo = open(obsidname, 'w+')

    if not os.path.exists(csv_folder):
        os.mkdir(csv_folder)
    for sample in f:
        sample = sample[:-1]
        sample = sample.replace(' ', '_')
        print("Sample name: %s" % sample)
        # Get obsid table
        csv_name = ("%s.csv") % sample
        csv_path = os.path.join(csv_folder, csv_name)
        print("find_chandra_obsid %s > %s" % (sample, csv_path))
        os.system("find_chandra_obsid %s > %s" % (sample, csv_path))
        # get obsid
        print("dmlist '%s[cols obsid]' data,raw > temp.csv" % (csv_path))
        os.system("dmlist '%s[cols obsid]' data,raw > temp.csv" % (csv_path))
        sp_table = pd.read_csv('temp.csv')
        obs_line = sp_table.values[-1, 0]
        obsid = str(int(obs_line))
        fo.write('%s\t' % obsid)
        # download all files
        print("download_chandra_obsid %s" % obsid)
        os.system("download_chandra_obsid %s" % obsid)
        # repro
        print("chandra_repro %s/. %s/repro" % (obsid, obsid))
        os.system("chandra_repro %s/. %s/repro" % (obsid, obsid))
        # change folder name
        print("mv %s %s" % (obsid, sample))
        os.system("mv %s %s" % (obsid, sample))
    
    # close of files
    f.close()
    fo.close()

if __name__ == "__main__":
    main()
