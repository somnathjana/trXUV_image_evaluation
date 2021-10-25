# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:30:18 2021

@author: jana
"""
# import packages
import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic as bs
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import affine_transform

# helper functions-------------------------------------------------------------
def kata_samay(tic, toc):
    time_taken = (toc-tic)
    hrs = int(time_taken//3600)
    mnts = int((time_taken-3600*hrs)//60)
    secs = round(time_taken-hrs*3600-mnts*60, 2)
    print('Taking {}:{}:{}'.format(hrs, mnts, secs))
    
# End of helper functions------------------------------------------------------

# functions to save image files to .h5py file----------------------------------
def keys2Remove():
    return ['Pt_No', 'asym1', 'asym10', 'asym2', 'asym3', 'asym4', 'asym5', 'asym6', 'asym7', 'asym8', 'asym9', 
                'dt', 'epoch', 'pre_scan_snapshot', 'ref', 'ref1', 'ref10', 'ref10M', 'ref1M', 'ref2', 'ref2M', 'ref3',
                'ref3M', 'ref4', 'ref4M', 'ref5', 'ref5M', 'ref6', 'ref6M', 'ref7', 'ref7M', 'ref8', 'ref8M', 'ref9',
                'ref9M', 'refM', 'rel', 'rel1', 'rel10', 'rel10M', 'rel1M', 'rel2', 'rel2M', 'rel3', 'rel3M', 'rel4',
                'rel4M', 'rel5', 'rel5M', 'rel6', 'rel6M', 'rel7', 'rel7M', 'rel8', 'rel8M', 'rel9', 'rel9M', 'relM',
                'spec', 'spec1', 'spec10', 'spec10M', 'spec1M', 'spec2', 'spec2M', 'spec3', 'spec3M', 'spec4', 'spec4M',
                'spec5', 'spec5M', 'spec6', 'spec6M', 'spec7', 'spec7M', 'spec8', 'spec8M', 'spec9', 'spec9M', 'specM']

def LoadToHDF(root, fileName, h5FilePath, h5FileName, fPathToSave):
    '''
    ''' 
    tic = time.time()
    with h5py.File(h5FilePath+h5FileName,'r') as h5Data:   # open .h5 data file
    
        with h5py.File(fPathToSave, 'w') as hf:
            scan_list = []
            for entry in os.scandir(root):
                scan_list.append(entry.name)
                scan_no = entry.name
                grp = hf.create_group(scan_no)
                
                key_list = h5Data['entry'+scan_no]['measurement'].keys()
                keysToRemove = keys2Remove()
                motors = [keys for keys in key_list if keys not in keysToRemove]
                grp.attrs['motors'] = motors
                for motor in motors:
                    grp.attrs[motor] = h5Data['entry'+scan_no]['measurement'][motor]
                
                filePath = root+entry.name+'/'
                for _, _, files in os.walk(filePath):
                    print('storing scan_no ', scan_no, ', number of files = ', len(files))
                    for file in files:
                        load_data = np.fromfile(filePath+file, dtype=np.double, sep=' ')
                        data = np.reshape(load_data, (255,1024))
                        grp.create_dataset(file, data=data, compression="gzip")
                    
    toc = time.time()
    time_taken = (toc-tic)
    hrs = int(time_taken//3600)
    mnts = int((time_taken-3600*hrs)//60)
    secs = round(time_taken-hrs*3600-mnts*60, 2)
    print('Taking {}:{}:{}'.format(hrs, mnts, secs))
    
def ImageToHDF5(root, fileName, h5FilePath='', h5FileName='', pathToSave=None):
    '''
    '''
    
    if pathToSave == None:
        pathToSave = root
        
    fPathToSave = pathToSave+fileName+'_image.h5py'
    if os.path.exists(fPathToSave):
        print("File "+fPathToSave+" already exists.")
        cond = ''
        while (cond not in ['y', 'yes', 'n', 'no']):
            cond = input("Do you want to rewrite the file (y/n)?")
            cond = cond.lower()
    else:
        cond = 'y'
    if cond in ['y', 'yes']:
        LoadToHDF(root, fileName, h5FilePath, h5FileName, fPathToSave)

# End of functions to save image files to .h5py file---------------------------
    
# functions to EvaluateEmage---------------------------------------------------
def data_im(self):
    dd = {}
    for i, sn in enumerate(self.scan_list):
        pt_list = list(self.hf[sn].keys())
        dd[sn] = {}
        for j, pn in enumerate(pt_list):
            dd[sn][pn] = self.hf[sn][pn]
    return dd

def crop_image(self, im_crop, sn, pn):
        assert len(im_crop)==2, "im_crop must contain 2 or no lists."
        #self.px_mid_spec = self.px_mid_spec-im_crop[0][2]
        #self.px_mid_ref = self.px_mid_ref-im_crop[1][2]
        if '.spec' in pn:   
            return self.hf[sn][pn][()][im_crop[0][0]:im_crop[0][1], im_crop[0][2]:im_crop[0][3]]
        else:
            return self.hf[sn][pn][()][im_crop[1][0]:im_crop[1][1], im_crop[1][2]:im_crop[1][3]]
        
def process_image(self, im_crop, im_shear, sn, pn):
        assert len(im_crop)==2, "im_crop must contain 2 or no lists."
        assert len(im_shear)==2, "im_shear must contain 2 or no elements."
        image_crop = crop_image(self, im_crop, sn, pn)
        if '.spec' in pn:
            return im_vshear(image_crop, im_shear[0])
        else:
            return im_vshear(image_crop, im_shear[1])
        
def rotate_image(self, im_rot, sn, pn):
        assert len(im_rot)==2, "im_rot must contain 2 or no elements."
        image = self.hf[sn][pn][()]
        if '.spec' in pn:
            return rotate(image, angle=im_rot[0])
        else:
            return rotate(image, angle=im_rot[1])

def im_vshear(image, shear):
        mat_shear = np.array([[1, 0.0],
                            [shear, 1],])
        (h, w) = image.shape
        ofs = int(h*shear)
        image_shear = affine_transform(image, mat_shear, offset=(0, -ofs), 
                                           output_shape=(h, w+int(h*shear)))
        return image_shear
        
def shear_image(self, im_shear, sn, pn):
        assert len(im_shear)==2, "im_shear must contain 2 or no elements."
        image = self.hf[sn][pn][()]
        if '.spec' in pn:
            shear =im_shear[0]
            return im_vshear(image, shear)
        else:
            shear =im_shear[1]
            return im_vshear(image, shear)
        
def FindPeaks(self, spectra, rel_height):
    height = np.max(spectra)/20
    px_mid = self.px_mid_spec
    peaks1 = find_peaks(spectra[:px_mid], height=height, distance=35, prominence=height/2)
    peaks2 = find_peaks(spectra[px_mid:], height=height, distance=25, prominence=height/2)
    peaks = np.concatenate((peaks1[0], peaks2[0]+(np.ones_like(peaks2[0])*px_mid).astype(int)))
    peak_heights = np.concatenate((peaks1[1]['peak_heights'], peaks2[1]['peak_heights']))
    PeakWidths = peak_widths(spectra, peaks, rel_height=rel_height)

    return peaks, peak_heights, PeakWidths

def FindPeaks_ref(self, spectra, rel_height):
    height = np.max(spectra)/25
    px_mid = self.px_mid_ref
    peaks1 = find_peaks(spectra[:px_mid], height=height, distance=30, prominence=height/2)
    peaks2 = find_peaks(spectra[px_mid:], height=height, distance=30, prominence=height/2)
    peaks = np.concatenate((peaks1[0], peaks2[0]+(np.ones_like(peaks2[0])*px_mid).astype(int)))
    peak_heights = np.concatenate((peaks1[1]['peak_heights'], peaks2[1]['peak_heights']))
    PeakWidths = peak_widths(spectra, peaks, rel_height=rel_height)

    return peaks, peak_heights, PeakWidths

def dataAuto(self, Data_arr, key_list, i, j, sn, n_Har):                        # not in use
    spectra = self.spectra[sn][key_list[0][j]]
    spectraM = self.spectra[sn][key_list[1][j]]
    spectra_ref = self.spectra[sn][key_list[2][j]]
    spectra_refM = self.spectra[sn][key_list[3][j]]
    
    peaks, _, PeakWidths = FindPeaks(self, spectra, rel_height=0.95)
    peaksM, _, PeakWidthsM = FindPeaks(self, spectraM, rel_height=0.95)
    peaks_ref, _, PeakWidths_ref = FindPeaks_ref(self, spectra_ref, rel_height=0.95)
    peaks_refM, _, PeakWidths_refM = FindPeaks_ref(self, spectra_refM, rel_height=0.95)
    
    assert len(peaks) == 17, print(sn, key_list[0][j])
    assert len(peaksM) == 17, print(sn, key_list[1][j])
    assert len(peaks_ref) == 17, print(sn, key_list[2][j])
    assert len(peaks_refM) == 17, print(sn, key_list[3][j])
    
    for k in range(n_Har):
        Data_arr[0, i, k, j] = np.sum(spectra[int(PeakWidths[2][k]):int(PeakWidths[3][k])])
        Data_arr[1, i, k, j] = np.sum(spectraM[int(PeakWidthsM[2][k]):int(PeakWidthsM[3][k])])
        Data_arr[2, i, k, j] = np.sum(spectra_ref[int(PeakWidths_ref[2][k]):int(PeakWidths_ref[3][k])])
        Data_arr[3, i, k, j] = np.sum(spectra_refM[int(PeakWidths_refM[2][k]):int(PeakWidths_refM[3][k])])
    return Data_arr

def show_mask(hf, sn, pn):
    spectra = hf.spectra[sn][pn[0]]
    spectra_ref = hf.spectra[sn][pn[1]]
    mask_l_spec = hf.mask['spec'][0]
    mask_r_spec = hf.mask['spec'][1]
    mask_l_ref = hf.mask['ref'][0]
    mask_r_ref = hf.mask['ref'][1]
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,4))
    ax1.plot(spectra)
    ax2.plot(spectra_ref)
    for i, ms in enumerate(mask_l_spec):
        ax1.axvline(ms, c='r', ls='--')
        ax1.axvline(mask_r_spec[i], c='g', ls='--')
    for i, mr in enumerate(mask_l_ref):
        ax2.axvline(mr, c='r', ls='--')
        ax2.axvline(mask_r_ref[i], c='g', ls='--')
    plt.show()

def find_mask(hf, sn, pn, rel_height=0.75):
    spectra = hf.spectra[sn][pn[0]]
    spectra_ref = hf.spectra[sn][pn[1]]
    peaks, _, peakWidths = FindPeaks(hf, spectra, rel_height=rel_height)
    peaks_ref, _, peakWidths_ref = FindPeaks_ref(hf, spectra_ref, rel_height=rel_height)
    mask_l_spec = np.rint(peakWidths[2])
    mask_r_spec = np.rint(peakWidths[3])
    mask_l_ref = np.rint(peakWidths_ref[2])
    mask_r_ref = np.rint(peakWidths_ref[3])
    
    hf.mask = {'spec':[mask_l_spec, mask_r_spec], 'ref':[mask_l_ref, mask_r_ref]}
    hf.peaks = {'spec': peaks, 'ref': peaks_ref}
    show_mask(hf, sn, pn)

def create_bins(bins):
    bins_diff = np.diff(bins)
    leftpad = np.array([bins[0]-bins_diff[0]])
    rightpad = np.array([bins[-1]+bins_diff[-1]])
    bins_pad = np.concatenate((leftpad, bins, rightpad))
    bins_new = bins_pad[:-1] + np.diff(bins_pad)/2
    return bins_new

def bin_data(self, data, bins=None):
    x = self.x
    x_all = self.x_all
    if bins is None:     # if no bins provided, x is used to bin
        #assert len(x)==data.shape[3], 'Length of x and data to be binned do not match'
        bins = create_bins(x)
    assert data.shape[2]==len(x_all), 'Length of x_all and data to be binned do not match'
    n_scan, n_har, n_pt = data.shape
    x_mean = bs(x_all, x_all, statistic='mean', bins=bins)[0]
    x_std = bs(x_all, x_all, statistic='std', bins=bins)[0]
    n_bins = len(bins)
    y_mean = np.zeros((4, n_har, n_bins-1))
    y_std = np.zeros((4, n_har, n_bins-1))
    for i in range(4):
        #y_mean[i,:,:] = bs(x_all, data_ASF[i,:,:], statistic='mean', bins=bins)[0] 
        #y_std[i,:,:] = bs(x_all, data_ASF[i,:,:], statistic='std', bins=bins)[0]
        y_mean[i,:,:] = bs(x_all, data[i,:,:], statistic='mean', bins=bins)[0] 
        y_std[i,:,:] = bs(x_all, data[i,:,:], statistic='std', bins=bins)[0]
    self.xmean = x_mean
    self.ymean = y_mean
    self.xstd = x_std
    self.ystd = y_std

def cal_asym(self, sl, har_list=None, bins=None):
    '''
    '''
    if har_list is None:
        if self.har_setting == 'Odd':
            har_list = [el for el in range(9)]
        if self.har_setting == 'EvenOdd':
            har_list = [el for el in range(17)]
    self.har_list = har_list
    self.create_data(sl)                         # creates self.data_conc
    data2plot = self.data_conc[:, har_list, :]
    bin_data(self, data2plot, bins)              # creates self.xmean, self.ymean etc.
    data_mean = self.ymean
    Ip = data_mean[0]/data_mean[2]
    Im = data_mean[1]/data_mean[3]
    asym = -(Ip - Im)/(Ip + Im)     # negative sign used to match the sign of asymmetry
    self.asym = asym

def plot_asym(self, sl, har_list, bins=None, normalized=False):
    cal_asym(self, sl, har_list, bins)
    asym = self.asym
    x = self.xmean
    ylabel='Asymmetry'
    if normalized:
        asym_umpump = asym[:, (x<-0.05)].mean(axis=1)                      # Change -0.05 to t0 or something
        asym = asym/asym_umpump.reshape((asym.shape[0],1))
        ylabel='Normalized Asymmetry'

    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.plot(x, asym.T, 'o')
    legends = []
    for j, n_asym in enumerate(har_list):
        legends.append('asym_'+str(n_asym))
    ax.legend(legends, frameon=False)
    ax.set(xlabel='Delay (ps)', ylabel=ylabel)
    plt.show()
    
def plot_asymVsEnergy(self, sl, delay_bins=[[-1000, 0]], normalized=False):
    cal_asym(self, sl)
    asym = self.asym
    x = self.x
    ylabel='Asymmetry'
    if normalized:
        asym_umpump = asym[:, (x<-0.05)].mean(axis=1)                      # Change -0.05 to t0 or something
        asym = asym/asym_umpump.reshape((asym.shape[0],1))
        ylabel='Normalized Asymmetry'
        
    if self.har_setting == 'Odd':
        har_ener = np.linspace(31, 47, 9)*1.55
    if self.har_setting == 'EvenOdd':
        har_ener = np.linspace(31, 47, 17)*1.55
    db = delay_bins
    n_bin = len(db)
    n_asym = asym.shape[0]
    asym_db = np.zeros((n_asym, n_bin))
    
    for i in range(n_bin):
        asym_db[:, i] = asym[:, (x>db[i][0])*(x<db[i][1])].mean(axis=1)
        #asym_db[:, i] = asym_db[:, i]/np.max(asym_db[:, i])
    
    fig, ax = plt.subplots()
    ax.plot(har_ener, asym_db, 'o', ls='--')
    legends = []
    for j, el in enumerate(db):
        legends.append(str(el[0])+'-'+str(el[1])+' ps')
    ax.legend(legends, frameon=False)
    ax.set(xlabel='Photon energy (eV)', ylabel=ylabel)
    ax.axhline(ls='--',c='k')
    plt.show()
    
    return har_ener, asym_db

def fig_kwargs(kwargs):
    figsize = kwargs.get('figsize', plt.rcParams['figure.figsize'])
    title = kwargs.get('title', '')
    ls = kwargs.get('ls', '')
    return figsize, title, ls

def plot_seq(self, seq, kwargs, har_list=None, bins=None):
    figsize, title, ls = fig_kwargs(kwargs)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    legends = []
    for i in range(len(seq)):
        sl = seq[i][0]
        cal_asym(self, sl, har_list, bins)
        ax.plot(self.xmean, self.asym.T, 'o', ls=ls)
        for j in range(len(self.har_list)):
            legends.append(str(round(self.har_ener[j], 2)) + ' eV_' + str(seq[i][1]))
    ax.legend(legends, frameon=False)
    ax.set_xlabel(self.motor)
    ax.set_ylabel('Asymmetry')
    plt.show()
#------------------------------------------------------------------------------
# The class EvaluateImage
#------------------------------------------------------------------------------
class EvaluateImage:
    '''
    '''
    def __init__(self, path, fileName):
        self.path = path
        self.fileName = fileName
        hf = h5py.File(path+fileName, 'r')
        self.hf = hf
        self.har_setting = 'EvenOdd'
        scan_list = list(hf.keys())
        self.scan_list = scan_list         # scan list containing all scan no.
        self.data_image = data_im(self)
        self.im_crop = []
        self.im_shear = []
        self.cache = {}
        self.spectra = {}
        self.sl = []                   # scan list whose spectra are evaluated.
        self.mask = {}
        self.px_mid_spec = 515
        self.px_mid_ref = 565
        self.motor = 'delay'
        if self.har_setting == 'Odd':
            self.har_ener = np.linspace(31, 47, 9)*1.55
        if self.har_setting == 'EvenOdd':
            self.har_ener = np.linspace(31, 47, 17)*1.55
        self.x_all = np.array([])
        self.x = np.array([])
        
    def create_specta(self, sl, im_crop=[], im_shear=[]):
        '''
        '''
        self.sl = list(set(self.sl).union(set(sl)))
        if not im_crop:
            im_crop = self.im_crop
        if not im_shear:
            im_shear = self.im_shear
        for sn in sl:
            self.cache[sn] = [im_crop, im_shear]
        self.cache['im_crop'] = im_crop
        self.cache['im_shear'] = im_shear 
        sd = {}
        for i, sn in enumerate(sl):
            pt_list = list(self.hf[sn].keys())
            sd[sn] = {}
            for j, pn in enumerate(pt_list):
                if not im_crop and not im_shear:
                    image = self.hf[sn][pn][()]
                if im_crop and not im_shear:
                    image = crop_image(self, im_crop, sn, pn)
                if im_crop and im_shear:
                    image = process_image(self, im_crop, im_shear, sn, pn)
                if not im_crop and im_shear:
                    image = shear_image(self, im_crop, im_shear, sn, pn)  
                    
                sd[sn][pn] = image.sum(axis=0)
                self.spectra.update(sd)
        
    def create_data(self, sl):
        '''
        '''
        # if already some spectra created, check if it is necessary to update the spectra
        if self.sl:
            # if image process parameters are updated, create spectra for all the scans
            if self.cache['im_crop']!=self.im_crop or self.cache['im_shear']!=self.im_shear:
                self.create_specta(sl)
            # if im_crop and im_shear are not updated, create spectra for scans for which different image process 
            # parameters were used earlier or if the spectra does not exist for any scan.
            else:
                list_update_spectra = []
                for sn in sl:
                    #if self.cache.get(sn) != None:
                    if sn in self.cache:
                        if self.cache[sn] != [self.im_crop, self.im_shear]:
                            list_update_spectra.append(sn)
                    else:
                        list_update_spectra.append(sn)
                self.create_specta(list_update_spectra )
        else:
            self.create_specta(sl)
        
        # Creating data array for all scans for all harmonics for all motor positons. The array dimension along the
        # scan no direction is concatenated. A x_all is also thus created by concatenating all the x (motor positons)
        # for all the scans.
        
        n_Har = 17 if self.har_setting=='EvenOdd' else 9
        x = np.array([])
        x_all = np.array([])
        x_dict = {}
        key_list = [[],[],[],[]] 
        for i, sn in enumerate(sl):
            all_keys = list(self.spectra[sn].keys())
            key_list[0] = [el for el in all_keys if '.spec' in el and 'M' not in el]
            key_list[1] = [el for el in all_keys if 'M' in el and '.spec' in el]
            key_list[2] = [el for el in all_keys if '.ref' in el and 'M' not in el]
            key_list[3] = [el for el in all_keys if 'M' in el and '.ref' in el]
            
            pt_list0 = [int(el[0:4])-1 for el in key_list[0]]                # pt_list stands for the list of points for a scan
            pt_list1 = [int(el[1:5])-1 for el in key_list[1]]
            pt_list2 = [int(el[0:4])-1 for el in key_list[2]]
            pt_list3 = [int(el[1:5])-1 for el in key_list[3]]
            
            # In case some images are not saved, consider only the points having all 4 images (spec, specM, ref, refM) saved.
            pt_list_temp = [pt_list0, pt_list1, pt_list2, pt_list3]
            pt_list = list(set.intersection(*map(set, pt_list_temp))) 
            
            # create x and x_all. x contains the longest x among all the scans in the scan list. 
            # x_all contains xs (concatenated) of all the scans. 
            x_dict[i] = self.hf[sn].attrs[self.motor][pt_list]
            if len(x_dict[i])>len(x):
                x = x_dict[i]
            x_all = np.concatenate((x_all, x_dict[i]))
            # Create the data array
            n_p = len(pt_list)
            spectra_all = np.zeros((4, n_Har, n_p))
            for j, pn in enumerate(pt_list):
                spec = self.spectra[sn][key_list[0][pn]]
                specM = self.spectra[sn][key_list[1][pn]]
                ref = self.spectra[sn][key_list[2][pn]]
                refM = self.spectra[sn][key_list[3][pn]]
                mask = self.mask
                for k in range(n_Har):
                    spectra_all[0, k, j] = np.sum(spec[int(mask['spec'][0][k]):int(mask['spec'][1][k])])
                    spectra_all[1, k, j] = np.sum(specM[int(mask['spec'][0][k]):int(mask['spec'][1][k])])
                    spectra_all[2, k, j] = np.sum(ref[int(mask['ref'][0][k]):int(mask['ref'][1][k])])
                    spectra_all[3, k, j] = np.sum(refM[int(mask['ref'][0][k]):int(mask['ref'][1][k])])
            if i==0:
                spectra_conc = spectra_all
            else:
                spectra_conc = np.concatenate((spectra_conc, spectra_all), axis=2)
        self.x = x
        self.x_dict = x_dict
        self.x_all = x_all
        self.data_conc = spectra_conc
            
    def data_arr(self, sl):             # Works only for same x axis for all scans in sl. Currently not in use
        '''
        '''
        list_noSpectra = []
        if self.sl != sl:
            list_noSpectra = list(set(sl) - set(self.sl))
            self.create_specta(list_noSpectra)
        
        n_scan = len(sl)
        # Obtaining the max number of motor positions for the whole scan list
        pt_list_all = self.spectra[sl[0]].keys()
        for i, sn in enumerate(sl):
            pt_list_temp = self.spectra[sn].keys()
            pt_list_all = list(set(pt_list_all).union(set(pt_list_temp)))
            
        n_pt0 = len([el for el in pt_list_all if '.spec' in el and 'M' not in el])
        n_pt1 = len([el for el in pt_list_all if '.spec' in el and 'M' in el])
        n_pt2 = len([el for el in pt_list_all if '.ref' in el and 'M' not in el])
        n_pt3 = len([el for el in pt_list_all if '.ref' in el and 'M' in el])        
        n_pt = min(n_pt0, n_pt1, n_pt2, n_pt3)
        #n_pt = len([el for el in pt_list_all if '.spec' in el and 'M' not in el])
        n_Har = 17
        Data_arr = np.zeros((4, n_scan, n_Har, n_pt))
        key_list = [[],[],[],[]] 
        for i, sn in enumerate(sl):
            pt_list = list(self.spectra[sn].keys())
            key_list[0] = [el for el in pt_list if '.spec' in el and 'M' not in el]
            key_list[1] = [el for el in pt_list if 'M' in el and '.spec' in el]
            key_list[2] = [el for el in pt_list if '.ref' in el and 'M' not in el]
            key_list[3] = [el for el in pt_list if 'M' in el and '.ref' in el]
            assert len(key_list[0])==len(key_list[1])==len(key_list[2])==len(key_list[3]), \
            "Some images are not saved in scan no "+ sn + '. Remove the images not having their partners.'
            
            for j in range(len(key_list[0])):
                spectra = self.spectra[sn][key_list[0][j]]
                spectraM = self.spectra[sn][key_list[1][j]]
                spectra_ref = self.spectra[sn][key_list[2][j]]
                spectra_refM = self.spectra[sn][key_list[3][j]]
                mask = self.mask
                for k in range(n_Har):
                    Data_arr[0, i, k, j] = np.sum(spectra[int(mask['spec'][0][k]):int(mask['spec'][1][k])])
                    Data_arr[1, i, k, j] = np.sum(spectraM[int(mask['spec'][0][k]):int(mask['spec'][1][k])])
                    Data_arr[2, i, k, j] = np.sum(spectra_ref[int(mask['ref'][0][k]):int(mask['ref'][1][k])])
                    Data_arr[3, i, k, j] = np.sum(spectra_refM[int(mask['ref'][0][k]):int(mask['ref'][1][k])])
        
        self.data_all = Data_arr
        
    def close(self):
        self.hf.close()