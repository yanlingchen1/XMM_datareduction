import os, glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
import subprocess as s
warnings.filterwarnings("ignore")

from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from scipy.optimize import curve_fit
from copy import deepcopy
import logging
from datetime import date

def create_logfile(name, savefile):
    import logging
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs messages to a file
    fh = logging.FileHandler(savefile)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add the handler to the logger
    logger.addHandler(fh)
    return logger
    # # log some messages
    # logger.debug('Debug message')
    # logger.info('Info message')
    # logger.warning('Warning message')
    # logger.error('Error message')
    # logger.critical('Critical message')

def plotting(x, y, y_err, model, xlim, instrument, figpath, xscale='linear', bkg=False):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.tick_params(direction='in', which='both', length=5, labelsize=15)
    ax2.tick_params(direction='in', which='both', length=5, labelsize=15)
    instrument = instrument.split('.')[0]
    plt.suptitle(instrument)
    ax1.set_ylabel(r'N', fontsize = 15.0)
    ax1.set_xscale(xscale)
    ax1.set_xlim(0, xlim[1])
    ax1.errorbar(x, y, yerr=y_err, fmt='o', capsize=2, label='Data')
    ax1.plot(x, model, label='Fit')
    ax1.legend(loc='best', fontsize=15.0)
    ax2.set_ylabel(r'Ratio', fontsize = 15.0)
    ax2.set_ylim(0.5,1.5)
    ax2.axhline(y=1.)
    ax2.errorbar(x, y/model, yerr=y_err/model, fmt='o', capsize=2)
    ax2.set_xlabel('Rate [cts/s]', fontsize = 15.0)
    plt.subplots_adjust(hspace=0.1)
    plt.show()
    plt.clf()
    # plt.savefig(f'{figpath}/{instrument}_ratehist.png')

def input_lc(fname, datapath, PN_cts_limit, EMOS_cts_limit):
    hdul = fits.open(f'{datapath}/{fname}')
    data = hdul[1].data
    time = data['time']
    rate = data['rate']
    rate_err = data['error']
    hdul.close()
    # Delete zero rates
    where = np.where(rate == 0)[0]
    time = np.delete(time, where)
    rate = np.delete(rate, where)
    rate_err = np.delete(rate_err, where)
    # filter maximum counts rate before fitting
    if('EPN' in fname or 'Oot' in fname):
        where = np.where(rate > PN_cts_limit)[0]
        time = np.delete(time, where)
        rate = np.delete(rate, where)
        rate_err = np.delete(rate_err, where)
            
    else:
        where = np.where(rate > EMOS_cts_limit)[0]
        time = np.delete(time, where)
        rate = np.delete(rate, where)
        rate_err = np.delete(rate_err, where)
    return rate, time, rate_err



def hist_rate(rate, binnum):
    hist, bin_edges = np.histogram(np.log10(rate), bins = binnum)
    bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])
    bin_mid = np.power(10, bin_mid)
    # delete where histdata is 0
    where = np.where(hist == 0)[0]
    bin_mid = np.delete(bin_mid, where)
    hist = np.delete(hist, where)
    return hist, bin_mid


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def fit_gauss(bin_mid, hist):
    init_vals = [max(hist), np.mean(bin_mid), np.std(bin_mid)]
    best_vals, covar = curve_fit(gaussian, bin_mid, hist, p0=init_vals,  bounds=((0, 0, 0),(np.inf, np.inf, np.inf)),maxfev=10000)
    return best_vals[0], best_vals[1], best_vals[2]
    
def fit(rate, time, scale, binnum):
    hist, bin_mid = hist_rate(rate, binnum)
    amp, mu,sigma = fit_gauss(bin_mid, hist)
    where = (rate<mu+scale*sigma)&(rate>mu-scale*sigma)
    newrate = rate[where]
    newtime = time[where]
    return newrate, newtime, amp, mu, sigma, hist, bin_mid

def fit_lc_dirty(iter_cts, fname, datapath, scale, binnum,  logger, figpath = '/stage/headat/yanling/XMM_datareduction/fig', PN_cts_limit=1.5, EMOS_cts_limit=1.5, PLOT=True):
    
    rate, time, rate_err = input_lc(fname, datapath, PN_cts_limit, EMOS_cts_limit)
    newrate, newtime, amp, mu, sigma, __, __ = fit(rate, time,  scale, binnum)
    i = 0
    while (i < iter_cts) :
        newrate, newtime, amp, mu, sigma, hist, bin_mid = fit(newrate, newtime, scale, binnum)
        logger.info(f'selected data num:{len(newtime)}; fitting properties (amp, mu, sigma): {amp}, {mu}, {sigma}')
        i+=1
    # make the plot if the PLOT ==True
    if PLOT==True:
        plotting(bin_mid, hist, np.sqrt(hist), gaussian(bin_mid, amp, mu, sigma), 
                    [mu-scale*sigma, mu+scale*sigma], fname, 
                    figpath, bkg=True)
        # check with the original lightcurve data
        plt.figure(figsize = (8,8))
        plt.scatter(newtime,newrate, alpha=0.5, s=2, color = 'b')#, yerr=dat['ERROR'])
        plt.scatter(time,rate, alpha=0.5, s=1, color = 'k')#, yerr=dat['ERROR'])
        plt.axhline(mu+scale*sigma, color = 'r')
        plt.axhline(mu-scale*sigma, color = 'r')
        plt.title(f'{fname.split(".")[0]}')
        plt.yscale('log')
        plt.savefig(f'{figpath}/{fname.split(".")[0]}_ratevstime.png')
        plt.show()
        plt.close()
    return amp, mu, sigma, hist, bin_mid