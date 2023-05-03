from fit_gauss_to_lc import fit_lc_dirty, hist_rate, fit_gauss, create_logfile
import numpy as np
import os
# run python3 -m pytest in /stage/headat/yanling/XMM_datareduction
from datetime import date

def test_hist_rate():
    '''
    test the hist_rate
    Requirement: bin the rate according to assigned binsize
    Input data: 
    rate = np.random.randint(0,10, 1000)
    Prepare test: from fit_gauss_to_lc import hist_rate
    Run test: hist_rate(rate, binnum)
    Expected result: an 1 x <binnum size array without 0 

    '''
    rate = np.random.rand(1,10, 1000)
    binnum = 120
    hist, bin_mid, __ = hist_rate(rate, binnum)
    print(len(hist))
    assert binnum ==120
    assert len(hist) <= binnum
    assert len(bin_mid) <= binnum
    assert np.sum(hist ==0) == 0

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def test_fit_gauss():
    '''
    test the fit_gauss
    Requirement: output right mu, sigma
    '''
    # Generate a large number of samples from the same distribution
    np.random.seed(42)
    n_samples = 10000
    # Compute the true parameters of the distribution
    true_amp = 5
    true_mu = 10
    true_sigma = 5
    xdata = np.linspace(1,20, n_samples)
    samples = gaussian(xdata, true_amp, true_mu, true_sigma) + 0.5 * np.random.normal(size=len(xdata))

    # Fit a Gaussian distribution to the samples
    amp, mu, sigma = fit_gauss(xdata, samples)
    # Compute the relative errors of the fitted parameters
    amp_error = np.abs(amp - true_amp) / true_amp
    mu_error = np.abs(mu - true_mu) / true_mu
    sigma_error = np.abs(sigma - true_sigma) / true_sigma
    print('Relative error in amp:', amp, true_amp, amp_error)
    print('Relative error in mu:', mu, true_mu, mu_error)
    print('Relative error in sigma:', sigma, true_sigma, sigma_error)
    assert amp_error< 0.01
    assert mu_error<0.01
    assert sigma_error<0.01

def test_fit_lc_dirty():
    # Create GTI and filter
    instruments = ['EMOS1', 'EMOS2', 'EPN', 'Oot']
    data_dir = '/stage/headat/yanling/xmm/reduction_230227/101'
    scale = [1,1,1,1]
    for i, instrument in enumerate(instruments):
        fname = f'rate_{instrument}.fits'
        # create logger
        logger = create_logfile('fit_lc_dirty', f'log/fit_lc_dirty_{date.today()}.log')
        logger.info(f'filtering {fname}')
        figpath = f'/stage/headat/yanling/XMM_datareduction/fig/{date.today()}'
        os.makedirs(figpath, exist_ok=True)
        amp, mean, sigma, hist, bin_mid = fit_lc_dirty(10, fname, data_dir, scale[i], 120, logger, figpath = figpath, PLOT=True)

# test_fit_gauss()
test_fit_lc_dirty()

