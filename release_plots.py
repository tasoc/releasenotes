# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:51:06 2017

@author: Dr. Mikkel N. Lund
"""
#===============================================================================
# Packages
#===============================================================================

from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
mpl.rcParams['font.family'] = 'serif'
from matplotlib import rc
rc('text', usetex=True)
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy.interpolate as INT
import astropy.io.fits as fits
from statsmodels.nonparametric.kde import KDEUnivariate as KDE

from scipy.stats import binned_statistic as binning
plt.ioff()


#==============================================================================
# Noise model as a function of magnitude and position
#==============================================================================

def ZLnoise(gal_lat):
    # RMS noise from Zodiacal background
    rms = (16-10)*(gal_lat/90 -1)**2 + 10 # e-1 / pix in 2sec integration
    return rms
    
def Pixinaperture(Tmag):
    # Approximate relation for pixels in aperture (based on plot in Sullivan et al.)
    pixels = (30 + (((3-30)/(14-7)) * (Tmag-7)))*(Tmag<14) + 3*(Tmag>=14) #+ np.random.normal(0, 2)
    return int(np.max([pixels, 3]))

def mean_flux_level(Tmag, Teff):    
    # Magnitude system based on Sullivan et al.
    collecting_area = np.pi*(10.5/2)**2 # square cm
    Teff_list = np.array([2450, 3000, 3200, 3400, 3700, 4100, 4500, 5000, 5777, 6500, 7200, 9700]) # Based on Sullivan
    Flux_list = np.array([2.38, 1.43, 1.40, 1.38, 1.39, 1.41, 1.43, 1.45, 1.45, 1.48, 1.48, 1.56])*1e6 # photons per sec; Based on Sullivan
    Magn_list = np.array([306, -191, -202, -201, -174, -132, -101, -80, -69.5, -40, -34.1, 35])*1e-3 #Ic-Tmag (mmag)
    
    
    Flux_int = INT.UnivariateSpline(Teff_list, Flux_list, k=1, s=0)
    Magn_int = INT.UnivariateSpline(Teff_list, Magn_list, k=1, s=0)
    
    Imag = Magn_int(Teff)+Tmag
    Flux = 10**(-0.4*Imag) * Flux_int(Teff) * collecting_area
    
    return Flux


def phot_noise(Tmag, Teff, cad, PARAM, verbose=False, sysnoise=60):
    
	# Calculate galactic latitude for Zodiacal noise
	gc= SkyCoord(PARAM['RA']*u.degree, PARAM['DEC']*u.degree, frame='icrs')
#    gc = SkyCoord(lon=PARAM['ELON']*u.degree, lat=PARAM['ELAT']*u.degree, frame='barycentrictrueecliptic')
	gc_gal = gc.transform_to('galactic')
	gal_lat0 = gc_gal.b.deg

	gal_lat = np.arcsin(np.abs(np.sin(gal_lat0*np.pi/180)))*180/np.pi
           
	# Number of 2 sec integrations in cadence
	integrations = cad/2
    
	# Number of pixels in aperture given Tmag
	pixels = int(Pixinaperture(Tmag))
    
	# noise values are in rms, so square-root should be used when factoring up
	Flux_factor = np.sqrt(integrations * pixels) 
        
	# Mean flux level in electrons per cadence    
	mean_level_ppm = mean_flux_level(Tmag, Teff) * cad # electrons
       
	# Shot noise
	shot_noise = 1e6/np.sqrt(mean_level_ppm)
        
	# Read noise
	read_noise = 10 * Flux_factor *1e6/mean_level_ppm # ppm
        
	# Zodiacal noise
	zodiacal_noise = ZLnoise(gal_lat) * Flux_factor *1e6/mean_level_ppm # ppm  
         
	# Systematic noise in ppm
	systematic_noise_ppm = sysnoise / np.sqrt(cad/(60*60)) # ppm / sqrt(hr)
    
    
	if verbose:
		print('Galactic latitude', gal_lat)
		print('Systematic noise in ppm', systematic_noise_ppm)
		print('Integrations', integrations)  
		print('Pixels', pixels)
		print('Flux factor', Flux_factor)
		print('Mean level ppm', mean_level_ppm)
		print('Shot noise', shot_noise)
		print('Read noise', read_noise)
		print('Zodiacal noise', zodiacal_noise)    

    
	PARAM['Galactic_lat'] = gal_lat
	PARAM['Pixels_in_aper'] = pixels
    
	noise_vals = np.array([shot_noise, zodiacal_noise, read_noise, systematic_noise_ppm])
	return noise_vals, PARAM # ppm per cadence
    

def compute_onehour_rms(time, flux):

	cad = int(np.nanmedian((np.diff(time)))*(60*60*24))
	
	D = np.array([cad-120, cad-1800, cad-20])
	Da = np.abs(D)
	
	Ns = np.array([30, 2, 1])
	N = Ns[np.argmin(Da)]
		
	bins = int(np.ceil(len(time)/N)) + 1
		
	idx_finite = np.isfinite(flux) & np.isfinite(time)
	
	flux_finite = flux[idx_finite]
	bin_means = np.array([])
	ii = 0;
	
	print(N)
	for ii in range(bins):
		try:
			m = np.nanmean(flux_finite[ii*N:(ii+1)*N])
			bin_means = np.append(bin_means, m)
		except:
			continue		
		
#	print(bin_means)	
#	bin_means, bin_edges, _ = binning(time[idx_finite], flux[idx_finite], statistic='mean', bins=bins)
#	bin_width = (bin_edges[1] - bin_edges[0])
#	bin_centers = bin_edges[1:] - bin_width/2
##	
#	dd = (np.median(np.diff(bin_centers))*60*60*24)
	
#		plt.figure()
##		plt.plot(bin_centers, bin_means)
#		plt.plot(bin_means)
#		plt.show()
	
	RMS = 1.4826*np.nanmedian(np.abs((bin_means - np.nanmedian(bin_means))))
	
	print(RMS)
#	square_diffs = np.abs((bin_means - np.nanmedian(bin_means)))**2
##	RMS = np.sqrt(np.nansum(() / len(bin_means))
#	RMS = np.sqrt(np.nanmedian(square_diffs))
	
	return RMS


# =============================================================================
# 
# =============================================================================


def plot_onehour_noise(data_paths, sector, cad=1800, sysnoise=0):
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	PARAM = {}
	
	# Add data values	
	files = np.array([])
	for d in data_paths:
		files = np.append(files, np.array([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.fits')]))
	files = files.flatten()
	
	rms_tmag_vals = np.zeros([len(files), 3])
	for i, f in enumerate(files):
		hdu = fits.open(f)
		tmag = hdu[0].header['TESSMAG']
		time = hdu[1].data['TIME']
		flux = hdu[1].data['FLUX_CORR']
		
		dt = np.nanmedian(np.diff(time))
#		flux = hdu[1].data['FLUX_RAW']
#		Q = hdu[1].data['QUALITY']
#		rms = compute_onehour_rms(time[(Q==0)], flux[(Q==0)])
		
		rms = compute_onehour_rms(time, flux)
		
		rms_tmag_vals[i, 0] = tmag
		
		if int(round(dt))==120:
			rms_tmag_vals[i, 1] = rms
		else:
			rms_tmag_vals[i, 2] = rms

		
		# TODO: Update elat+elon based on observing sector?
		PARAM['RA']=hdu[0].header['RA_OBJ']
		PARAM['DEC']=hdu[0].header['DEC_OBJ']

	
	idx_sc = np.nonzero(rms_tmag_vals[:, 1])
	idx_lc = np.nonzero(rms_tmag_vals[:, 2])
	
	ax.scatter(rms_tmag_vals[idx_sc, 0], rms_tmag_vals[idx_sc, 1], marker='o', facecolors='None', color='k')
	ax.scatter(rms_tmag_vals[idx_lc, 0], rms_tmag_vals[idx_lc, 2], marker='o', facecolors='None', color='r')
	
	
	# Plot theoretical lines
	mags = np.linspace(3.5, 16.5, 50)
	vals = np.zeros([len(mags), 4])
	
    
	for i in range(len(mags)):
		vals[i,:], _ = phot_noise(mags[i], 5775, cad, PARAM, sysnoise=sysnoise, verbose=False)    
        
	
	ax.semilogy(mags, vals[:, 0], 'r-')   
	ax.semilogy(mags, vals[:, 1], 'g--')   
	ax.semilogy(mags, vals[:, 2], '-')   
	ax.semilogy(mags, np.sqrt(np.sum(vals**2, axis=1)), 'k-')   
	ax.axhline(y=sysnoise, color='b', ls='--') 
	
	
	ax.set_xlim([3.5, 16.5])
	ax.set_ylim([10, 1e5])  
	
	ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
	ax.set_ylabel(r'$\rm RMS\,\, (ppm\,\, hr^{-1})$', fontsize=16, labelpad=10)
	
	ax.xaxis.set_major_locator(MultipleLocator(2))
	ax.xaxis.set_minor_locator(MultipleLocator(1))
	ax.tick_params(direction='out', which='both', pad=5, length=3) 
	ax.tick_params(which='major', pad=6, length=5,labelsize='15') 
	plt.tight_layout()
	
	save_path = 'plots/sector%02d/' %sector
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	fig.savefig(os.path.join(save_path, 'rms_noise.png') )
	 
	plt.show()
	
# =============================================================================
# 	
# =============================================================================

def plot_pixinaperture(data_path, sector, cad=1800, sysnoise=0):
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	

	# Add data values	
	files = np.array([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.fits')])
	
	ap_tmag_vals = np.zeros([len(files), 2])
	for i, f in enumerate(files):
		hdu = fits.open(f)
		tmag = hdu[0].header['TESSMAG']
		ap = hdu[3].data
		
		in_aperture = (ap & 2+8 != 0)
#		print(in_aperture)
		ap_tmag_vals[i, 0] = tmag
		ap_tmag_vals[i, 1] = np.sum(in_aperture)
		
	
	# Colour by distance from camera centre
	ax.scatter(ap_tmag_vals[:, 0], ap_tmag_vals[:, 1], marker='o', facecolors='None', color='k')
	
	mags = np.linspace(3.5, 16.5, 500)
	pix = np.asarray([Pixinaperture(m) for m in mags], dtype='float64')
	ax.plot(mags, pix, color='k', ls='-')
    
	
	ax.set_xlim([3.5, 16.5])
	ax.set_ylim([1, 100])  
	
	ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
	ax.set_ylabel('Pixels in aperture', fontsize=16, labelpad=10)
	
	ax.xaxis.set_major_locator(MultipleLocator(2))
	ax.xaxis.set_minor_locator(MultipleLocator(1))
	ax.yaxis.set_major_locator(MultipleLocator(20))
	ax.yaxis.set_minor_locator(MultipleLocator(10))
	ax.tick_params(direction='out', which='both', pad=5, length=3) 
	ax.tick_params(which='major', pad=6, length=5,labelsize='15') 
	plt.tight_layout()
	
	save_path = 'plots/sector%02d/' %sector
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	fig.savefig(os.path.join(save_path, 'pix_in_aperture.png') )
	 
	plt.show()


# =============================================================================
# 
# =============================================================================

def plot_mag_dist(data_path, sector):
	

	# Add data values	
	files = np.array([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.fits')])
	
	tmag_vals_sc = np.array([])
	tmag_vals_lc = np.array([])
	for i, f in enumerate(files):
		hdu = fits.open(f)
		tmag = hdu[0].header['TESSMAG']
		time = hdu[1].data['TIME']
		
		dt = np.nanmedian(np.diff(time))


		if int(round(dt))==120:
			tmag_vals_sc = np.append(tmag_vals_sc, tmag)
		else:
			tmag_vals_lc = np.append(tmag_vals_lc, tmag)
			
			
			
			
	fig = plt.figure()
	ax = fig.add_subplot(111)		
					
	try:	
		kde_lc = KDE(tmag_vals_lc)
		kde_lc.fit(gridsize=1000)
		ax.plot(kde_lc.support, kde_lc.density, label='LC')
		ax.scatter(tmag_vals_lc, np.zeros_like(tmag_vals_lc), lw=1, marker='+', s=80)
	except:
		pass
	
	
	try:
		kde_sc = KDE(tmag_vals_sc)
		kde_sc.fit(gridsize=1000)
		ax.plot(kde_sc.support, kde_sc.density, label='SC')
		ax.scatter(tmag_vals_sc, np.zeros_like(tmag_vals_sc), lw=1, marker='+', s=80)
	except:
		pass		
	
#	ax.set_xlim([3.5, 16.5])
	ax.set_ylim(ymin=0)  
	
	ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
	ax.set_ylabel('KDE', fontsize=16, labelpad=10)
	
	ax.xaxis.set_major_locator(MultipleLocator(2))
	ax.xaxis.set_minor_locator(MultipleLocator(1))
	ax.tick_params(direction='out', which='both', pad=5, length=3) 
	ax.tick_params(which='major', pad=6, length=5,labelsize='15') 
	plt.tight_layout()
	
	save_path = 'plots/sector%02d/' %sector
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	fig.savefig(os.path.join(save_path, 'magnitudes.png') )
	
	ax.legend(frameon=False, prop={'size':12} ,loc='upper right', borderaxespad=0,handlelength=2.5, handletextpad=0.4) 

	
	plt.show()
	
	
	
# =============================================================================
# 
# =============================================================================

if __name__ == "__main__":  

	
	data_path = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TDA5/'
	
	
	plot_onehour_noise([data_path,], cad = 1800, sector=1, sysnoise=0)
	
	
	plot_pixinaperture(data_path, sector=1)
	
	plot_mag_dist(data_path, sector=1)