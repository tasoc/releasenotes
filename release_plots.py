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
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
mpl.rcParams['font.family'] = 'serif'
from matplotlib import rc
rc('text', usetex=True)
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy.interpolate as INT
from astropy.io import fits
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
#from pywcsgrid2.allsky_axes import make_allsky_axes_from_header, allsky_header
#import matplotlib.patheffects
#import pywcsgrid2.healpix_helper as healpix_helper
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

# =============================================================================
#
# =============================================================================

def compute_onehour_rms(flux, cad):

	if cad==120:
		N=30
	elif cad==1800:
		N=2
	else:
		N=1

	bins = int(np.ceil(len(flux)/N)) + 1

	idx_finite = np.isfinite(flux)


	flux_finite = flux[idx_finite]
	bin_means = np.array([])
	ii = 0;

	for ii in range(bins):
		try:
			m = np.nanmean(flux_finite[ii*N:(ii+1)*N])
			bin_means = np.append(bin_means, m)
		except:
			continue


	# Compute robust RMS value (MAD scaled to RMS)
	RMS = 1.4826*np.nanmedian(np.abs((bin_means - np.nanmedian(bin_means))))
	PTP = np.nanmedian(np.abs(np.diff(flux_finite)))

	return RMS, PTP


# =============================================================================
#
# =============================================================================


def plot_onehour_noise(data_paths, sector, cad=1800, sysnoise=0):

	norm = colors.Normalize(vmin=0, vmax=len(data_paths)-1)
	scalarMap = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('tab10') )


	fig = plt.figure()
	fig.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
	ax = fig.add_subplot(111)

	fig2 = plt.figure()
	fig2.subplots_adjust(left=0.145, wspace=0.3, top=0.945, bottom=0.145, right=0.975)
	ax2 = fig2.add_subplot(111)

	PARAM = {}

	# Add data values
#	files = np.array([])
#	files = files.flatten()

#	cols = np.array(['r', 'b', 'c', 'g', 'm'])

	for k, d in enumerate(data_paths):
#		files = np.append(files, np.array([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.fits')]))
		files = np.array([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.fits.gz')])

		print(k, d)
#		if k==0:
#			tot_rms_tmag_vals = np.zeros([len(files), 6])

		rms_tmag_vals = np.zeros([len(files), 5])
		for i, f in enumerate(files):
			with fits.open(f) as hdu:
				tmag = hdu[0].header['TESSMAG']
				flux = hdu[1].data['FLUX_CORR']

				rms_tmag_vals[i, 0] = tmag

	#			if k==0:
	#				tot_rms_tmag_vals[i, 0] = tmag

				if hdu[1].header.get('NUM_FRM', 60) == 60:
					rms, ptp = compute_onehour_rms(flux, 120)
					rms_tmag_vals[i, 1] = rms
					rms_tmag_vals[i, 3] = ptp

	#				tot_rms_tmag_vals[i,k+1] = rms
	#				tot_rms_tmag_vals[i,k+1] = np.nanmedian(np.diff(flux))
				else:
					rms, ptp = compute_onehour_rms(flux, 1800)
					rms_tmag_vals[i, 2] = rms
					rms_tmag_vals[i, 4] = ptp

				# TODO: Update elat+elon based on observing sector?
				PARAM['RA'] = hdu[0].header['RA_OBJ']
				PARAM['DEC'] = hdu[0].header['DEC_OBJ']

		print(k, d)
#		if k==0:
#			tot_rms_tmag_vals = np.zeros([len(files), 6])

		rms_tmag_vals = np.zeros([len(files), 6])
		for i, f in enumerate(files):
			hdu = fits.open(f)
			tmag = hdu[0].header['TESSMAG']
			flux = hdu[1].data['FLUX_CORR']

			rms_tmag_vals[i, 0] = tmag
			rms_tmag_vals[i, -1] = hdu[0].header['TICID']


			if hdu[1].header.get('NUM_FRM',60)==60:
				rms, ptp = compute_onehour_rms(flux, 120)
				rms_tmag_vals[i, 1] = rms
				rms_tmag_vals[i, 3] = ptp

#				tot_rms_tmag_vals[i,k+1] = rms
#				tot_rms_tmag_vals[i,k+1] = np.nanmedian(np.diff(flux))
			else:
				rms, ptp = compute_onehour_rms(flux, 1800)
				rms_tmag_vals[i, 2] = rms
				rms_tmag_vals[i, 4] = ptp


			# TODO: Update elat+elon based on observing sector?
			PARAM['RA']=hdu[0].header['RA_OBJ']
			PARAM['DEC']=hdu[0].header['DEC_OBJ']


		idx_sc = np.nonzero(rms_tmag_vals[:, 1])
		idx_lc = np.nonzero(rms_tmag_vals[:, 2])

		rgba_color = scalarMap.to_rgba(k)

		ax.scatter(rms_tmag_vals[idx_sc, 0], rms_tmag_vals[idx_sc, 1], marker='o', facecolors='None', edgecolor=rgba_color)
		ax.scatter(rms_tmag_vals[idx_lc, 0], rms_tmag_vals[idx_lc, 2], marker='s', facecolors='None', edgecolor=rgba_color)

		ax2.scatter(rms_tmag_vals[idx_sc, 0], rms_tmag_vals[idx_sc, 3], marker='o', facecolors='None', edgecolor=rgba_color)
		ax2.scatter(rms_tmag_vals[idx_lc, 0], rms_tmag_vals[idx_lc, 4], marker='s', facecolors='None', edgecolor=rgba_color)

	# Plot theoretical lines
	mags = np.linspace(3.5, 16.5, 50)
	vals = np.zeros([len(mags), 4])
	vals2 = np.zeros([len(mags), 4])

#	print(tot_rms_tmag_vals)

#	plt.figure()
#	plt.scatter(tot_rms_tmag_vals[:, 0], tot_rms_tmag_vals[:, 1] - tot_rms_tmag_vals[:, 3], facecolors='r', marker='+', color='r')
#	plt.scatter(tot_rms_tmag_vals[:, 0], tot_rms_tmag_vals[:, 2] - tot_rms_tmag_vals[:, 3], facecolors='b', marker='+', color='b')
#	plt.scatter(tot_rms_tmag_vals[:, 0], tot_rms_tmag_vals[:, 4] - tot_rms_tmag_vals[:, 3], facecolors='g', marker='+', color='g')
#	plt.scatter(tot_rms_tmag_vals[:, 0], tot_rms_tmag_vals[:, 5] - tot_rms_tmag_vals[:, 3], facecolors='m', marker='+', color='m')

	for i in range(len(mags)):
		vals[i,:], _ = phot_noise(mags[i], 5775, cad, PARAM, sysnoise=sysnoise, verbose=False)

	ax.semilogy(mags, vals[:, 0], 'r-')
	ax.semilogy(mags, vals[:, 1], 'g--')
	ax.semilogy(mags, vals[:, 2], '-')
	ax.semilogy(mags, np.sqrt(np.sum(vals**2, axis=1)), 'k-')
	ax.axhline(y=sysnoise, color='b', ls='--')

	for i in range(len(mags)):
		vals2[i,:], _ = phot_noise(mags[i], 5775, 120, PARAM, sysnoise=sysnoise, verbose=False)

	ax2.semilogy(mags, vals2[:, 0], 'r-')
	ax2.semilogy(mags, vals2[:, 1], 'g--')
	ax2.semilogy(mags, vals2[:, 2], '-')
	ax2.semilogy(mags, np.sqrt(np.sum(vals2**2, axis=1)), 'k-')

	for i in range(len(mags)):
		vals[i,:], _ = phot_noise(mags[i], 5775, cad, PARAM, sysnoise=sysnoise, verbose=False)
	tot_noise = np.sqrt(np.sum(vals**2, axis=1))

	noi_vs_mag = INT.UnivariateSpline(mags, tot_noise)
	idx = (rms_tmag_vals[:, 1]/noi_vs_mag(rms_tmag_vals[:, 0]) < 1)
	print([int(x) for x in rms_tmag_vals[idx, -1]])
	print([x for x in rms_tmag_vals[idx, 0]])

	ax.semilogy(mags, vals[:, 0], 'r-')
	ax.semilogy(mags, vals[:, 1], 'g--')
	ax.semilogy(mags, vals[:, 2], '-')
	ax.semilogy(mags, tot_noise, 'k-')
	ax.axhline(y=sysnoise, color='b', ls='--')

	for i in range(len(mags)):
		vals2[i,:], _ = phot_noise(mags[i], 5775, 120, PARAM, sysnoise=sysnoise, verbose=False)
	tot_noise2 = np.sqrt(np.sum(vals2**2, axis=1))

	ax2.semilogy(mags, vals2[:, 0], 'r-')
	ax2.semilogy(mags, vals2[:, 1], 'g--')
	ax2.semilogy(mags, vals2[:, 2], '-')
	ax2.semilogy(mags, tot_noise2, 'k-')

	ax.set_xlim([3.5, 16.5])
	ax.set_ylim([10, 1e5])
	ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
	ax.set_ylabel(r'$\rm RMS\,\, (ppm\,\, hr^{-1})$', fontsize=16, labelpad=10)
	ax.xaxis.set_major_locator(MultipleLocator(2))
	ax.xaxis.set_minor_locator(MultipleLocator(1))
	ax.tick_params(direction='out', which='both', pad=5, length=3)
	ax.tick_params(which='major', pad=6, length=5,labelsize='15')

	###########
	ax2.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
	ax2.set_ylabel('point-to-point MDV (ppm)', fontsize=16, labelpad=10)
	ax.tick_params(direction='out', which='both', pad=5, length=3)
	ax.tick_params(which='major', pad=6, length=5,labelsize='15')

	ax.yaxis.set_ticks_position('both')

	###########
	ax2.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
	ax2.set_ylabel(r'$\rm point-to-point\,\, MDV\,\, (ppm)$', fontsize=16, labelpad=10)
	ax2.set_xlim([3.5, 16.5])
	ax2.set_yscale("log", nonposy='clip')
	ax2.xaxis.set_major_locator(MultipleLocator(2))
	ax2.xaxis.set_minor_locator(MultipleLocator(1))
	ax2.tick_params(direction='out', which='both', pad=5, length=3)
	ax2.tick_params(which='major', pad=6, length=5,labelsize='15')
	ax2.yaxis.set_ticks_position('both')

#	plt.tight_layout()

	save_path = 'plots/sector%02d/' %sector
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	fig.savefig(os.path.join(save_path, 'rms_noise.png'), bb_inches='tight')
	fig2.savefig(os.path.join(save_path, 'mvd_noise.png'), bb_inches='tight')

	plt.show()

# =============================================================================
#
# =============================================================================

def plot_pixinaperture(data_path, sector, cad=1800, sysnoise=0):

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# Add data values
	files = np.array([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.fits.gz')])

	# Add data values
	files = np.array([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.fits.gz')])

	ap_tmag_vals = np.zeros([len(files), 2])
	for i, f in enumerate(files):
		hdu = fits.open(f)
		tmag = hdu[0].header['TESSMAG']
		ap = hdu[3].data

		in_aperture = (ap & 2+8 != 0)
#		print(in_aperture)
		ap_tmag_vals[i, 0] = tmag
		ap_tmag_vals[i, 1] = np.sum(in_aperture)
		if np.sum(in_aperture)==15:
			print(hdu[0].header['TICID'])

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
	ax.yaxis.set_ticks_position('both')
	plt.tight_layout()

	save_path = 'plots/sector%02d/' %sector
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	fig.savefig(os.path.join(save_path, 'pix_in_aperture.png'), bb_inches='tight')

	plt.show()


# =============================================================================
#
# =============================================================================

def plot_mag_dist(data_path, sector):

	# Add data values
	files = np.array([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.fits.gz')])

	# Add data values
	files = np.array([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.fits.gz')])

	tmag_vals_sc = np.array([])
	tmag_vals_lc = np.array([])
	for i, f in enumerate(files):
		hdu = fits.open(f)
		tmag = hdu[0].header['TESSMAG']
		#time = hdu[1].data['TIME']

		dt = hdu[1].header['TIMEDEL'] * 86400

		if dt < 1000:
			tmag_vals_sc = np.append(tmag_vals_sc, tmag)
		else:
			tmag_vals_lc = np.append(tmag_vals_lc, tmag)


	fig = plt.figure()
	ax = fig.add_subplot(111)

	if len(tmag_vals_lc) > 0:
		kde_lc = KDE(tmag_vals_lc)
		kde_lc.fit(gridsize=1000)
		ax.fill_between(kde_lc.support, 0, kde_lc.density*len(tmag_vals_lc), color='b', alpha=0.3, label='1800s')
		ax.scatter(tmag_vals_lc, np.zeros_like(tmag_vals_lc), lw=1, marker='|', c='k', s=80)

	if len(tmag_vals_sc) > 0:
		kde_sc = KDE(tmag_vals_sc)
		kde_sc.fit(gridsize=1000)
		ax.fill_between(kde_sc.support, 0, kde_sc.density*len(tmag_vals_sc), color='r', alpha=0.3, label='120s')
		ax.scatter(tmag_vals_sc, np.zeros_like(tmag_vals_sc), lw=1, marker='|', c='k', s=80)

	tmag_all = np.append(tmag_vals_lc, tmag_vals_sc)
	kde_all = KDE(tmag_all)
	kde_all.fit(gridsize=1000)
	ax.plot(kde_all.support, kde_all.density*len(tmag_all), 'k-', lw=1.5, label='All')

#	ax.set_xlim([3.5, 16.5])
	ax.set_ylim(ymin=0)
	ax.set_xlabel('TESS magnitude', fontsize=16, labelpad=10)
	ax.set_ylabel('Number of stars', fontsize=16, labelpad=10)
	ax.xaxis.set_major_locator(MultipleLocator(2))
	ax.xaxis.set_minor_locator(MultipleLocator(1))
	ax.tick_params(direction='out', which='both', pad=5, length=3)
	ax.tick_params(which='major', pad=6, length=5,labelsize='15')
	ax.yaxis.set_ticks_position('both')
	plt.tight_layout()
	ax.legend(frameon=False, prop={'size':12} ,loc='upper right', borderaxespad=0,handlelength=2.5, handletextpad=0.4)

	save_path = 'plots/sector%02d/' %sector
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	fig.savefig(os.path.join(save_path, 'magnitudes.png'), bb_inches='tight')
	plt.show()

# =============================================================================
#
# =============================================================================

#def do_allsky(ax, coord):
#
#    gh_gal = ax[coord].get_grid_helper()
#
#    for d in range(0, 361, 30):
#        axis = gh_gal.new_floating_axis(nth_coord=0, value=d,
#                                        axes=ax,
#                                        axis_direction='bottom',
#                                        allsky=True)
#        ax.axis["a=%d" % d] = axis
#        axis.set_ticklabel_direction("-")
#        axis.set_axislabel_direction("-")
#        axis.toggle(all=False)
#        axis.get_helper().set_extremes(-90,90)
#        axis.line.set_color("0.7")
#        axis.set_zorder(2.)
#
#    gh_gal.locator_params(nbins=9)
#
#    axis = gh_gal.new_floating_axis(nth_coord=1, value=0,
#                                    axes=ax,
#                                    axis_direction='bottom',
#                                    allsky=True)
#    from mpl_toolkits.axisartist.floating_axes import ExtremeFinderFixed
#
#    #glon_min, glon_max = -180+0.001, 180
#    glon_min, glon_max = 0, 360 - 0.001
#    axis.get_helper().set_extremes(glon_min, glon_max)
#    gh_gal.grid_finder.extreme_finder = ExtremeFinderFixed((glon_min, glon_max, -90, 90))
#
#
#    axis.set_ticklabel_direction("-")
#    axis.set_axislabel_direction("-")
#    axis.set_zorder(5.)
#    axis.toggle(all=False, ticklabels=True)
#    axis.line.set_linewidth(1.5)
#    ax.axis["b=0"] = axis
#
#    ef = matplotlib.patheffects.withStroke(foreground="w", linewidth=3)
#    axis.major_ticklabels.set_path_effects([ef])
#
#    ax.grid()
#
#    ax["gal"].annotate("G.C.", (0,0), xycoords="data",
#                       xytext=(20, 10), textcoords="offset points",
#                       ha="left",
#                       arrowprops=dict(arrowstyle="->"),
#                       bbox=dict(fc="0.5", ec="none", alpha=0.3))
#
#    return ax


#def get_LAB_healpix_data():
#    import pyfits
#    fname = "LAB_fullvel.fits"
#    f = pyfits.open(fname)
#
#    #ordering = f[1].header["ordering"]
#    nside = f[1].header["nside"]
#    data = f[1].data["temperature"]
#
#    healpix_data = healpix_helper.HealpixData(nside, data.flat,nested=False, flipy=True,coord="gal")
#
#    return healpix_data


#if 1:
#
#    proj_list = ["CYP", "CEA", "CAR", "MER", "SFL", "PAR", "MOL", ]
#
#    DO_HEALPIX = True
#
#    if DO_HEALPIX:
#        healpix_data = get_LAB_healpix_data()
#    else:
#        healpix_data = None
#
#    for proj in proj_list:
#        fig = plt.figure()
#        rect = 111
#
#        coord, lon_center = "fk5", 180
#        header = allsky_header(coord=coord, proj=proj,
#                               lon_center=lon_center, cdelt=0.2)
#        ax = make_allsky_axes_from_header(fig, rect, header, lon_center=lon_center)
#
#        ax.set_title("proj = %s" % proj, position=(0.5, 1.1))
#
#        do_allsky(ax, "gal")
#
#        if healpix_data is not None:
#            d = healpix_data.get_projected_map(header)
#            im = ax.imshow(d**.5, origin="lower", cmap="gist_heat_r")
#            c1, c2 = im.get_clim()
#            im.set_clim(c1, c2*0.8)
#
#    plt.show()

# =============================================================================
#
# =============================================================================

if __name__ == "__main__":

	plt.close('all')

#	path0 = '/home/mikkelnl/ownCloud/Documents/Asteroseis/Emil/TESS_alerts/'
	path0 = r'C:\Users\au195407\Downloads\Ny mappe'
#	data_paths = np.array([path0 + '08', path0 + '09', path0 + '10', path0 + '11', path0 + '12'])
	#data_paths = np.array([path0 + '10',]

	plot_onehour_noise([path0], cad=1800, sector=1, sysnoise=0)
	plot_pixinaperture(path0, sector=1)
	plot_mag_dist(path0, sector=1)
