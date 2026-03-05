import os
import sys
import glob
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

from astropy.table import Table, vstack, join
from astropy.io import fits

from scipy.special import gammaincinv
from scipy.optimize import curve_fit

import matplotlib as mpl
from matplotlib import pyplot as plt

# sys.path.append('/data1/imsang/Spyder_Projects/my_modules')
# from my_visualize import plot_images as vi
# import my_matplotlib
# my_matplotlib.import_my_plt_settings()


##################
## My Functions ##
##################

def getGalStars_grouped(gal_id, host_halo_id, grouped_stars, all_subhalos_of_host, h, remove_far_stars=False, redshift=None, gal_row=None):
    
    ## --- get subhalo_idx of this galaxy --- ##
    if (len(all_subhalos_of_host) <= 1):
        subhalo_id = 0
    else:
        subhalo_id = np.where(all_subhalos_of_host['ID'] == gal_id)[0][0]
    ## -------------------------------------- ##
    
    subhalo_stars = grouped_stars.get_group((host_halo_id, subhalo_id))
    
    # ## --- using 2D half-mass radius --- ##
    # if remove_far_stars and (gal_row is not None):
    #     rh_mass_gal = gal_row['R1/2(M*_2D)(kpc)'] / 1000 * (1 + redshift) * h
    #     xcen, ycen = gal_row['x (cMpc)'] * h, gal_row['y (cMpc)'] * h
        
    #     dx = subhalo_stars['x [cMpc/h]'] - xcen
    #     dy = subhalo_stars['y [cMpc/h]'] - ycen
    #     r_cen_mask = np.sqrt(dx**2 + dy**2) < 5 * rh_mass_gal
    #     subhalo_stars = subhalo_stars[r_cen_mask]
    # ## ---------------------------------- ##

    ## --- using 3D half-mass radius --- ##
    if remove_far_stars and (gal_row is not None):
        # print(f" remove far stars? -> {remove_far_stars}")

        rh_mass_gal = gal_row['R1/2(M*_3D)(kpc)'] / 1000 * (1 + redshift) * h
        xcen = gal_row['x (cMpc)'] * h
        ycen = gal_row['y (cMpc)'] * h
        zcen = gal_row['z (cMpc)'] * h
        
        dx = subhalo_stars['x [cMpc/h]'] - xcen
        dy = subhalo_stars['y [cMpc/h]'] - ycen
        dz = subhalo_stars['z [cMpc/h]'] - zcen
        r_cen_mask = np.sqrt(dx**2 + dy**2 + dz**2) < 5 * rh_mass_gal
        
        # print(f"Galaxy ID: {gal_id}, Host Halo ID: {host_halo_id}")
        # print(f"3D half-mass radius (cMpc): {rh_mass_gal}")
        # print(f"Number of stars before removing far stars: {len(subhalo_stars)}")
        # print(f"Number of stars after removing far stars: {np.sum(r_cen_mask)}")
        
        subhalo_stars = subhalo_stars[r_cen_mask]
    ## ---------------------------------- ##

    return subhalo_stars

def getGalStars(gal_id, host_halo_id, galCat, sCatTable, allsubs, h, remove_far_stars=False, redshift=None):

    stars_in_host = sCatTable[sCatTable['halo_idx'] == host_halo_id]

    # print(f"Galaxy ID: {gal_id}, Host Halo ID: {host_halo_id}")
    
    ## --- get subhalo_idx of this galaxy --- ##
    ## subhalo_idx is assigned in order of appearance in the substructure catalog
    all_subhalos = allsubs[allsubs['Host Halo ID'] == host_halo_id]
    if (len(all_subhalos) <= 1):
        subhalo_id = 0
    else:
        subhalo_ids = np.arange(len(all_subhalos))
        all_subhalos['subhalo_id'] = subhalo_ids
        subhalo_id = all_subhalos[all_subhalos['ID'] == gal_id]['subhalo_id']
    ## -------------------------------------- ##

    # print(f"Galaxy ID: {gal_id}, Host Halo ID: {host_halo_id}, Subhalo ID: {subhalo_id}")

    subhalo_stars = sCatTable[(sCatTable['halo_idx'] == host_halo_id) & (sCatTable['subhalo_idx'] == subhalo_id)]

    # print(f"Number of stars in the host halo: {len(stars_in_host)}")
    # print(f"Number of stars in the subhalo: {len(subhalo_stars)}")

    ## --- using 2D half-mass radius --- ##
    # if remove_far_stars:
    #     rh_mass_gal = galCat[galCat['ID'] == gal_id]['R1/2(M*_2D)(kpc)']
    #     rh_mass_gal = rh_mass_gal / 1000           # convert [kpc] -> [Mpc]
    #     rh_mass_gal = rh_mass_gal * (1 + redshift) # convert [Mpc] -> [cMpc]
    #     rh_mass_gal = rh_mass_gal * h              # convert [cMpc] -> [cMpc/h]

    #     xcen   = galCat[galCat['ID'] == gal_id]['x (cMpc)'] * h
    #     ycen   = galCat[galCat['ID'] == gal_id]['y (cMpc)'] * h
    #     dx, dy = (subhalo_stars['x [cMpc/h]'] - xcen), (subhalo_stars['y [cMpc/h]'] - ycen)
        
    #     r_cen_mask = np.sqrt(dx**2 + dy**2) < 5*rh_mass_gal
    #     subhalo_stars = subhalo_stars[r_cen_mask]
    ## ---------------------------------- ##

    ## --- using 3D half-mass radius --- ##
    if remove_far_stars:
        rh_mass_gal = galCat[galCat['ID'] == gal_id]['R1/2(M*_3D)(kpc)'] / 1000 * (1 + redshift) * h
        xcen = galCat[galCat['ID'] == gal_id]['x (cMpc)'] * h
        ycen = galCat[galCat['ID'] == gal_id]['y (cMpc)'] * h
        zcen = galCat[galCat['ID'] == gal_id]['z (cMpc)'] * h
        
        dx = subhalo_stars['x [cMpc/h]'] - xcen
        dy = subhalo_stars['y [cMpc/h]'] - ycen
        dz = subhalo_stars['z [cMpc/h]'] - zcen
        r_cen_mask = np.sqrt(dx**2 + dy**2 + dz**2) < 5 * rh_mass_gal
        if (gal_id == 64):
            print(f"Galaxy ID: {gal_id}, Host Halo ID: {host_halo_id}")
            print(f"3D half-mass radius (cMpc): {rh_mass_gal}")
            print(f"Number of stars before removing far stars: {len(subhalo_stars)}")
            print(f"Number of stars after removing far stars: {np.sum(r_cen_mask)}")
    ## ---------------------------------- ##

    return subhalo_stars


def plt_xy(gal_idx, galCat, sCatTable, allsubs, h, ax=None, remove_far_stars=False, redshift=None):

    gal   = galCat[gal_idx]
    stars = getGalStars(gal['ID'], gal['Host Halo ID'], galCat, sCatTable, allsubs, h=h, remove_far_stars=remove_far_stars, redshift=redshift)

    x_stars, y_stars = stars['x [cMpc/h]']/h, stars['y [cMpc/h]']/h
    rLum_stars = stars['rLum [Lsun]']
    x_cen, y_cen = gal['x (cMpc)'], gal['y (cMpc)']

    if ax is None: fig, ax = plt.subplots(1,1, figsize=(6,6))
    else         : fig = ax.figure

    ax.scatter(x_stars, y_stars, s=1, c='k', alpha=0.33)
    ax.scatter(x_cen, y_cen, s=50, c='r', marker='x', label='Galaxy Center')

    ax.text(x=0.05, y=0.95, s=f"Galaxy ID : {gal['ID']}", ha='left', va='top', transform=ax.transAxes)
    ax.text(x=0.05, y=0.89, s=f"Host Halo ID : {gal['Host Halo ID']}", ha='left', va='top', transform=ax.transAxes)
    ax.text(x=0.05, y=0.83, s=f"# of stars : {len(stars)}", ha='left', va='top', transform=ax.transAxes)

    dx1 = np.max(x_stars) - x_cen + 0.02
    dx2 = x_cen - np.min(x_stars) + 0.02
    dx0 = 0.08

    dy1 = np.max(y_stars) - y_cen + 0.02
    dy2 = y_cen - np.min(y_stars) + 0.02
    dy0 = 0.08

    dx = max(dx1, dx2, dx0, dy1, dy2, dy0)

    ax.set_xlim(gal['x (cMpc)'] - dx, gal['x (cMpc)'] + dx)
    ax.set_ylim(gal['y (cMpc)'] - dx, gal['y (cMpc)'] + dx)

    ax.set_xlabel('x [cMpc]')
    ax.set_ylabel('y [cMpc]')
    if ax is None:
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    ax.legend(loc='upper right')

    return fig, ax

def get_geometry_from_inertia(stars, h, xy_cen=None):
    x, y = stars['x [cMpc/h]'] / h, stars['y [cMpc/h]'] / h
    weights = stars['rLum [Lsun]']

    # 1) Find Center
    if xy_cen is not None:
        x_cen = xy_cen[0] ## should be in [cMpc]
        y_cen = xy_cen[1] ## should be in [cMpc]
    else:
        x_cen = np.average(x, weights=weights)
        y_cen = np.average(y, weights=weights)
    dx, dy = (x - x_cen), (y - y_cen)
    
    # 2) Calculate inertia tensor (Luminosity weighted)
    Ixx = np.sum(weights * dx**2) / np.sum(weights)
    Iyy = np.sum(weights * dy**2) / np.sum(weights)
    Ixy = np.sum(weights * dx * dy) / np.sum(weights)
    
    # 3) Calculate eigen decomposition to find major/minor axes and angle
    # Tensor matrix [[Ixx, Ixy], [Ixy, Iyy]]
    matrix = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    evals, evecs = np.linalg.eig(matrix)
    
    # 4) Sort eigenvalues (largest is major axis)
    sort_indices = np.argsort(evals)[::-1]
    evals = evals[sort_indices]
    evecs = evecs[:, sort_indices]
    
    a_rms = np.sqrt(evals[0])
    b_rms = np.sqrt(evals[1])
    
    # Axis ratio (q = b/a)
    q = b_rms / a_rms
    ellipticity = 1 - q
    
    # Theta (angle of major axis)
    theta = np.arctan2(evecs[1, 0], evecs[0, 0])
    
    return x_cen, y_cen, theta, q, ellipticity



def add_geometry_to_plot(gal_idx, galCat, sCatTable, allsubs, h, ax):

    x_cen, y_cen, theta, q, ellip = get_geometry_from_inertia(gal_idx, galCat, sCatTable, allsubs, h)

    # Add lines for major and minor axes
    length = 0.1 # length of the lines in cMpc
    dx_major = length * np.cos(theta)
    dy_major = length * np.sin(theta)
    dx_minor = length * np.cos(theta + np.pi/2)
    dy_minor = length * np.sin(theta + np.pi/2)
    ax.plot([x_cen - dx_major, x_cen + dx_major], [y_cen - dy_major, y_cen + dy_major], c='g', ls='--', lw=0.5, alpha=0.5)
    ax.plot([x_cen - dx_minor, x_cen + dx_minor], [y_cen - dy_minor, y_cen + dy_minor], c='g', ls='-.', lw=0.5, alpha=0.5)

def get_elliptical_radius(x, y, x0, y0, theta, q):
    dx, dy = (x - x0), (y - y0)
    
    # coordinate transformation
    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
    
    # elliptical radius (distance along major axis)
    r_ell = np.sqrt(x_rot**2 + (y_rot / q)**2)
    return r_ell


def get_radial_profile(stars, h, x0, y0, theta, q, ellip, nbins=10):

    x, y = stars['x [cMpc/h]'] / h, stars['y [cMpc/h]'] / h
    weights = stars['rLum [Lsun]']
    
    # 1) Get Elliptical Radius for each star particle
    r_ell = get_elliptical_radius(x, y, x0, y0, theta, q)
    
    # 2) Bin particles by elliptical radius
    rell_min = np.min(r_ell)
    rell_max = np.max(r_ell)
    bins = np.logspace(np.log10(rell_min), np.log10(rell_max), nbins+1)
    
    lum_in_bins, bin_edges = np.histogram(r_ell, bins=bins, weights=weights)
    area_in_bins = np.pi * q * (bin_edges[1:]**2 - bin_edges[:-1]**2)

    surface_brightness = lum_in_bins / area_in_bins
    r_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # leave only valid (positive SB) data
    valid = surface_brightness > 0
    r_data = r_centers[valid]
    sb_data = surface_brightness[valid]
    sb_data = sb_data / 1e10 # convert [Lsun/cMpc^2] -> [1e10 Lsun/cMpc^2]

    return r_data, sb_data, bin_edges, rell_min, rell_max

# def get_radial_profile_old(gal_idx, galCat, sCatTable, allsubs, h, nbins=10, xy_cen=None):

#     ## --- Get Particle Data --- ##
#     gal   = galCat[gal_idx]
#     stars = getGalStars(gal['ID'], gal['Host Halo ID'], galCat, sCatTable, allsubs, h=h)

#     x, y = stars['x [cMpc/h]']/h, stars['y [cMpc/h]']/h
#     weights = stars['rLum [Lsun]']
    
#     # 1) Get Geometry
#     x0, y0, theta, q, ellip = get_geometry_from_inertia(gal_idx, galCat, sCatTable, allsubs, h=h, xy_cen=xy_cen)
    
#     # 2) Get Elliptical Radius for each star particle
#     r_ell = get_elliptical_radius(x, y, x0, y0, theta, q)
    
#     # 3) Bin particles by elliptical radius
#     rell_min = np.min(r_ell)
#     rell_max = np.max(r_ell)
#     bins = np.logspace(np.log10(rell_min), np.log10(rell_max), nbins+1)
    
#     lum_in_bins, bin_edges = np.histogram(r_ell, bins=bins, weights=weights)
#     area_in_bins = np.pi * q * (bin_edges[1:]**2 - bin_edges[:-1]**2)

#     surface_brightness = lum_in_bins / area_in_bins
#     r_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#     # leave only valid (positive SB) data
#     valid = surface_brightness > 0
#     r_data = r_centers[valid]
#     sb_data = surface_brightness[valid]
#     sb_data = sb_data / 1e10 # convert [Lsun/cMpc^2] -> [1e10 Lsun/cMpc^2]

#     return r_data, sb_data, bin_edges, rell_min, rell_max


def plt_radial_profile(r, sb, gal_idx, galCat, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
    else:
        fig = ax.figure

    ax.scatter(r, sb, marker='x', color='k')
    ax.text(x=0.05, y=0.05, s=f"Galaxy ID : {galCat[gal_idx]['ID']}", ha='left', va='bottom', transform=ax.transAxes)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Radius along the major axis [cMpc]')
    ax.set_ylabel(r'Surface Brightness [$10^{10}$ $L_\odot$/cMpc$^2$]')

    ## adjust xlim, ylim ##
    r_min, r_max = np.min(r), np.max(r)
    sb_min, sb_max = np.min(sb), np.max(sb)
    ax.set_xlim(r_min*0.5, r_max*2)
    ax.set_ylim(sb_min*0.5, sb_max*2)

    if ax is None:
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

    return fig, ax


def log_sersic1d(r, Ie, re, n):
    """
    Logarithmic Sersic Profile
    Parameters:
    -----------
    r : array-like
        Radius along the major axis.
    Ie : float
        Intensity at the effective radius (linear scale).
    re : float
        Effective radius.
    n : float
        Sersic index.
    Returns:
    --------
    log10(I) : array-like
        Logarithm (base 10) of the intensity at radius r.
    """
    
    # get bn (as astropy does)
    bn = gammaincinv(2 * n, 0.5)

    # Sersic Profile: I(r) = Ie * exp( -bn * ( (r/re)^(1/n) - 1 ) )
    logI = np.log10(Ie) - bn * ((r/re)**(1.0/n) - 1.0) * np.log10(np.e)
    return logI
