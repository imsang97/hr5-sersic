import os
import sys
import glob
import time

from tqdm import tqdm
import multiprocessing as mp

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


## --- import my functions --- ##
# from myFuncs_3_calcSersicN import getGalStars
from myFuncs_3_calcSersicN import getGalStars_grouped
from myFuncs_3_calcSersicN import plt_xy
from myFuncs_3_calcSersicN import get_geometry_from_inertia
from myFuncs_3_calcSersicN import get_elliptical_radius
from myFuncs_3_calcSersicN import get_radial_profile
from myFuncs_3_calcSersicN import plt_radial_profile
from myFuncs_3_calcSersicN import log_sersic1d
## -------------------------- ##


## --- Settings --- ##
global h 
h    = 0.684
nCPU = 6

snap_num = 96
snap_str = str(snap_num).zfill(3)

if   snap_num == 64: redshift = 4.582214
elif snap_num == 84: redshift = 3.111296
elif snap_num == 96: redshift = 2.446762

remove_far_stars = True


gal_cat_pwd   = f"/md/imsang/ODIN_obs_LSS/HR5_cats/all_galaxy_catalog/"
gal_cat_fname = f"All_galaxy_catalog_Mstar_min_2.137424e+08_pured_{snap_str}.txt"

all_sub_pwd   = gal_cat_pwd
all_sub_fname = f"substructure_catalogue_{snap_str}.txt"

stars_pwd   = f"/md/imsang/ODIN_obs_LSS/HR5_cats/star_particles/sn{snap_str}/"
stars_fname = f"shalo_stars_00{snap_str}_withMags.fits"
## ---------------- ##



###############
## Main Code ##
###############

## Global Variables ##
global galCat
global allsubs
global sCatTable

## -- Read HR5 Galaxy Catalog -- ##
print(f"\n --- Reading Galaxy Catalog --- ")
print(f" * pwd       : {gal_cat_pwd}")
print(f" * file name : {gal_cat_fname}")

galCat = Table.read(gal_cat_pwd + gal_cat_fname, format='csv')
print(f" => # of galaxies : {len(galCat)} ")

galCat['ID'] = galCat['ID'].astype('int64')
galCat['Host Halo ID'] = galCat['Host Halo ID'].astype('int64')

# print(f"\n\n ** Show the first 10 rows of the galaxy catalog ** ")
# display(galCat[:10]['ID','Host Halo ID', 'x (cMpc)', 'y (cMpc)', 'z (cMpc)', 'Mstar (Msun)'])
## ------------------------------ ##


## -- Read HR5 Substructure Catalog -- ##
print(f"\n --- Reading Substructure Catalog --- ")
print(f" * pwd       : {all_sub_pwd}")
print(f" * file name : {all_sub_fname}")

allsubs = pd.read_csv(all_sub_pwd + all_sub_fname, sep='\s+', header=1, usecols=[0,1])

allsubs.rename(columns={'Host':'Host Halo ID'}, inplace=True)
allsubs = Table().from_pandas(allsubs)
print(f" => # of substructures : {len(allsubs)} ")

# print(f"\n\n ** Show the first 10 rows of the substructure catalog ** ")
# display(allsubs[:10])
## ----------------------------------- ##



## -- Read Star Particle Data -- ##
print(f"\n --- Reading Star Particle Data --- ")
print(f" * pwd       : {stars_pwd}")
print(f" * file name : {stars_fname}")

with fits.open(stars_pwd + stars_fname) as hdul:
    data = hdul[1].data
    # print(data.columns)

    id          = data['id']
    x           = data['x [cMpc/h]']
    y           = data['y [cMpc/h]']
    z           = data['z [cMpc/h]']
    # mass        = data['mass [Msun/h]']
    rmag        = data['rLum [Lsun]']
    halo_idx    = data['halo_idx']
    subhalo_idx = data['subhalo_idx']

# sCatTable = Table([id, x, y, mass, rmag, halo_idx, subhalo_idx], names=('id', 'x [cMpc/h]', 'y [cMpc/h]', 'mass [Msun/h]', 'rLum [Lsun]', 'halo_idx', 'subhalo_idx'))
# sCatTable = Table([id, x, y, rmag, halo_idx, subhalo_idx], names=('id', 'x [cMpc/h]', 'y [cMpc/h]', 'rLum [Lsun]', 'halo_idx', 'subhalo_idx'))

sCatTable = Table([id, x, y, z, rmag, halo_idx, subhalo_idx],
                   names=('id', 'x [cMpc/h]', 'y [cMpc/h]', 'z [cMpc/h]', 'rLum [Lsun]', 'halo_idx', 'subhalo_idx'))

print(f"\n => # of stars : {len(sCatTable)} ")

sCat_df = sCatTable.to_pandas()
grouped_stars = sCat_df.groupby(['halo_idx', 'subhalo_idx'])

del sCatTable

# ## indexing for faster query..
# from collections import defaultdict

# star_index_map = defaultdict(list)
# for i, (h_idx, sh_idx) in enumerate(zip(sCatTable['halo_idx'], sCatTable['subhalo_idx'])):
#     star_index_map[(h_idx, sh_idx)].append(i)

# global_star_map = dict(star_index_map)

## ------------------------------- ##



# print(f"\n\n ** Show the first 10 rows of the star catalog ** ")
# display(sCatTable[0:10])


##############################
## Fit to 1D Sersic Profile ##
##############################

def worker_fit(idx):
    """
    for each galaxy
    """
    try:
        # x0, y0, theta_det, q_det, ellip_det = get_geometry_from_inertia(idx, galCat, sCatTable, allsubs, h)
        
        xcen, ycen = galCat[idx]['x (cMpc)'], galCat[idx]['y (cMpc)']
        
        # stars = getGalStars(galCat[idx]['ID'], galCat[idx]['Host Halo ID'], galCat, sCatTable, allsubs, h=h, remove_far_stars=remove_far_stars, redshift=redshift)
        # print(f"Galaxy idx: {idx}, # of stars: {len(stars)}")

        all_sub_host = allsubs[allsubs['Host Halo ID'] == galCat[idx]['Host Halo ID']]
        stars = getGalStars_grouped(galCat[idx]['ID'], galCat[idx]['Host Halo ID'], 
                                    grouped_stars, all_sub_host, h=h, 
                                    remove_far_stars=remove_far_stars, redshift=redshift, gal_row=galCat[idx])
        # print(f"Galaxy idx: {idx}, # of stars: {len(stars)}")


        # x0, y0, theta_det, q_det, ellip_det = get_geometry_from_inertia(stars, h, xy_cen=(xcen, ycen))
        x0, y0, theta_det, q_det, ellip_det = get_geometry_from_inertia(stars, h)
        r, sb, bins, rell_min, rell_max = get_radial_profile(stars, h, x0, y0, theta_det, q_det, ellip_det, nbins=10)

        #print(x0, y0, theta_det, q_det, ellip_det)
        #print(r, sb)

        p0     = [np.max(sb), np.median(r), 1.0] # Ie, re, n

        ## --- set the fitting boundary --- ##
        # bounds = ([1e-10, 1e-5, 0.1], [np.inf, np.inf, 10.0]) # Lower, Upper
        low_re, max_re = rell_min, rell_max
        bounds = ([1e-10, low_re, 0.1], [np.inf, max_re, 10.0]) # Lower, Upper
        ## --------------------------------- ##

        popt, pcov = curve_fit(log_sersic1d, r, np.log10(sb), p0=p0, bounds=bounds)
        popv = np.sqrt(np.diag(pcov))
        # return idx, popt, popv
        return idx, popt, popv, x0, y0, theta_det, q_det
    except Exception:
        # return idx, [np.nan]*3, [np.nan]*3
        return idx, [np.nan]*3, [np.nan]*3, np.nan, np.nan, np.nan, np.nan

# print(f"\n >>> Now Fitting to 1D Sersic Profile... ")


# --- Test with Random Sample --- ##
# galCat = galCat[0:1000] # For testing..
# n_rand   = 1000
# rand_idx = np.random.choice(len(galCat), size=n_rand, replace=False)
# galCat   = galCat[rand_idx]
## ------------------------------- ##

if __name__ == "__main__":
    s_time = time.time()

    target_indices = range(len(galCat))
    print(f"\n >>> Now Fitting to 1D Sersic Profile using {nCPU} cores... ")

    # for results
    results_popt  = np.full((len(galCat), 3), np.nan)
    results_popv  = np.full((len(galCat), 3), np.nan)
    results_x0    = np.full(len(galCat), np.nan)
    results_y0    = np.full(len(galCat), np.nan)
    results_theta = np.full(len(galCat), np.nan)
    results_q     = np.full(len(galCat), np.nan)

    # multi-proc
    with mp.Pool(processes=nCPU) as pool:
        # for idx, popt, popv in tqdm(pool.imap_unordered(worker_fit, target_indices), total=len(target_indices)):
        for idx, popt, popv, x0, y0, theta_det, q_det in tqdm(pool.imap_unordered(worker_fit, target_indices), total=len(target_indices)):
            results_popt[idx] = popt
            results_popv[idx] = popv
            results_x0[idx] = x0
            results_y0[idx] = y0
            results_theta[idx] = theta_det
            results_q[idx] = q_det


    print(f"\n >>> Fitting is done! Saving results... ")

    cat_to_save = galCat['ID','Host Halo ID', 'Mstar (Msun)', 'R1/2(r_2D)']
    
    cat_to_save['best_Ie']     = results_popt[:, 0]
    cat_to_save['best_Ie_err'] = results_popv[:, 0]
    cat_to_save['best_re']     = results_popt[:, 1]
    cat_to_save['best_re_err'] = results_popv[:, 1]
    cat_to_save['best_n']      = results_popt[:, 2]
    cat_to_save['best_n_err']  = results_popv[:, 2]

    cat_to_save['best_q'] = results_q
    cat_to_save['best_theta'] = results_theta
    cat_to_save['best_x0'] = results_x0
    cat_to_save['best_y0'] = results_y0

    sav_pwd   = gal_cat_pwd
    sav_fname = f"All_galaxy_catalog_Mstar_min_2.137424e+08_pured_{snap_str}_withSersicFit_onlyFewCols_Multi.txt"

    print(f" * Save path : {sav_pwd + sav_fname}")
    cat_to_save.write(sav_pwd + sav_fname, format='ascii.fixed_width_two_line', overwrite=False)

    e_time = time.time()
    print(f"\n >>> Total Elapsed Time : {(e_time - s_time)/60:.2f} minutes")

# galCat = galCat[0:10] # For testing..

# popt_list = []
# popv_list = []

# for check_idx in tqdm(range(len(galCat))):
#     x0, y0, theta_det, q_det, ellip_det = get_geometry_from_inertia(check_idx, galCat, sCatTable, allsubs, h)
#     r, sb, bins = get_radial_profile(gal_idx=check_idx, galCat=galCat, sCatTable=sCatTable, allsubs=allsubs, h=h)

#     # Initial guess
#     p0 = [np.max(sb),   # for Ie
#           np.median(r), # for re
#           1.0           # for n 
#     ]

#     popt, pcov = curve_fit(log_sersic1d, r, np.log10(sb), p0=p0, bounds=bounds)
    
#     popt_list.append(popt)
#     popv_list.append(np.sqrt(np.diag(pcov)))


# print(f"\n >>> Fitting is done! ")

# ## Save the results

# cat_to_save = galCat['ID','Host Halo ID', 'Mstar (Msun)', 'R1/2(r_2D)']
# cat_to_save['best_Ie']     = np.array(popt_list)[:,0]
# cat_to_save['best_Ie_err'] = np.array(popv_list)[:,0]
# cat_to_save['best_re']     = np.array(popt_list)[:,1]
# cat_to_save['best_re_err'] = np.array(popv_list)[:,1]
# cat_to_save['best_n']      = np.array(popt_list)[:,2]
# cat_to_save['best_n_err']  = np.array(popv_list)[:,2]

# sav_pwd   = f"/md/imsang/ODIN_obs_LSS/HR5_cats/all_galaxy_catalog/"
# sav_fname = f"All_galaxy_catalog_Mstar_min_2.137424e+08_pured_{snap_str}_withSersicFit.txt"

# print(f"\n --- Saving the results --- ")
# print(f" * pwd       : {sav_pwd}")
# print(f" * file name : {sav_fname}")

# cat_to_save.write(sav_pwd + sav_fname, format='ascii.fixed_width_two_line', overwrite=True)
