#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yan-Ning Kuo yk545@cornell.edu
"""
import numpy as np
import scipy.stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import netCDF4 as nc
import sys
from lanczos_filter_Wills18GRL_Github import *

### Removing linear trend
def delintrd(input_x,input_y):
        shapey = len(input_y.shape)
        if (shapey==1):
                pred = np.polyval(np.polyfit(input_x,input_y,1),input_x)
                residual = input_y-pred
        elif (shapey==2):
                residual = np.empty_like(input_y)*np.nan
                for i in range(input_y.shape[1]):
                        pred = np.polyval(np.polyfit(input_x,input_y[:,i],1),input_x)
                        residual[:,i] = input_y[:,i]-pred
        elif (shapey==3):
                residual = np.empty_like(input_y)*np.nan
                for i in range(input_y.shape[1]):
                        for j in range(input_y.shape[2]):
                            if (~np.isnan(input_y[0,i,j])):
                                pred = np.polyval(np.polyfit(input_x,input_y[:,i,j],1),input_x)
                                residual[:,i,j] = input_y[:,i,j]-pred
        detrd_var = residual
        return detrd_var

### Calculating linear trend
def lintrd(input_x,input_y):
        shapey = len(input_y.shape)
        if (shapey==1):
                trd = np.empty((1,)) * np.nan
                trd = np.polyfit(input_x,input_y,1)[0]
                pred = np.polyval(np.polyfit(input_x,input_y,1),input_x)
                residual = np.empty((len(input_x),))
                residual = input_y[:]-pred[:]
                var_residual = (1/(len(input_x)-1)*np.nansum(residual*residual))
                var_x = (1/(len(input_x)-1)*np.nansum((input_x-np.nanmean(input_x))*(input_x-np.nanmean(input_x))))
                trd_std = np.sqrt((var_residual/((len(input_x)-1)*var_x)))
        elif (shapey==2):
                trd = np.empty((input_y.shape[1],)) * np.nan
                trd_std = np.empty((input_y.shape[1],)) * np.nan
                for i in range(len(input_y[0])):
                        trd[i] = np.polyfit(input_x,input_y[:,i],1)[0]
                        pred = np.polyval(np.polyfit(input_x,input_y[:,i],1),input_x)
                        residual = input_y[:,i]-pred
                        var_residual = (1/(len(input_x)-1)*np.nansum(residual*residual))
                        var_x = (1/(len(input_x)-1)*np.nansum((input_x-np.nanmean(input_x))*(input_x-np.nanmean(input_x))))
                        trd_std[i] = np.sqrt((var_residual/((len(input_x)-1)*var_x)))
        elif (shapey==3):
                trd = np.empty((input_y.shape[1],input_y.shape[2])) * np.nan
                trd_std = np.empty((input_y.shape[1],input_y.shape[2])) * np.nan
                for i in range(len(trd)):
                        for j in range(len(trd[0])):
                                if (np.isnan(input_y[0,i,j])==False):
                                    trd[i,j] = np.polyfit(input_x,input_y[:,i,j],1)[0]
                                    pred = np.polyval(np.polyfit(input_x,input_y[:,i,j],1),input_x)
                                    residual = input_y[:,i,j]-pred
                                    var_residual = (1/(len(input_x)-1)*np.nansum(residual*residual))
                                    var_x = (1/(len(input_x)-1)*np.nansum((input_x-np.nanmean(input_x))*(input_x-np.nanmean(input_x))))
                                    trd_std[i,j] = np.sqrt((var_residual/((len(input_x)-1)*var_x)))
        return trd, trd_std

### calculate annual mean of monthly mean
def mon2yr_mean(var):
        shapevar = len(var.shape)
        if (shapevar==1):
                var_yr = np.empty((int(len(var)/12),))
                for t in range(len(var_yr)):
                        str_m = 12*t
                        edd_m = 12*(t+1)
                        var_yr[t] = np.nanmean(var[str_m:edd_m],0)
        elif (shapevar==2):
                var_yr = np.empty((int(len(var)/12),var.shape[1]))
                for t in range(len(var_yr)):
                        str_m = 12*t
                        edd_m = 12*(t+1)
                        var_yr[t,:] = np.nanmean(var[str_m:edd_m,:],0)
        elif (shapevar==3):
                var_yr = np.empty((int(len(var)/12),var.shape[1],var.shape[2]))
                for t in range(len(var_yr)):
                        str_m = 12*t
                        edd_m = 12*(t+1)
                        var_yr[t,:,:] = np.nanmean(var[str_m:edd_m,:,:],0)
        elif (shapevar==4):
                var_yr = np.empty((var.shape[0],int(var.shape[1]/12),var.shape[2],var.shape[3]))
                for t in range(len(var_yr[0])):
                        str_m = 12*t
                        end_m = 12*(t+1)
                        var_yr[:,t,:,:] = np.nanmean(var[:,str_m:end_m,:,:],1)

        return var_yr

### perform multivariate linear regression
def mlr(X,var):
        Xshape = X.shape ## (time, # of predictors)
        varshape = var.shape ## (time, lat, lon)
        reg = np.empty((Xshape[1],varshape[1],varshape[2])) * np.nan
        reg_t = np.empty((Xshape[1],varshape[1],varshape[2])) * np.nan
        residual = np.empty((varshape[0],varshape[1],varshape[2])) * np.nan
        reg_prd = np.empty((Xshape[1],varshape[0],varshape[1],varshape[2])) * np.nan
        for j in range(varshape[1]):
                for k in range(varshape[2]):
                        beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),var[:,j,k])
                        residual[:,j,k] = var[:,j,k]-np.dot(beta,X.T)
                        var_beta = (np.var(residual[:,j,k]))*np.linalg.inv(np.dot(X.T,X))
                        for l in range(Xshape[1]):
                                reg[l,j,k] = beta[l]
                                reg_t[l,j,k] = beta[l]/np.sqrt(var_beta[l,l])
                                reg_prd[l,:,j,k] = beta[l]*X[:,l]
        return reg, reg_t, reg_prd, residual

### calculate area-weighted mean
def area_weighted_mean(var,varmask,lat,lon,strlat,endlat,strlon,endlon):
    # Calculating the area-weighted mean
    rad = 4.0*np.arctan(1.0)/180.0
    re = 6371220.0
    rr = re*rad
    dlon = abs(lon[2]-lon[1])*rr
    dx = dlon*np.cos(lat*rad)
    dy = np.empty((len(lat)))
    dy[0] = abs(lat[1]-lat[0])*rr
    for i in range(len(lat)-2):
        j = i+1
        dy[j] = abs((lat[i+2]-lat[i])*rr*0.5)
    dy[-1] = abs(lat[-1]-lat[-2])*rr
    #print(dy)
    area = dx*dy
    AREA = np.empty((len(lat),1))
    AREA[:,0] = area
    glo_area = np.tile(AREA,[1,len(lon)])
    counted_area = np.nansum(glo_area[strlat:endlat,strlon:endlon]*varmask[strlat:endlat,strlon:endlon])
    var_mean = np.empty((len(var)))
    for t in range(len(var)):
        sum_var = np.nansum(glo_area[strlat:endlat,strlon:endlon]*var[t,strlat:endlat,strlon:endlon]*varmask[strlat:endlat,strlon:endlon])
        var_mean[t] = sum_var/counted_area
    return var_mean

### removing seasonal cycle
def rmsc(var):
        varshape = var.shape ## (time, lat, lon)
        var_sc = np.empty((12,varshape[1],varshape[2])) * np.nan
        for i in range(12):
                varm = np.empty((int(varshape[0]/12),varshape[1],varshape[2])) * np.nan
                for j in range(int(varshape[0]/12)):
                        varm[j,:,:] = var[j*12+i,:,:]
                var_sc[i,:,:] = np.nanmean(varm,0)
        var_rmsc = var - np.tile(var_sc,(int(varshape[0]/12),1,1))
        return var_rmsc

### perform Lanczos filter with codes from lanczos_filter_Wills18GRL_Github.py
### The code of lanczos_filter_Wills18GRL_Github.py is directly from https://github.com/rcjwills/lfca/blob/master/Python/lanczos_filter.py
def lanczos_filter_YNKuo(var,T,cutoff = 120):
        varshape = var.shape ## (time, space) or (time, lat, lon)
        if (len(varshape)==1):
                varf = np.empty((varshape[0],)) * np.NAN
                p = np.polyfit(T,var,1)
                sig = var - p[0]*T - p[1]
                tmp1 = np.concatenate((np.flipud(sig),sig,np.flipud(sig)))
                tmp_filt = lanczos_filter(tmp1,1,1./cutoff)[0]
                varf = tmp_filt[int(np.size(tmp_filt)/3):2*int(np.size(tmp_filt)/3)]+p[0]*T+p[1]
        if (len(varshape)==2):
                varf = np.empty((varshape[0],varshape[1])) * np.nan
                for i in range(varshape[1]):
                        p = np.polyfit(T,var[:,i],1)
                        sig = var[:,i] - p[0]*T - p[1]
                        tmp1 = np.concatenate((np.flipud(sig),sig,np.flipud(sig)))
                        tmp_filt = lanczos_filter(tmp1,1,1./cutoff)[0]
                        varf[:,i] = tmp_filt[int(np.size(tmp_filt)/3):2*int(np.size(tmp_filt)/3)]+p[0]*T+p[1]
        if (len(varshape)==3):
                varf = np.empty((varshape[0],varshape[1],varshape[2])) * np.nan
                for i in range(varshape[1]):
                        for j in range(varshape[2]):
                                p = np.polyfit(T,var[:,i,j],1)
                                sig = var[:,i,j] - p[0]*T - p[1]
                                tmp1 = np.concatenate((np.flipud(sig),sig,np.flipud(sig)))
                                tmp_filt = lanczos_filter(tmp1,1,1./cutoff)[0]
                                varf[:,i,j] = tmp_filt[int(np.size(tmp_filt)/3):2*int(np.size(tmp_filt)/3)]+p[0]*T+p[1]
        return varf
    
