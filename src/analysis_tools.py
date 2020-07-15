#!/usr/bin/env python3

import glob,sys,re

from scipy.optimize import curve_fit

def hockeystick(x,a,y_0,x_knee):
    return (x<x_knee)*(y_0 + a*x) + (x>=x_knee)*(y_0 + a*x_knee)

def fit_log_conv(xdata,ydata,**kwargs):

    popt, pcov = curve_fit(hockeystick,xdata,ydata,**kwargs)

    a = popt[0]
    y_0 = popt[1]
    x_knee = popt[2]

    return a,y_0,x_knee


def get_simfile_prop(basestr,return_None=False):

    simfile = glob.glob(basestr+'*')

    if len(simfile)==0:
        print('No file found!')
        if return_None:
            return None,None
        else:
            sys.exit()
    elif len(simfile) > 1:
        print('Multiple files found:')
        for k,simf in enumerate(simfile):
            print('[' + str(k+1) + ']  ' + simf)

        filenum = input('Please choose the file to use by its number.')
        try:
            filenum = int(filenum) - 1
        except:
            print('Could not parse number!')
            sys.exit()

        simfile = simfile[filenum]
    else:
        simfile = simfile[0]

    timestamp_regex = re.compile('[\-T:\.0-9]+(?=\.[a-z])')

    timestamp = timestamp_regex.findall(simfile)[0]

    return simfile, timestamp
