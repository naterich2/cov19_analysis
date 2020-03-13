#!/usr/bin/env python
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import curve_fit

"""Analysis on COVID-19

"""
data = pd.read_csv('/home/nrichman/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
glob = data.sum(axis=0).iloc[2:]
glob_no_china = (glob - np.sum(data[data.loc[:,'Country/Region'] == 'China'],axis=0)[4:]).values.astype('float')
def exp_fun(x,a,b):#,x_0):
    return a*np.exp(b*(x))# - x_0))
def logistic_fun(x,L,k,x_0):
    return L/(1 + np.exp(-k*(x - x_0)))
plt.style.use('ggplot')
fig,ax = plt.subplots(1,1,figsize=(10,10))
line, = ax.plot([],[])
exp_fit, = ax.plot([],[])
log_fit, = ax.plot([],[])
params_exp = []
params_log = []
for day in range(20,glob_no_china.shape[0]):
    x = np.arange(0,day)
    popt_exp,pcov_exp = curve_fit(exp_fun,x,glob_no_china[0:day],bounds=([.5,0],[np.inf,.5]))
    popt_log,pcov_log = curve_fit(logistic_fun,x,
            glob_no_china[0:day],
            p0=[np.max(glob_no_china[0:day])*2,2e-1,0],
            bounds=([1,0,0],[.6*6000000000,1,1000]),
            maxfev=5000)
    params_exp += popt_exp
    params_log += popt_log

def animate(t):
    day = t + 20
    x_plot = np.linspace(0,day*1.2)
    e = params_exp[t]
    l = params_log[t]
    y_exp = e[0]*np.exp(e[1]*(x_plot))#-popt_exp[2]))
    y_log = l[0]/(1 + np.exp(-l[1]*(x_plot-l[2])))
    line.set_data(x,glob_no_china[0:day])
    exp_fit.set_data(x_plot,y_exp)
    log_fit.set_data(x_plot,y_log)
    ax.set_xlim([0,np.max(x_plot)*1.1])
    ax.set_ylim([0,np.max(glob_no_china[0:day])*2])

my_animation = animation.FuncAnimation(fig,animate,frames=(glob_no_china.shape[0]-20),interval=400,repeat_delay=1000)
ax.legend(['Data','Exponential fit','Logistic fit'])
plt.show()

#def main(args):


#if __name__ = 'main':
#
#    parser = argparse.ArgumentParser(description=__doc__)
#    parser.add_argument('input'.help='input file')
#    main(parser.parse_args())
