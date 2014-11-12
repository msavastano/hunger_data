# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 13:19:01 2014

@author: mike
"""

from __future__ import print_function
import requests
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandasql import *
import scipy.stats as stat
import re
import time
import codecs

hunger = pd.read_csv('ghi.csv', header = 3)
hungerCountries = pd.read_csv('2014GHIdata.csv', header = 6)
hunger=hunger.replace('<5', float(5))
hunger=hunger.convert_objects(convert_numeric=True)
hunger.index = hungerCountries['Country'] 
a = hunger.dropna(thresh=1)


def make_col_of_row_means(frame):
    #t is exclusive    
    ms = []
    for index, row in frame.iterrows():
        ms.append(np.mean(row))
    return ms
    
def interp_row_means(df):
    '''
    param df = pd.DataFrame
    
    iterates through df and sets nan values to the row mean
    
    return new df
    '''
    for index, row in df.iterrows():
        for cell in row:
            if np.isnan(cell):
                row.fillna(np.mean(row), inplace=True)
    return df
    
'''
hunger=hunger.convert_objects(convert_numeric=True)
hunger=hunger.replace('<5', 'int(5)')
hunger['mean'] = make_col_of_row_means(hunger)
hunger['Country'] = hungerCountries['Country']
a = hunger.dropna(thresh=1)
hunger.index = hungerCountries['Country'] 
b=interp_row_means(b)
bT=b.T
b['mean'] = make_col_of_row_means(b)
b.loc['totals']=bT['mean']
b.loc['totals'][-1]=np.mean(b.loc['totals'])
bT['mean'] = make_col_of_row_means(bT)


fig, axes = plt.subplots(1, 2)
ax1,ax2 = axes
ax1.set_xlabel('Countries')
ax2.set_xlabel('Countries')
ax1.set_ylabel('GHI')
ax2.set_ylabel('GHI')
ax1.set_ylim(0,40)
ax2.set_ylim(0,40)
ax1.bar(np.arange(len(hunger_high)),height=hunger_high['mean'])
ax2.bar(np.arange(len(hunger_low)),height=hunger_low['mean'])
ax1.set_xticks(np.arange(len(hunger_high)),minor=False)
ax2.set_xticks(np.arange(len(hunger_low)),minor=False)
plt.gcf().subplots_adjust(bottom=0.39)
ax1.set_xticklabels(list(hunger_high['Country']),rotation=90,ha='left')
ax2.set_xticklabels(list(hunger_low['Country']),rotation=90,ha='left')

fig,axes = plt.subplots(1,1)
ax = axes
ax.plot(hunger_high_no_na.T)
ax.legend(hunger_high_no_na.T, labels=hunger_high_no_na_c['Country'],loc='lower left', prop={'size':7}, ncol=2)
ax.set_xticks(np.arange(len(hunger_high_no_na.T)),minor=False)
ax.set_xticklabels(list(hunger_high_no_na_c.columns.values),rotation=0,ha='left')
ax.set_xlabel('Years')
ax.set_ylabel('GHI')
ax.set_title('GHI 1988-2013 : Highest Countries')
'''