# Read Telemetry
# Advanced Signal Processing (TM12005)
# Made by: M.S. van Schie (m.vanschie@erasmusmc.nl) & M.M. de Boer (m.m.deboer@erasmusmc.nl)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd


#%%
leads = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

def read_ecg_mat(path, plotresult=True):
    # open datafile
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    ecg = data['ecg'].sig[:,leads.index('II')]

    fs = data['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data['ecg'].start_vec)

    nSamples = data['ecg'].sig.shape[0]
    t = pd.date_range(
        start=t0,
        periods=nSamples,
        freq=pd.Timedelta(seconds=1/fs)
    )
        
    # Plot signal in time domain
    if plotresult:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t, ecg)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ECG (mV)")
        ax.set_title("Raw ECG Signal")
        plt.show()
    
    return ecg, fs, t


#%%

# ecg, fs, t = read_ecg_mat(r"path\file.mat", plotresult=True)
