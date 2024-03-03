!#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:11:24 2024

@author: mnsgsty
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_reliability_all_metrics(flows_filename_1, metric_1, flows_filename_2, metric_2, flows_filename_3, metric_3,
                                 flows_filename_4, metric_4, flows_filename_5, metric_5):
    '''
    Function that obtains the reliability for all hours of the year for each file (metric of complementarity)
    and plots the result for all metrics
    
   
    Parameters
    ----------
    flows_filename_i: TYPE, required
        DESCRIPTION, filenames (string) for each metric, currently receives five files
    metric_i: TYPE, required
        DESCRIPTION, metric from the corresponding i-th filename, for labelling in the plots

    Returns
    -------
    Plot for the metric specified
    
    ''' 
    flows_1 = pd.read_csv(flows_filename_1)
    flows_2 = pd.read_csv(flows_filename_2)
    flows_3 = pd.read_csv(flows_filename_3)
    flows_4 = pd.read_csv(flows_filename_4)
    flows_5 = pd.read_csv(flows_filename_5)

    rel_1 = np.array(flows_1['Reliability'])
    x_1 = np.array(flows_1.index)
    y_1 = np.flip(np.sort(rel_1))

    data_1 = pd.DataFrame({'x': x_1,
                         'y': y_1})

    rel_2 = np.array(flows_2['Reliability'])
    x_2 = np.array(flows_2.index)
    y_2 = np.flip(np.sort(rel_2))

    data_2 = pd.DataFrame({'x': x_2,
                         'y': y_2})

    rel_3 = np.array(flows_3['Reliability'])
    x_3 = np.array(flows_3.index)
    y_3 = np.flip(np.sort(rel_3))

    data_3 = pd.DataFrame({'x': x_3,
                         'y': y_3})

    rel_4 = np.array(flows_4['Reliability'])
    x_4 = np.array(flows_4.index)
    y_4 = np.flip(np.sort(rel_4))

    data_4 = pd.DataFrame({'x': x_4,
                         'y': y_4})
    
    rel_5 = np.array(flows_5['Reliability'])
    x_5 = np.array(flows_5.index)
    y_5 = np.flip(np.sort(rel_5))

    data_5 = pd.DataFrame({'x': x_5,
                         'y': y_5})


    fig, ax = plt.subplots()
    ax.plot(data_1['x'], data_1['y'], color='#E76F51', linestyle='-', alpha=1, label=f"{metric_1} sites")
    ax.plot(data_2['x'], data_2['y'], color='#779B6C', linestyle='-', alpha=1, label=f"{metric_2} sites")
    ax.plot(data_3['x'], data_3['y'], color='#E9C46A', linestyle='-', alpha=1, label=f"{metric_3} sites")
    ax.plot(data_4['x'], data_4['y'], color='#2A9D8F', linestyle='-', alpha=1, label=f"{metric_4} sites")
    ax.plot(data_5['x'], data_5['y'], color='#264653', linestyle='-', alpha=1, label=f"{metric_5} sites")

    ax.set_xlim(0,8760)
    ax.set_ylim(0,1.1)

    ax.set_ylabel("Share of met demand")
    # ax.set_xlabel("Share of hours in the year (%)")
    ax.set_xlabel("Hour")
    ax.legend()

    ax.set_xticks(ticks=[0, 2190, 4380, 6570, 8760], labels=["0", "2190", "4380", "6570", "8760"])
    # plt.xticks(ticks=[0, 876, 1752, 2628, 3504, 4380, 5256, 6132, 7008, 7884, 8760], labels=["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"])
    ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.25,0,0))
    plt.grid(visible=True, alpha=0.5)


    return fig

def plot_reliability_alphas(flows_filename_1, flows_filename_2, flows_filename_3,
                            flows_filename_4, flows_filename_5, flows_filename_6):
    '''
    Function that obtains the reliability for all hours of the year for each file (metric of complementarity)
    and plots the result for all metrics
    
   
    Parameters
    ----------
    flows_filename_i: TYPE, required
        DESCRIPTION, filenames (string) for each metric, currently receives five files
    metric_i: TYPE, required
        DESCRIPTION, metric from the corresponding i-th filename, for labelling in the plots

    Returns
    -------
    Plot for the metric specified
    
    ''' 
    flows_1 = pd.read_csv(flows_filename_1)
    flows_2 = pd.read_csv(flows_filename_2)
    flows_3 = pd.read_csv(flows_filename_3)
    flows_4 = pd.read_csv(flows_filename_4)
    flows_5 = pd.read_csv(flows_filename_5)
    flows_6 = pd.read_csv(flows_filename_6)

    rel_1 = np.array(flows_1['Reliability'])
    x_1 = np.array(flows_1.index)
    y_1 = np.flip(np.sort(rel_1))

    data_1 = pd.DataFrame({'x': x_1,
                         'y': y_1})

    rel_2 = np.array(flows_2['Reliability'])
    x_2 = np.array(flows_2.index)
    y_2 = np.flip(np.sort(rel_2))

    data_2 = pd.DataFrame({'x': x_2,
                         'y': y_2})

    rel_3 = np.array(flows_3['Reliability'])
    x_3 = np.array(flows_3.index)
    y_3 = np.flip(np.sort(rel_3))

    data_3 = pd.DataFrame({'x': x_3,
                         'y': y_3})

    rel_4 = np.array(flows_4['Reliability'])
    x_4 = np.array(flows_4.index)
    y_4 = np.flip(np.sort(rel_4))

    data_4 = pd.DataFrame({'x': x_4,
                         'y': y_4})
    
    rel_5 = np.array(flows_5['Reliability'])
    x_5 = np.array(flows_5.index)
    y_5 = np.flip(np.sort(rel_5))

    data_5 = pd.DataFrame({'x': x_5,
                         'y': y_5})
    
    rel_6 = np.array(flows_6['Reliability'])
    x_6 = np.array(flows_6.index)
    y_6 = np.flip(np.sort(rel_6))

    data_6 = pd.DataFrame({'x': x_6,
                         'y': y_6})


    fig, ax = plt.subplots()
    ax.plot(data_1['x'], data_1['y'], color='#277588', linestyle='-', alpha=0.3)
    ax.plot(data_2['x'], data_2['y'], color='#cf3759', linestyle='-', alpha=0.3)
    ax.plot(data_3['x'], data_3['y'], color='#277588', linestyle='-', alpha=0.6)
    ax.plot(data_4['x'], data_4['y'], color='#cf3759', linestyle='-', alpha=0.6)
    ax.plot(data_5['x'], data_5['y'], color='#277588', linestyle='-', alpha=1, label="ARD sites")
    ax.plot(data_6['x'], data_6['y'], color='#cf3759', linestyle='-', alpha=1, label="MVAR sites")

    ax.set_xlim(0,8760)
    ax.set_ylim(0,1.1)

    ax.set_ylabel("Share of met demand")
    # ax.set_xlabel("Share of hours in the year (%)")
    ax.set_xlabel("Hour")
    ax.legend()

    ax.set_xticks(ticks=[0, 2190, 4380, 6570, 8760], labels=["0", "2190", "4380", "6570", "8760"])
    # plt.xticks(ticks=[0, 876, 1752, 2628, 3504, 4380, 5256, 6132, 7008, 7884, 8760], labels=["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"])
    ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.25,0,0))
    plt.grid(visible=True, alpha=0.5)


    return fig


def find_max_res_demand_day(filename):
    flows = pd.read_csv(filename)
    unmet = flows['Unmet_load']
    days_total_unmet = []
    total_df = pd.DataFrame()
    while not unmet.empty:
        day_unmet = unmet.iloc[0:24].sum()
        days_total_unmet.append(day_unmet)
        series = pd.Series(days_total_unmet)
        unmet.drop(index=unmet.index[0:24], axis=0, inplace=True)
        
    total_df = pd.concat([total_df, series], ignore_index=True)
    max_day = total_df.idxmax()
     
    return max_day

def find_peak_res_demand_day(filename):
    flows = pd.read_csv(filename)
    peak_residual = flows['Unmet load'].max()
    prd_hour = flows['Unmet load'].idxmax()
    return peak_residual


def find_max_gen_day(filename):
    flows = pd.read_csv(filename)
    total_gen = pd.DataFrame()
    total_gen = flows['Wind'] + flows['PV']
    days_total_gen = []
    total_df = pd.DataFrame()
    while not total_gen.empty:
        day_gen = total_gen.iloc[0:24].sum()
        days_total_gen.append(day_gen)
        series = pd.Series(days_total_gen)
        total_gen.drop(index=total_gen.index[0:24], axis=0, inplace=True)
        
    total_df = pd.concat([total_df, series], ignore_index=True)
       
    max_day = total_df.idxmax()
    
    return max_day


def plot_max_res_demand_day(gen_pv_filename, gen_wind_filename, demand_filename, max_day=0, metric="ARD"):
    '''
    Function that obtains day generation and load profile dataa for maximum residual demand day, and
    stackplots generation for that day. This function has colormaps harcoded for each metric and can
    plot ARD, PRD, MVAR, and TVAR
    
   
    Parameters
    ----------
    gen_pv_filename : TYPE, required
        DESCRIPTION, PV generation file from optimisation of the metric to plot
    gen_wind_filename : TYPE, required
         DESCRIPTION, Wind  generation file from optimisation of the metric to plot
    demand_filename : TYPE, required
        DESCRIPTION, load profile with datetime index, in GW
    max_day : TYPE, optional (int)
        DESCRIPTION, Output of day of the year with maximum residual demand
        from find_max_res_demand_day function. The default is 0 (first of January).
    metric : TYPE, optional (string)
        DESCRIPTION, Complementarity metric whose maximum residual demand day will be plotted
    
    Returns
    -------
    Plot for the metric specified
    
    '''   
    gen_pv_df = pd.read_csv(gen_pv_filename)
    gen_pv_df.drop(['fecha'], axis=1, inplace=True)
    gen_wind_df = pd.read_csv(gen_wind_filename)
    gen_wind_df.drop(['fecha'], axis=1, inplace=True)

    load_df = pd.read_csv(demand_filename)
    load_df['demand'] = load_df['demand']/1000
    day_pv_profile = pd.DataFrame()
    day_pv_profile = gen_pv_df.iloc[(max_day * 24): (max_day*24)+24, :]
    day_pv_profile.reset_index(inplace=False)
    
    day_wind_profile = pd.DataFrame()
    day_wind_profile = gen_wind_df.iloc[(max_day * 24): (max_day*24)+24, :]
    day_wind_profile.reset_index(inplace=False)

    day_load_profile = load_df.iloc[(max_day * 24): (max_day*24)+24, :]
    
    col_names = ['SON', 'MUL', 'BCS', 'CHIH', 'JAL', 'GUE', 'TAM', 'PUE', 'VER',
                  'OAX', 'CHIA', 'YUC']
    wind_other = pd.DataFrame()
    pv_other = pd.DataFrame()
    day_pv = day_pv_profile.copy()
    day_wind = day_wind_profile.copy()
    for loc in col_names:
        
        if (day_wind[f'{loc}'].max() <= 3):
            wind_other = pd.concat([wind_other, day_wind[f'{loc}']], axis=1)
            day_wind.drop([f'{loc}'], axis=1, inplace=True)
            
        if (day_pv[f'{loc}'].max() <= 0.7):
            pv_other = pd.concat([pv_other, day_pv[f'{loc}']], axis=1)
            day_pv.drop([f'{loc}'], axis=1, inplace=True)
    
    wind_other['Other wind'] = wind_other.sum(axis=1) 
    day_wind = pd.concat([day_wind, wind_other['Other wind']], axis=1)   
    pv_other['Other'] = pv_other.sum(axis=1)
    day_pv = pd.concat([day_pv, pv_other['Other']], axis=1)
    
    index_day = list(range(0,24,1))
    
    if metric == "ARD":
        # # Labels for ARD
        label = ['SON', 'MUL', 'BCS', 'YUC', 'Other wind', 'SON', 'MUL', 'TAM', 'YUC', 'Other PV']
        cmap= ['#030455', '#023E8A', '#015BA0', '#0077B6','#0083BD', '#FA6E23', '#FC8B29', '#FD992C', '#FEA72F','#F2CB58']
    
        fig, ax = plt.subplots()
        ax.plot(index_day, day_load_profile['demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, day_wind.T, day_pv.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
        
        
    elif metric == "PRD":
        # Labels and colormap for PRD
        label = ['MUL', 'BCS', 'CHIH', 'TAM', 'CHIA', 'Other wind', 'YUC', 'Other PV']
        cmap =  ['#030455', '#023E8A', '#015BA0', '#0077B6','#0083BD', '#00A5D0', '#FA6E23', '#F2CB58']
    
        fig, ax = plt.subplots()
        ax.plot(index_day, day_load_profile['demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, day_wind.T, day_pv.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
        
        
    elif metric == "TVAR":
        # # # Labels for Minimized total variability in profile
        label = ['MUL', 'BCS', 'TAM', 'Other wind', 'YUC', 'Other PV']
        cmap = ['#030455', '#023E8A', '#015BA0', '#0077B6',
                '#FA6E23', '#FC8B29',]

        fig, ax = plt.subplots()
        ax.plot(index_day, day_load_profile['demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, day_wind.T, day_pv.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
        
    elif metric == "MVAR":
        # # # Labels for Minimized maximum variability in profile
        label = ['SON', 'MUL', 'BCS', 'TAM', 'Other wind', 'SON', 'MUL', 'YUC',]
        cmap = ['#030455', '#023E8A', '#015BA0', '#0077B6','#0083BD',
                '#FA6E23', '#FC8B29', '#FD992C']
        day_pv = day_pv.iloc[:,[0,2,1]]
        fig, ax = plt.subplots()
        ax.plot(index_day, day_load_profile['demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, day_wind.T, day_pv.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
    
    return fig


def plot_peak_res_demand_day(gen_pv_filename, gen_wind_filename, demand_filename, max_day=0, metric="ARD"):
    '''
    Function that obtains day generation and load profile dataa for peak residual demand hour of the year, and
    stackplots generation for that day. This function has colormaps harcoded for each metric and can
    plot ARD, PRD, MVAR, and TVAR
    
   
    Parameters
    ----------
    gen_pv_filename : TYPE, required
        DESCRIPTION, PV generation file from optimisation of the metric to plot
    gen_wind_filename : TYPE, required
         DESCRIPTION, Wind  generation file from optimisation of the metric to plot
    demand_filename : TYPE, required
        DESCRIPTION, load profile with datetime index, in GW
    max_day : TYPE, optional (int)
        DESCRIPTION, Output of day of the year with maximum residual demand
        from find_max_res_demand_day function. The default is 0 (first of January).
    metric : TYPE, optional (string)
        DESCRIPTION, Complementarity metric whose maximum residual demand day will be plotted
    
    Returns
    -------
    Plot for the metric specified
    
    '''   
    gen_pv_df = pd.read_csv(gen_pv_filename)
    gen_pv_df.drop(['fecha'], axis=1, inplace=True)
    gen_wind_df = pd.read_csv(gen_wind_filename)
    gen_wind_df.drop(['fecha'], axis=1, inplace=True)

    load_df = pd.read_csv(demand_filename)
    day_pv_profile = pd.DataFrame()
    day_pv_profile = gen_pv_df.iloc[(max_day * 24): (max_day*24)+24, :]
    day_pv_profile.reset_index(inplace=False)
    
    day_wind_profile = pd.DataFrame()
    day_wind_profile = gen_wind_df.iloc[(max_day * 24): (max_day*24)+24, :]
    day_wind_profile.reset_index(inplace=False)

    day_load_profile = load_df.iloc[(max_day * 24): (max_day*24)+24, :]
    
    col_names = ['SON', 'MUL', 'BCS', 'CHIH', 'JAL', 'GUE', 'TAM', 'PUE', 'VER',
                  'OAX', 'CHIA', 'YUC']
    wind_other = pd.DataFrame()
    pv_other = pd.DataFrame()
    day_pv = day_pv_profile.copy()
    day_wind = day_wind_profile.copy()
    for loc in col_names:
        
        if (day_wind[f'{loc}'].max() <= 3):
            wind_other = pd.concat([wind_other, day_wind[f'{loc}']], axis=1)
            day_wind.drop([f'{loc}'], axis=1, inplace=True)
            
        if (day_pv[f'{loc}'].max() <= 0.7):
            pv_other = pd.concat([pv_other, day_pv[f'{loc}']], axis=1)
            day_pv.drop([f'{loc}'], axis=1, inplace=True)
    
    wind_other['Other wind'] = wind_other.sum(axis=1) 
    day_wind = pd.concat([day_wind, wind_other['Other wind']], axis=1)   
    pv_other['Other'] = pv_other.sum(axis=1)
    day_pv = pd.concat([day_pv, pv_other['Other']], axis=1)
    
    index_day = list(range(0,24,1))
    
    if metric == "ARD":
        # # Labels for ARD
        label = ['SON', 'MUL', 'BCS', 'YUC', 'Other wind', 'SON', 'MUL', 'TAM', 'YUC', 'Other PV']
        cmap= ['#030455', '#023E8A', '#015BA0', '#0077B6','#0083BD', '#FA6E23', '#FC8B29', '#FD992C', '#FEA72F','#F2CB58']
    
        fig, ax = plt.subplots()
        ax.plot(index_day, day_load_profile['demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, day_wind.T, day_pv.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
        
        
    elif metric == "PRD":
        # Labels and colormap for PRD
        label = ['MUL', 'BCS', 'CHIH', 'TAM', 'CHIA', 'Other wind', 'YUC', 'Other PV']
        cmap =  ['#030455', '#023E8A', '#015BA0', '#0077B6','#0083BD', '#00A5D0', '#FA6E23', '#F2CB58']
    
        fig, ax = plt.subplots()
        ax.plot(index_day, day_load_profile['demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, day_wind.T, day_pv.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
        
        
    elif metric == "TVAR":
        # # # Labels for Minimized total variability in profile
        label = ['MUL', 'BCS', 'TAM', 'Other wind', 'YUC', 'Other PV']
        cmap = ['#030455', '#023E8A', '#015BA0', '#0077B6',
                '#FA6E23', '#FC8B29',]

        fig, ax = plt.subplots()
        ax.plot(index_day, day_load_profile['demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, day_wind.T, day_pv.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
        
    elif metric == "MVAR":
        # # # Labels for Minimized maximum variability in profile
        label = ['SON', 'MUL', 'BCS', 'TAM', 'Other wind', 'SON', 'MUL', 'YUC',]
        cmap = ['#030455', '#023E8A', '#015BA0', '#0077B6','#0083BD',
                '#FA6E23', '#FC8B29', '#FD992C']
        day_pv = day_pv.iloc[:,[0,2,1]]
        fig, ax = plt.subplots()
        ax.plot(index_day, day_load_profile['demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, day_wind.T, day_pv.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
    
    return fig


def plot_average_profiles(pv_gen_filename, wind_gen_filename, load_filename, metric="ARD"):
    '''
    Function that obtains average daily generation and load profiles, and stackplots
    each location's average profile. Currently hardcoded for the colormaps and is able to
    plot for ARD, PRD, and TVAR
    
    Parameters
    ----------
    pv_gen_filename : TYPE, required
        DESCRIPTION, generated PV generation from optimisation
     wind_gen_filename : TYPE, required
         DESCRIPTION, generated Wind generation from optimisation
    load_filename : TYPE, required
        DESCRIPTION, load profile with datetime index
    metric : TYPE, optional (string)
        DESCRIPTION, The default is ARD.
    
    Returns
    -------
    Plot for the metric specified
    
    '''
    
    col_names = ['SON', 'MUL', 'BCS', 'CHIH', 'JAL', 'GUE', 'TAM', 'PUE', 'VER',
                  'OAX', 'CHIA', 'YUC']

    
    gen_pv = pd.read_csv(pv_gen_filename)
    gen_pv = gen_pv.set_index(['fecha'])
    gen_pv.index = pd.to_datetime(gen_pv.index)
    gen_wind = pd.read_csv(wind_gen_filename)
    gen_wind = gen_wind.set_index(['fecha'])
    gen_wind.index = pd.to_datetime(gen_wind.index)
    load_df = pd.read_csv(load_filename)
    
    index_day = list(range(0,24,1))
    avg_load_prof = pd.DataFrame(index=index_day)
    load_df = load_df.set_index(['fecha'])
    load_df.index = pd.to_datetime(load_df.index)
    load_df['demand'] = load_df['demand']/1000
    avg_load_prof['avg_demand'] = load_df.groupby(load_df.index.hour).mean()
    avg_load_prof['Demand_std'] = load_df.groupby(load_df.index.hour).std()
    avg_load_prof['CoV'] = avg_load_prof['Demand_std'] / avg_load_prof['avg_demand']    
    
    
    avg_pv_site = pd.DataFrame(index=index_day)
    avg_wind_site = pd.DataFrame(index=index_day)
    
    wind_other = pd.DataFrame(index=index_day)
    pv_other = pd.DataFrame(index=index_day)
    
    for loc in col_names:
        avg_pv_site[f'{loc}'] = gen_pv[f'{loc}'].groupby(gen_pv.index.hour).mean()
        avg_wind_site[f'{loc}'] = gen_wind[f'{loc}'].groupby(gen_wind.index.hour).mean()
    
    
    for loc in col_names:
        
        if (avg_wind_site[f'{loc}'].max() <= 3):
            wind_other = pd.concat([wind_other, avg_wind_site[f'{loc}']], axis=1)
            avg_wind_site.drop([f'{loc}'], axis=1, inplace=True)
            
        if (avg_pv_site[f'{loc}'].max() <= 0.7):
            pv_other = pd.concat([pv_other, avg_pv_site[f'{loc}']], axis=1)
            avg_pv_site.drop([f'{loc}'], axis=1, inplace=True)
    
    
    wind_other['Other wind'] = wind_other.sum(axis=1) 
    avg_wind_site = pd.concat([avg_wind_site, wind_other['Other wind']], axis=1)   
    pv_other['Other'] = pv_other.sum(axis=1)
    avg_pv_site = pd.concat([avg_pv_site, pv_other['Other']], axis=1)
    
    if metric == "ARD":
        ### Labels and colormap for ARD
        label = ['SON', 'MUL', 'BCS', 'TAM', 'PUE', 'OAX', 'CHIA', 'YUC', 'Other wind', 'SON', 'MUL', 'TAM', 'YUC', 'Other PV']
        cmap= ['#030455', '#023E8A', '#015BA0', '#0077B6','#0083BD', '#00A5D0', '#00B4D8', '#48CAE4', '#6CD5EA', '#FA6E23', '#FC8B29', '#FD992C', '#FEA72F','#F2CB58']
    
        fig, ax = plt.subplots()
        ax.plot(index_day, avg_load_prof['avg_demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, avg_wind_site.T, avg_pv_site.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
        
        
    elif metric == "PRD":
        ### Labels and colormap for PRD
        label = ['SON', 'MUL', 'BCS', 'CHIH', 'TAM', 'CHIA', 'Other wind', 'YUC', 'Other PV']
        cmap =  ['#030455', '#023E8A', '#015BA0', '#0077B6','#0083BD', '#00A5D0', '#00B4D8', '#FA6E23', '#F2CB58']
    
        fig, ax = plt.subplots()
        ax.plot(index_day, avg_load_prof['avg_demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, avg_wind_site.T, avg_pv_site.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)
        
        
    elif metric == "TVAR":
        ### Labels and colormap for Minimized total variability in profile
        label = ['MUL', 'BCS', 'TAM', 'OAX', 'Other wind', 'SON', 'YUC', 'Other PV']
        cmap = ['#030455', '#023E8A', '#015BA0', '#0077B6', '#00A5D0',
                '#FA6E23', '#FC8B29', '#FD992C',]
    
        fig, ax = plt.subplots()
        ax.plot(index_day, avg_load_prof['avg_demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, avg_wind_site.T, avg_pv_site.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)

    elif metric == "MVAR":
        # # Labels for Minimized maximum variability in profile
        label = ['MUL', 'BCS', 'TAM', 'OAX', 'Other wind', 'SON', 'MUL', 'YUC',]
        cmap = ['#030455', '#023E8A', '#015BA0', '#0077B6', '#00A5D0',
                '#FA6E23', '#FC8B29', '#FD992C',]
        avg_pv_site = avg_pv_site.iloc[:,[0,2,1]]
        fig, ax = plt.subplots()
        ax.plot(index_day, avg_load_prof['avg_demand'], color='k', linestyle='--', label="Demand")
        ax.set_ylim(0,50)
        ax.set_xlim(0,23)
        ax.set_ylabel("GWh")
        ax.set_xlabel("Hour")
        ax.stackplot(index_day, avg_wind_site.T, avg_pv_site.T, labels = label, colors=(cmap))
        plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["0", "4", "8", "12", "16", "20"])
        ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.35,0,0))
        plt.grid(visible=True, alpha=0.5)


    return fig

def plot_energy_balance(file1, metric1, file2, metric2, file3, metric3, file4, metric4, 
                        file5, metric5):
    
    '''
    Function that obtains average daily generation and load profiles, and stackplots
    each location's average profile. Currently hardcoded for the colormaps and is able to
    plot for ARD, PRD, and TVAR
    
    Parameters
    ----------
    pv_gen_filename : TYPE, required
        DESCRIPTION, generated PV generation from optimisation
     wind_gen_filename : TYPE, required
         DESCRIPTION, generated Wind generation from optimisation
    load_filename : TYPE, required
        DESCRIPTION, load profile with datetime index
    metric : TYPE, optional (string)
        DESCRIPTION, The default is ARD.
    
    Returns
    -------
    Plot for the five metrics
    '''

    flows_1 = pd.read_csv(file1)
    flows_2 = pd.read_csv(file2)
    flows_3 = pd.read_csv(file3)
    flows_4 = pd.read_csv(file4)
    flows_5 = pd.read_csv(file5)
    # flows_6 = pd.read_csv(file6)
    
    flows_1['sigma'] = flows_1['Unmet_load'] - flows_1['Overgeneration']
    flows_1 = flows_1.sort_values(by='sigma', ascending=True)
    flows_1 = flows_1.reset_index(drop=True)
    
    flows_2['sigma'] = flows_2['Unmet_load'] - flows_2['Overgeneration']
    flows_2 = flows_2.sort_values(by='sigma', ascending=True)
    flows_2 = flows_2.reset_index(drop=True)
    
    flows_3['sigma'] = flows_3['Unmet_load'] - flows_3['Overgeneration']
    flows_3 = flows_3.sort_values(by='sigma', ascending=True)
    flows_3 = flows_3.reset_index(drop=True)
    
    flows_4['sigma'] = flows_4['Unmet_load'] - flows_4['Overgeneration']
    flows_4 = flows_4.sort_values(by='sigma', ascending=True)
    flows_4 = flows_4.reset_index(drop=True)
    
    flows_5['sigma'] = flows_5['Unmet_load'] - flows_5['Overgeneration']
    flows_5 = flows_5.sort_values(by='sigma', ascending=True)
    flows_5 = flows_5.reset_index(drop=True)
    
    # flows_6['sigma'] = flows_6['Unmet_load'] - flows_6['Overgeneration']
    # flows_6 = flows_6.sort_values(by='sigma', ascending=True)
    # flows_6 = flows_6.reset_index(drop=True)
    
    # fig, [ax, ax1, ax2] = plt.subplots(1, 2, 3, sharey=False, figsize=(6, 4))
    fig, ax = plt.subplots()
    #### Define inset 
    left, bottom, width, height = [0.45, 0.2, 0.4, 0.4] # These are in unitless percentages of the figure size for the inset. (0,0 is bottom left)
    ax1 = fig.add_axes([left, bottom, width, height])
    
    
    ### General plot data
    ax.set_xlim(-100,8860)
    ax.set_ylim(-200, 50)
    ax.plot(flows_1.index, flows_1['sigma'], color='#E76F51', linestyle='-', alpha=1, label=f"{metric1} sites")
    ax.plot(flows_2.index, flows_2['sigma'], color='#779B6C', linestyle='-', alpha=1, label=f"{metric2} sites")
    ax.plot(flows_3.index, flows_3['sigma'], color='#E9C46A', linestyle='-', alpha=1, label=f"{metric3} sites")
    ax.plot(flows_4.index, flows_4['sigma'], color='#2A9D8F', linestyle='-', alpha=1, label=f"{metric4} sites")
    ax.plot(flows_5.index, flows_5['sigma'], color='#264653', linestyle='-', alpha=1, label=f"{metric5} sites")
    # ax.plot(flows_6.index, flows_6['sigma'], color='#264653', linestyle='-', alpha=1, label=f"{metric6} sites")
    # ax.set_xticks(ticks=[0, 2190, 4380, 6570, 8760], labels=["0", "25", "50", "75", "100"])
    ax.set_xticks(ticks=[0, 2190, 4380, 6570, 8760], labels=["0", "2190", "4380", "6570", "8760"])
    ax.set_ylabel("Residual power (GW)")
    # ax.set_xlabel("Share of hours in the year (%)")
    ax.set_xlabel("Hour")
    ax.axhline(y=0, lw=1, color='k')
    ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.3,0,0))
    ax.grid(visible=True, alpha=0.5)
    
    #### Inset data
    ax1.set_xlim(3504,4250)
    ax1.set_ylim(-3, 3)
    ax1.plot(flows_1.index, flows_1['sigma'], color='#E76F51', linestyle='-', alpha=1, label=f"{metric1} sites")
    ax1.plot(flows_2.index, flows_2['sigma'], color='#779B6C', linestyle='-', alpha=1, label=f"{metric2} sites")
    ax1.plot(flows_3.index, flows_3['sigma'], color='#E9C46A', linestyle='-', alpha=1, label=f"{metric3} sites")
    ax1.plot(flows_4.index, flows_4['sigma'], color='#2A9D8F', linestyle='-', alpha=1, label=f"{metric4} sites")
    ax1.plot(flows_5.index, flows_5['sigma'], color='#264653', linestyle='-', alpha=1, label=f"{metric5} sites")
    # ax1.plot(flows_6.index, flows_6['sigma'], color='#264653', linestyle='-', alpha=1, label=f"{metric6} sites")
    # ax1.set_xticks(ticks=[2978, 3066, 3504, 3942, 4380], labels=["", "35", "40", "45", "50"])
    ax1.set_xticks(ticks=[2978, 3066, 3504, 3942, 4380], labels=["", "3066", "3504", "3942", "4380"])
    ax1.tick_params( labelsize=8)
    ax1.axhline(y=0, lw=0.5, color='k')
    
    plt.grid(visible=True, alpha=0.5)
    plt.gcf().set_size_inches(6, 4)

    return fig


def plot_energy_balance_one_metric(file1, metric1):
    
    flows_file = pd.read_csv(file1)
    flows = pd.DataFrame()
    flows['sigma'] = flows_file['Unmet_load'] - flows_file['Overgeneration']
    flows = flows.sort_values(by='sigma', ascending=True)
    flows = flows.reset_index(drop=True)
    
    charge = flows[(flows <= 0).all(axis=1)]
    discharge = flows[(flows >= 0).all(axis=1)]
    
    fig, ax = plt.subplots()

    ax.set_xlim(-100,8860)
    ax.set_ylim(-90, 40)
    ax.plot(flows.index, flows['sigma'], color='#264653', linestyle='-', alpha=1, label=f"{metric1} sites")

    ax.fill_between(charge.index, 0, charge['sigma'], alpha=0.2, label='Charge', color='#FC8B29')
    ax.fill_between(discharge.index, 0, discharge['sigma'], alpha=0.2, label='Discharge', color='b')

    # ax.set_xticks(ticks=[0, 2190, 4380, 6570, 8760], labels=["0", "25", "50", "75", "100"])
    ax.set_xticks(ticks=[0, 2190, 4380, 6570, 8760], labels=["0", "2190", "4380", "6570", "8760"])
    ax.set_ylabel("Residual power (GW)")
    # ax.set_xlabel("Share of hours in the year (%)")
    ax.set_xlabel("Hour")
    ax.axhline(y=0, lw=1, color='k')
    ax.legend(loc='lower center', fontsize='small', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.3,0,0))
    ax.grid(visible=True, alpha=0.5)
    
    plt.grid(visible=True, alpha=0.5)
    plt.gcf().set_size_inches(6, 4)

    return fig

#%% Generate plots

## Define paths
base_folder = os.path.dirname(__file__)
data_folder = os.path.join(base_folder, "data")
demand_folder = os.path.join(data_folder, "demand")
pml_folder = os.path.join(data_folder, "pml")
res_folder = os.path.join(data_folder, "RES")
results_folder = os.path.join(base_folder, "results")
plots_folder = os.path.join(results_folder, "plot results")
figs_folder = os.path.join(plots_folder, "figs")


demand_filename = os.path.join(demand_folder, "demand_SEN_2019.csv")
PV_filename = os.path.join(res_folder, "PV_MX_locs_2019.csv")
wind_filename = os.path.join(res_folder, "wind_MX_locs_2019.csv")
local_marginal_prices_filename = os.path.join(pml_folder, "pml_SEN_2019.csv")


al = 1
ard_flows_filename = os.path.join(results_folder, f"alpha{al}-RE_flows_ARD_sites.csv")
prd_flows_filename = os.path.join(results_folder, f"alpha{al}-RE_flows_PRD_sites.csv")
tvar_flows_filename = os.path.join(results_folder, f"alpha{al}-RE_flows_TVAR_sites.csv")
gen_flows_filename = os.path.join(results_folder, f"alpha{al}-RE_flows_GEN_sites.csv")
mvar_flows_filename = os.path.join(results_folder, f"alpha{al}-RE_flows_MVAR_sites.csv")
lolp_flows_filename = os.path.join(results_folder, f"alpha{al}-RE_flows_LOLP_sites.csv")
capacities_filename = os.path.join(results_folder, f"alpha{al}-RE_capacities_ARD_sites.csv")

#### Read files and plot for average daily profiles
### ARD
pv_gen_filename = os.path.join(plots_folder, "PV_gen_ARD.csv")
pv_avg_filename = os.path.join(plots_folder, "PV_avggen_ARD.csv")
wind_gen_filename = os.path.join(plots_folder, "Wind_gen_ARD.csv")
wind_avg_filename = os.path.join(plots_folder, "Wind_avggen_ARD.csv")
ard_figname = os.path.join(figs_folder, "ard_avgMixgen.png")
# plot_average_profiles(pv_gen_filename, wind_gen_filename, demand_filename, metric="ARD").savefig(ard_figname, dpi=500, bbox_inches="tight")


### PRD
pv_gen_prd_filename = os.path.join(plots_folder, "PV_gen_PRD.csv")
pv_avg_prd_filename = os.path.join(plots_folder, "PV_avggen_PRD.csv")
wind_gen_prd_filename = os.path.join(plots_folder, "Wind_gen_PRD.csv")
wind_avg_prd_filename = os.path.join(plots_folder, "Wind_avggen_PRD.csv")
prd_figname = os.path.join(figs_folder, "prd_avgMixgen.png")
# plot_average_profiles(pv_gen_prd_filename, wind_gen_prd_filename, demand_filename, metric="PRD").savefig(prd_figname, dpi=500, bbox_inches='tight')

### TVAR
pv_gen_tvar_filename = os.path.join(plots_folder, "PV_gen_TVAR.csv")
pv_avg_tvar_filename = os.path.join(plots_folder, "PV_avggen_TVAR.csv")
wind_gen_tvar_filename = os.path.join(plots_folder, "Wind_gen_TVAR.csv")
wind_avg_tvar_filename = os.path.join(plots_folder, "Wind_avggen_TVAR.csv")
tvar_figname = os.path.join(figs_folder, "tvar_avgMixgen.png")
# plot_average_profiles(pv_gen_tvar_filename, wind_gen_tvar_filename, demand_filename, metric="TVAR").savefig(tvar_figname, dpi=500, bbox_inches='tight')

### MVAR
pv_gen_mvar_filename = os.path.join(plots_folder, "PV_gen_MVAR.csv")
pv_avg_mvar_filename = os.path.join(plots_folder, "PV_avggen_MVAR.csv")
wind_gen_mvar_filename = os.path.join(plots_folder, "Wind_gen_MVAR.csv")
wind_avg_mvar_filename = os.path.join(plots_folder, "Wind_avggen_MVAR.csv")
mvar_figname = os.path.join(figs_folder, "mvar_avgMixgen.png")
# plot_average_profiles(pv_gen_mvar_filename, wind_gen_mvar_filename, demand_filename, metric="MVAR").savefig(mvar_figname, dpi=500, bbox_inches='tight')



### find day with the maximum total residual demand in that day for each metric
max_day_ard = int(find_max_res_demand_day(ard_flows_filename))
max_day_prd = int(find_max_res_demand_day(prd_flows_filename))
max_day_tvar = int(find_max_res_demand_day(tvar_flows_filename))
max_day_mvar = int(find_max_res_demand_day(mvar_flows_filename))




# find_peak_res_demand_day(ard_flows_filename)


### Get the hourly data from the day with maximum residual demand for each metric and plot result
ard_maxday_figname = os.path.join(figs_folder, "ard_maxday_mixgen.png")
# plot_max_res_demand_day(pv_gen_filename, wind_gen_filename, demand_filename, max_day=max_day_ard, metric="ARD").savefig(ard_maxday_figname, dpi=500, bbox_inches='tight')

prd_maxday_figname = os.path.join(figs_folder, "prd_maxday_mixgen.png")
# plot_max_res_demand_day(pv_gen_prd_filename, wind_gen_prd_filename, demand_filename, max_day=max_day_prd, metric="PRD").savefig(prd_maxday_figname, dpi=500, bbox_inches='tight')

tvar_maxday_figname = os.path.join(figs_folder, "tvar_maxday_mixgen.png")
# plot_max_res_demand_day(pv_gen_tvar_filename, wind_gen_tvar_filename, demand_filename, max_day=max_day_tvar, metric="TVAR").savefig(tvar_maxday_figname, dpi=500, bbox_inches='tight')

mvar_maxday_figname = os.path.join(figs_folder, "mvar_maxday_mixgen.png")
# plot_max_res_demand_day(pv_gen_mvar_filename, wind_gen_mvar_filename, demand_filename, max_day=max_day_mvar, metric="MVAR").savefig(mvar_maxday_figname, dpi=500, bbox_inches='tight')




# balance_figname = os.path.join(figs_folder, "residual_load_vs_hours.png")
# plot_energy_balance(prd_flows_filename, "PRD", ard_flows_filename, "ARD", 
#                     mvar_flows_filename, "MVAR", tvar_flows_filename, "TVAR",
#                     gen_flows_filename, "GEN").savefig(balance_figname, dpi=500, bbox_inches='tight')


# one_metric_balance_figname = os.path.join(figs_folder, "one_metric_balance.png")
# plot_energy_balance_one_metric(ard_flows_filename, "ARD").savefig(one_metric_balance_figname, dpi=500, bbox_inches='tight')

# reliability_figname = os.path.join(plots_folder, "reliability_all_metrics.png")
# plot_reliability_all_metrics(prd_flows_filename, "PRD", ard_flows_filename, "ARD", mvar_flows_filename,
#                              "MVAR", tvar_flows_filename, "TVAR", gen_flows_filename, "GEN").savefig(reliability_figname, dpi=500, bbox_inches='tight')

ard_1_filename = os.path.join(results_folder, "alpha1-RE_flows_ARD_sites.csv")
mvar_1_filename = os.path.join(results_folder, "alpha1-RE_flows_MVAR_sites.csv")
ard_15_filename = os.path.join(results_folder, "alpha1.5-RE_flows_ARD_sites.csv")
mvar_15_filename = os.path.join(results_folder, "alpha1.5-RE_flows_MVAR_sites.csv")
ard_3_filename = os.path.join(results_folder, "alpha3-RE_flows_ARD_sites.csv")
mvar_3_filename = os.path.join(results_folder, "alpha3-RE_flows_MVAR_sites.csv")


reliability_alphas_figname = os.path.join(plots_folder, "reliability_alphas.png")
plot_reliability_alphas(ard_1_filename, mvar_1_filename, ard_15_filename, mvar_15_filename,
                        ard_3_filename, mvar_3_filename).savefig(reliability_alphas_figname, dpi=500, bbox_inches='tight')


