#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:46:17 2023

@author: mnsgsty
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import os

# Define folders and data filenames
base_folder = os.path.dirname(__file__)
data_folder = os.path.join(base_folder, "data")
demand_folder = os.path.join(data_folder, "demand")
pml_folder = os.path.join(data_folder, "pml")
res_folder = os.path.join(data_folder, "RES")
results_folder = os.path.join(base_folder, "results")
plots_folder = os.path.join(results_folder, "plot results")

results_folder = os.path.join(base_folder, "results")
demand_filename = os.path.join(demand_folder, "demand_SEN_2019.csv")
PV_filename = os.path.join(res_folder, "PV_MX_locs_2019.csv")
wind_filename = os.path.join(res_folder, "wind_MX_locs_2019.csv")
local_marginal_prices_filename = os.path.join(pml_folder, "pml_SEN_2019.csv")

## Change values to optimise for different alphas, and the label for complementarity metric
al = 1
metric = "MVAR"
 
ts = 8760           # total timesteps, hours
sites = 12          # sites to consider


'''National demand data for 2019'''
demand_df = pd.read_csv(demand_filename)
demand_df = demand_df.drop('fecha', axis=1)


'''Capacity factor profiles for 2019 for each location'''
PV_locs_df = pd.read_csv(PV_filename)
PV_locs_df = PV_locs_df.drop('Unnamed: 0', axis=1)
wind_locs_df = pd.read_csv(wind_filename)
wind_locs_df = wind_locs_df.drop('Unnamed: 0', axis=1)

'''Local marginal prices in 2019'''
electricity_prices_df = pd.read_csv(local_marginal_prices_filename)
electricity_prices_df = electricity_prices_df.drop(['fecha'], axis=1) # in MXN/MWh
# Scale the local marginal price profile 
unmet_load_penalty_cost = (electricity_prices_df * 0.052 / 1000000) # in B. USD/GWh (2019 USD)

# Enable running for specific timeperiods, ts <= 8760
PV_locs_df = PV_locs_df.loc[0:ts-1, :]
wind_locs_df = wind_locs_df.loc[0:ts-1, :]
demand_df = demand_df.loc[0:ts-1,:]
demand_df = demand_df/1000  #demand profile in GW
unmet_load_penalty_cost = unmet_load_penalty_cost.loc[0:ts-1,:]

# proportion of annual demand met by renewables
alpha = cp.Parameter(pos=True, value=al)   


CFprof_pv = cp.Constant(PV_locs_df.to_numpy(dtype=float))
CFprof_wind = cp.Constant(wind_locs_df.to_numpy(dtype=float))
demand = cp.Constant(demand_df.to_numpy(dtype=float).flatten())
penalty_lol = cp.Constant(unmet_load_penalty_cost.to_numpy(dtype=float).flatten())


capacity_pv = cp.Variable((sites), nonneg=True)
capacity_wind = cp.Variable((sites), nonneg=True)
load_unmet = cp.Variable((ts), nonneg=True)
gtn_over = cp.Variable((ts), nonneg=True)
balance = cp.Variable((ts))
share_unmet = cp.Variable((ts), nonneg=True)
variability = cp.Variable((ts-1))
gen_pv = cp.Variable((ts, sites), nonneg=True)
gen_wind = cp.Variable((ts, sites))
total_RE_gen = cp.Variable((ts), nonneg=True)

constraints = []

for site in range(sites):
    constraints.append(gen_pv[:, site] == CFprof_pv[:, site] * capacity_pv[site])
    constraints.append(gen_wind[:, site] == CFprof_wind[:, site] * capacity_wind[site])
    

# calculate total generation
constraints.append(total_RE_gen == cp.sum(gen_pv, axis=1) + cp.sum(gen_wind, axis=1))

# constrain total renewables to be less than an annual proportion of total demand.
constraints.append(cp.sum(total_RE_gen) == alpha * cp.sum(demand))

# energy balance
constraints.append(balance == demand - total_RE_gen)

load_unmet = cp.pos(balance)
gtn_over = cp.neg(balance)


for t in range(1,ts):
    constraints.append(variability[t-1] == (total_RE_gen[t] - total_RE_gen[t-1]))


avg_res_demand = cp.sum(load_unmet) / ts
peak_res_demand = cp.max(load_unmet)
total_res_demand = cp.sum(load_unmet)
total_generation = cp.sum(total_RE_gen)
lolp = (cp.sum(load_unmet) / cp.sum(demand)) * 100
var = cp.tv(total_RE_gen)
max_var = cp.max(cp.norm(variability))


cost_pv = 0.872
cost_wind = 1.471 
pv_life = 25
wind_life = 30

system_cost = ((cost_pv / pv_life) * cp.sum(capacity_pv)) + ((cost_wind / wind_life) * cp.sum(capacity_wind)) + (cp.sum(cp.multiply(penalty_lol, load_unmet)))

#### uncomment only the OBJECTIVE function of the metric specified in line 33

# objective = cp.Minimize(peak_res_demand)
# objective = cp.Minimize(avg_res_demand)
# objective = cp.Maximize(total_generation)
# objective = cp.Minimize(lolp)
# objective = cp.Minimize(var)
objective = cp.Minimize(max_var)


prob = cp.Problem(objective, constraints)


result = prob.solve(solver = cp.MOSEK, warm_start=True, qcp=False, verbose=False)
# result = prob.solve(solver = cp.SCS, warm_start=True, max_iters=1000, verbose=False)


#### Save results

flows_filename = os.path.join(results_folder, f"alpha{al}-RE_flows_{metric}_sites.csv")
capacities_filename = os.path.join(results_folder, f"alpha{al}-RE_capacities_{metric}_sites.csv")

res = {'Demand': demand.value, 'Unmet_load': load_unmet.value, 'Overgeneration': gtn_over.value, 'Wind': cp.sum(gen_wind, axis=1).value,
        'PV': cp.sum(gen_pv, axis=1).value, 'Fraction Unmet': (load_unmet.value/demand.value), 'Fraction Curtailed': (gtn_over.value/demand.value),
        'Reliability': 1-(load_unmet.value/demand.value), 
        }

pd.DataFrame(data=res).to_csv(flows_filename)

pd.DataFrame({'PV_capacities': capacity_pv.value,
              'Wind_Capacities': capacity_wind.value}).to_csv(capacities_filename)


#### Save total and average generation files per optimisation 

save_pv_filename = os.path.join(plots_folder, f"PV_gen_{metric}.csv")
save_wind_filename = os.path.join(plots_folder, f"Wind_gen_{metric}.csv")
save_avg_pv_filename = os.path.join(plots_folder, f"PV_avggen_{metric}.csv")
save_avg_wind_filename = os.path.join(plots_folder, f"Wind_avggen_{metric}.csv")

index_day = list(range(0,24,1))

## Demand data with date index
load_df = pd.read_csv(demand_filename)
load_df = load_df.set_index(['fecha'])
load_df.index = pd.to_datetime(load_df.index)
load_df['demand'] = load_df['demand']/1000 # in GWh instead of MWh

avg_load_prof = pd.DataFrame(index=index_day)

avg_load_prof['avg_demand'] = load_df.groupby(load_df.index.hour).mean()
avg_load_prof['Demand_std'] = load_df.groupby(load_df.index.hour).std()
avg_load_prof['CoV'] = avg_load_prof['Demand_std'] / avg_load_prof['avg_demand']


total_RE_gen_df = pd.DataFrame(index=load_df.index, data=total_RE_gen.value, columns=['total_RE_gen'])
avg_total_RE_gen = pd.DataFrame(index=index_day)

avg_total_RE_gen['avg_total_RE_gen'] = total_RE_gen_df.groupby(total_RE_gen_df.index.hour).mean()
avg_total_RE_gen['std'] = total_RE_gen_df.groupby(total_RE_gen_df.index.hour).std()
avg_total_RE_gen['CoV'] = avg_total_RE_gen['std'] / avg_total_RE_gen['avg_total_RE_gen']


load_unmet_df = pd.DataFrame(index=load_df.index, data=load_unmet.value, columns=['Residual_demand'])
avg_load_unmet = pd.DataFrame(index=index_day)
avg_load_unmet['Mean daily unmet load'] = load_unmet_df.groupby(load_unmet_df.index.hour).mean()
avg_load_unmet['std daily unmet load'] = load_unmet_df.groupby(load_unmet_df.index.hour).std()
avg_load_unmet['CoV'] = avg_load_unmet['std daily unmet load']/avg_load_unmet['Mean daily unmet load']

col_names = ['SON', 'MUL', 'BCS', 'CHIH', 'JAL', 'GUE', 'TAM', 'PUE', 'VER',
              'OAX', 'CHIA', 'YUC']
pv_gen_df = pd.DataFrame(index=load_df.index, data=gen_pv.value, columns=col_names)
wind_gen_df = pd.DataFrame(index=load_df.index, data=gen_wind.value, columns=col_names)

avg_pv_site = pd.DataFrame(index=index_day)
avg_wind_site = pd.DataFrame(index=index_day)

wind_other = pd.DataFrame(index=index_day)
pv_other = pd.DataFrame(index=index_day)

for loc in col_names:
    avg_pv_site[f'{loc}'] = pv_gen_df[f'{loc}'].groupby(pv_gen_df.index.hour).mean()
    avg_wind_site[f'{loc}'] = wind_gen_df[f'{loc}'].groupby(wind_gen_df.index.hour).mean()


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


pv_gen_df.to_csv(save_pv_filename)
wind_gen_df.to_csv(save_wind_filename)
avg_pv_site.to_csv(save_avg_pv_filename)
avg_wind_site.to_csv(save_avg_wind_filename)