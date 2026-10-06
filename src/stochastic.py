#%% import
# -*- coding: utf-8 -*-
"""
Grid construction
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import imageio
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import matplotlib.colors as mpcolors
from matplotlib.ticker import LogLocator
from matplotlib.ticker import SymmetricalLogLocator
import seaborn as sns
import time as timing
from tqdm import tqdm
from functools import reduce
from matplotlib import path
import scipy.optimize as opt


import pandas as pd
import csv

import os
import sys


import cProfile
import pstats

from functions import *
from release_config import OUTPUT_DIR, project_path, selected_drug




plt.close('all')
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['savefig.transparent'] = False


#%% test parameters
#plot thing
doPlot = 1 #boolean
savePlot = 1
doPlotQuantity = 1
savePlotQuantity = 1
doGif = 1
saveCSV = 1
saveMatrix = 0
debug = False
doProfile=0


seed = 1234567
seed = int(os.environ.get("TB_SEED", rand.randint(1000,10000000,1)[0]))
rand.seed(seed)

dx = 20 #space step (in both x and y)
dt = 0.01 #time step ABM
dt2 = 0.1 #for the Drug and Ox simulations
dt3 = 1 #for Drug and Oxygen after treatment


# drug_name = 'BDQ'
drug_name = selected_drug()


# #BDQ
if drug_name == 'BDQ':
    Len_x= 3400 #mesh width (rounded to a multiple of dx)
    Len_y= 3400 #mesh height
    D_out = 450000 #400 mg
    D_out_change = 150000 #after 3 days reduce the dose from 400 to 40

    MBC90 = 2280 #drug concentration to kill TBslow  MW= 555g/mol x MBC90= 4,12 uM = 2300ug/l = 2300 ng/ml 
    D_max = 13330
    D_min = 5538
    shape = 37
    b = 32 #unitless
    elim_rate = 0.6397
    C0drug = 3000 #boundary concentration
    Len_x_fit = 3400
    if Len_x == None:
        Len_x = Len_x_fit
    if dx == None:
        dx = 34
    out_don_thickness = 350 #um the thickness of the outer donuts
    
    ka1 = 1.0     # rapid gastrointestinal absorption
    ke  = 5.0     # rapid clearance driven by local blood flow
    ka2 = 15.0    # rapid, strong lysosomal uptake
    ka3 = 0.0058   # very slow lysosomal release (5.5-month half-life)
    #(ka3 slightly increased to offset losses in the 2D simulation)
    n_don = 5

    Drugkill_f = 2280
    Drugkill_s = 2280
    Drugkill_mi = 2400


#TBAJ587
elif drug_name == 'TBAJ587':
    Len_x= 4000 #mesh width (rounded to a multiple of dx)
    Len_y= 4000 #mesh height

    D_out = 700000 #300 mg flat -> 26000        (19000 old) at 3day
    D_out_change = 200000 #after 3 days reduce the dose from 300 to 30 #30 mg flat -> 11000
    MBC90 = 332 #drug concentration to kill TBslow  MW= 615g/mol x MBC90= 0,54 uM = 332 ug/l = 332 ng/ml paper Bustion 2025
    D_max = 9170
    D_min = 5870
    shape = 83
    b = 35
    elim_rate = 0.19
    C0drug = 7000 #plot max colorbar
    Len_x_fit = 4000
    if Len_x == None:
        Len_x = Len_x_fit
    if dx == None:
        dx = 40
    out_don_thickness = 500 #um the thickness of the outer donuts
    
    ka1 = 1.2     # rapid gastrointestinal absorption
    ke  = 6.5     # rapid clearance driven by local blood flow
    ka2 = 8.0    # rapid, strong lysosomal uptake
    ka3 = 0.0025   # very slow lysosomal release (5.5-month half-life)
    n_don = 4

    Drugkill_f = 300
    Drugkill_s = 332
    Drugkill_mi = 360





Len_x = round(float(os.environ.get("TB_DIAMETER", Len_x)) / dx) * dx
Len_y = Len_x  # the circular ABM uses a square computational mesh
n_don = int(os.environ.get("TB_N_DON", n_don))
b_param = int(b*Len_x_fit/100) #b*Len_x_fit/100 in um
T = int(os.environ.get("TB_T_END", "500"))  # final time in days
snapshot_days = {5, 25, 50, 84, 100, 200, 250, 300, 350, T}

#Drug
t_start_dose = 84 #first dose after 12-16 weeks (day 84-112)
t_change_dose = 3
t_stop_dose = 180
t_dissipation_drug = 170


if D_out_change != None and t_change_dose != None:
    doses = [(float(t), D_out) for t in range(t_start_dose, t_start_dose + t_change_dose)] + [(float(t), D_out_change) for t in range(t_start_dose + t_change_dose, t_start_dose + t_stop_dose)]
else:
    doses = [(float(t), D_out) for t in range(t_start_dose, t_start_dose + t_stop_dose)]



params_pk = (ka1, ke, ka2, ka3)
BC_conc = create_bc_function(doses=doses, params=params_pk, t_final=T)
 

R=round(Len_x/2)

C0oxy = 76.16 #micromoli (do O2) * 1e-8/dx^2

ke_tb = 1.1642 * 10000 * 7
ke_t = 362.4
ke_mr = 362.4
ke_ma = 724.8
ke_mi = 1087.2
ke_mci = 1449.6

Dox = (1.728)*1e8 #micrometri^2/day (2x86400=172800)

#movement and life times
movT = 0.007 #day
movMr = 0.014
movMa = 0.325
movMi = 1. #same for Mci
movTB = 1e10
movCaseum = 1 #move in centre direction with a switch with the other cell
mov_vec = [1e10, movTB, movTB, movT, movMr, movMa, movMi, movMi, movCaseum]

lifeT = 3 #day
lifeMr = 100
lifeMa = 10
lifeMi = 50 #same for Mci
lifeTB = 1e10
life_vec = [1e10, lifeTB, lifeTB, lifeT, lifeMr, lifeMa, lifeMi, lifeMi, 1e10]

mov_vec_np = np.array(mov_vec)
life_vec_np = np.array(life_vec)

Nici = 10 # Tb treshold for Mi -> Mci
Ncib = 20 # Tb treshold for Mci -> burst
NstartRec = 250 #[10-250]

Oslow_to_fast = 0.5 * C0oxy
Ofast_to_slow = 0.03 * C0oxy

P_kill_T_Mi = 0.01

P_activ_MrMa = 0.09

Pnew_Mr = 200 / 400
Pnew_T = 300 / 400

P_Mi_move_Tb = 0.7 #Mi is more attracted to Mtb when it can engulf it
P_Ma_deat_caseum = 0.4
P_Ma_healt_caseum = 40
P_MrMi = 0.0005
P_Mr_kill_tb = 0.0003
P_activ_MiMa = 0.01

replication_fast = 1 #day (time between replications)
replication_slow = 3

reduction_factor = 1
reductionT = 5 #[5-10]
life_factorT = 2 #[1.5-2]
reductionMr = 5 #[5-10]


times = np.linspace(1,T,T, dtype=int)

n_times = len(times)


t_saveMatrix = t_start_dose
 
#initial things
Nx = int(Len_x/dx)+1
Ny = int(Len_y/dx)+1
x_val = np.arange(0,Len_x + dx, dx)
y_val = np.arange(0,Len_y + dx, dx)
Ds = hill_Ds(Nx, D_max, D_min, shape, b, ratio=Len_x/Len_x_fit)


D_case = '2Dhill'
DDrug_title = f'{D_case} Dmax{D_max}_Dmin{D_min}_shape{shape}_b{b}_Dout{D_out}_elimrate{elim_rate}'


#%% creation of the mesh, boundary conditions and initial conditions multi D

meshDrug = np.zeros((Nx,Nx))
meshOx = np.zeros((Nx,Nx))
meshCellular = np.ones((Nx, Nx))*1000
meshMovTime = np.ones((Nx, Nx))*1e10
meshLifeTime = np.ones((Nx, Nx))*1e10
meshInfected = np.zeros((Nx,Nx))
meshReplication = np.ones((Nx,Nx))*1e10


meshDistance = np.zeros((Nx,Nx)) #represent the distance of a point from the centre 
meshProbN = np.ones((Nx,Nx))
meshProbR = np.ones((Nx,Nx))
meshProbS = np.ones((Nx,Nx))
meshProbL = np.ones((Nx,Nx))

Ds_matrix = np.zeros((2*Nx-1, 2*Nx-1))
Ds_matrix.fill(Ds[0])
x = np.linspace(0, Nx-1, Nx, dtype=int)
y = np.linspace(0, Ny-1, Ny, dtype=int)
x, y = np.meshgrid(x, y)

dr = R/n_don

#creation Ds_matrix
r_circles = np.linspace(R, dx, round((Nx-1)/2))
# circle center at (R, R)
cx, cy = R, R 
# coordinate grid for Ds_matrix
# i, j index Ds_matrix (size 2*Nx-1)
X, Y = np.ogrid[:2*Nx-1, :2*Ny-1]
# Convertiamo gli indici in coordinate reali
# index 0 maps to 0 and index 2*Nx-2 maps to 2*R
coords_x2 = X * (dx / 2)
coords_y2 = Y * (dx / 2)
# compute each point's distance from the center
dist_sq = (coords_x2 - cx)**2 + (coords_y2 - cy)**2
# radial assignment
for k in range(len(r_circles)):
    mask = dist_sq <= r_circles[k]**2
    Ds_matrix[mask] = Ds[k]
    



coords_x = np.linspace(0, 2*R, Nx)
coords_y = np.linspace(0, 2*R, Ny)
X, Y = np.meshgrid(coords_x, coords_y)
# 2. compute each cell's distance from the center (R, R)
distance = np.sqrt((X - R)**2 + (Y - R)**2)

#mesh_inside_index_don
r_circles = np.linspace(dr, R, n_don)
if out_don_thickness != None:
    r_circles[-2] = R-out_don_thickness

previous = []
mesh_inside_index_don = []
mesh_inside_index_don_central_line = []

for k in range(len(r_circles)):
    r = r_circles[k]            
    #BC ceerchio
    lower_dist = np.asarray(np.where(np.abs(distance) < r)).T
    #interior points
    inner = []
    inner_central_line = []
    for el in lower_dist:
        i = int(el[0])
        j= int(el[1])
        if [i,j] not in previous:
            if j == int(Ny/2):
                inner_central_line += [[i,j]]
            inner += [[i,j]]
            previous += [[i,j]]
    mesh_inside_index_don += [inner]
    mesh_inside_index_don_central_line += [inner_central_line]


#BC ceerchio
mask_esterno = distance > R 
BC_index_tupla = np.where(np.abs(distance - R) < dx) # ([arrray row index], [array colum index])
BC_index_np = np.asarray(BC_index_tupla).T
BC_index = BC_index_np.tolist()

last_don = []
for el in mesh_inside_index_don[-1]:
    if el not in BC_index:
        last_don += [el]
mesh_inside_index_don[-1] = last_don

mesh_inside_index_don_tupla = [tuple(np.array(d).T) for d in mesh_inside_index_don]

#interior points
mesh_inside_index = []
inside = np.asarray(np.where(distance < R)).T.tolist()
for el in inside:
    if el not in BC_index:
        mesh_inside_index += [el]

mesh_inside_index_np = np.array(mesh_inside_index)
mesh_inside_index_tupla = tuple(mesh_inside_index_np.T)

        


#distance matrix
Nx2 = round((Nx-1)/2)
for i in range(Nx):
    for j in range(Nx):
        meshDistance[i,j] = np.sqrt((i-Nx2)**2 + (j-Nx2)**2)

#probability matrix
for i in range(Nx):
    for j in range(Nx):
        meshProbN[i,j] = 0.25*(1 + (1-(meshDistance[i,j]/Nx2))*((Nx2-j)/(np.abs(Nx2-j)+np.abs(Nx2-i))))
        meshProbR[i,j] = 0.25*(1 + (1-(meshDistance[i,j]/Nx2))*((Nx2-i)/(np.abs(Nx2-j)+np.abs(Nx2-i))))
        meshProbS[i,j] = 0.25*(1 - (1-(meshDistance[i,j]/Nx2))*((Nx2-j)/(np.abs(Nx2-j)+np.abs(Nx2-i))))
        meshProbL[i,j] = 0.25*(1 - (1-(meshDistance[i,j]/Nx2))*((Nx2-i)/(np.abs(Nx2-j)+np.abs(Nx2-i))))

meshProbN[Nx2,Nx2] = 0.25
meshProbR[Nx2,Nx2] = 0.25
meshProbS[Nx2,Nx2] = 0.25
meshProbL[Nx2,Nx2] = 0.25

#%% resolution 2D multi D
if doProfile:
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()


if saveMatrix:
    meshCellular_copiaprova = np.ones((Nx, Nx))*1000
    meshMovTime_copiaprova = np.ones((Nx, Nx))*1e10
    meshLifeTime_copiaprova = np.ones((Nx, Nx))*1e10
    meshInfected_copiaprova = np.zeros((Nx,Nx))
    meshReplication_copiaprova = np.ones((Nx,Nx))*1e10

base_seed = seed
for iteration in tqdm(range(int(os.environ.get("TB_N_SIM", "1")))):
    seed = base_seed + iteration
    rand.seed(seed)
    plt.close('all')
    # print(iteration)
    named_tuple = timing.localtime() # get struct_time
    time_string = timing.strftime("%Y%m%d%H%M%S", named_tuple)
    folder = str(OUTPUT_DIR / (time_string+f'_{iteration:04d}_{drug_name}_stochastic_agent_simulation_Tend_{T}'))
    os.makedirs(folder, exist_ok=True)
    if savePlot or saveCSV or saveMatrix:
        folderCSV = folder+'/csv'
        folderPNG = folder+'/png'
        os.makedirs(folderCSV, exist_ok=True)
        os.makedirs(folderPNG, exist_ok=True)
        
    if saveCSV:
        all_csv_rows = []
        if n_don == 4:
            header = ['time id', 'time',\
                          'ox BC', 'ox 20', 'ox 40', 'ox 60', 'ox 80', 'ox centre',\
                          'N Tb', 'N caseum', 'N healty', 'N T cell', 'N Mr', 'N Ma', 'N Mi',\
                          'sim C in don1', 'sim C in don2', 'sim C in don3', 'sim C in don4', 'centre']
        elif n_don == 5:
            header = ['time id', 'time',\
                          'ox BC', 'ox 20', 'ox 40', 'ox 60', 'ox 80', 'ox centre',\
                          'N Tb', 'N caseum', 'N healty', 'N T cell', 'N Mr', 'N Ma', 'N Mi',\
                          'sim C in don1', 'sim C in don2', 'sim C in don3', 'sim C in don4', 'sim C in don5' , 'centre']
    
    
        csv_name = f"{time_string} Stochastic ABM simulation"
        
        with open(folderCSV+'/'+csv_name+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
    
        
        
    meshDrug = np.zeros((Nx,Ny))
    meshOx = np.zeros((Nx,Nx))
    meshCellular = np.ones((Nx, Nx), dtype= int)*1000
    meshMovTime = np.ones((Nx, Nx))*1e10
    meshLifeTime = np.ones((Nx, Nx))*1e10
    meshInfected = np.zeros((Nx,Nx), dtype= int)
    meshReplication = np.ones((Nx,Nx))*1e10 
    
    random_meshMovTime = np.abs(rand.normal(size = (Nx,Nx)))
    random_meshLifeTime = np.abs(rand.normal(size = (Nx,Nx))) #aggiunta
    random_meshReplication = np.abs(rand.normal(size = (Nx,Nx)))
    # 0 empty, 1 TBslow, 2 Tbfast, 3 Tcell, 4 Mr, 5 Ma, 6 Mi, 7 Mci, 8 caseum
    for i in range(n_don):
        don = np.asarray(mesh_inside_index_don[i]).T
        if i == 0:
            meshCellular[don[0], don[1]] = 0
        elif i == 1:
            meshCellular[don[0], don[1]] = 0
        elif i == 2:
            meshCellular[don[0], don[1]] = 0
        else:
            meshCellular[don[0], don[1]] = 0
    
    
    meshCellular[Nx2, Nx2] = 2
    meshMovTime[Nx2, Nx2] = movTB
    meshLifeTime[Nx2, Nx2] = lifeTB
    meshReplication[Nx2, Nx2] = replication_fast * random_meshReplication[Nx2, Nx2]
    
    meshCellular[np.asarray(BC_index).T[0], np.asarray(BC_index).T[1]] = 0
    meshMovTime[np.asarray(BC_index).T[0], np.asarray(BC_index).T[1]] = 1e10
    meshLifeTime[np.asarray(BC_index).T[0], np.asarray(BC_index).T[1]] = 1e10
    
    
    #set the oxygen boundary condition
    for BC in BC_index:
        meshOx[BC[0],BC[1]] = C0oxy
    
    #resolution
    y_val_plot, x_val_plot = np.meshgrid(y_val, x_val)
    
    Ntb = []
    Ncaseum = []
    Nhealty = []
    Ntcell = []
    Nmr = []
    Nmi = []
    Nma = []
    Perc_ox_centre = [] #I consider the radius on x-axis left
    Perc_ox_20 = []
    Perc_ox_40 = []
    Perc_ox_60 = []
    Perc_ox_80 = []
    Perc_ox_boundary = []
    
    
    if doPlot:
        plt.ion() #plot interattivo
    else:
        plt.ioff()
    
    if doPlot:
        # colors = ['white', 'goldenrod', 'blue', 'red', 'lime', 'magenta', 'yellow', 'darkorange', 'black', 'white']
        colors = ['white', 'goldenrod', 'yellow', 'red', 'limegreen', 'blue', 'aqua', 'aqua', 'black', 'white']
        #empty, Tb slow, Tb fast, T cell, Mr, Ma, Mi, Mci, caseum, else
        colors2 = sns.color_palette("hot", 20)
        colors2 = np.flip(colors2, axis=0)
        colors2[0] = [1,1,1]
        colors2[-1] = [0,0,0]
        colors3 = sns.color_palette("cool", 6)
        cmap = mpcolors.LinearSegmentedColormap.from_list("MyCmap", colors, len(colors))
        cmap2 = mpcolors.LinearSegmentedColormap.from_list("MyCmap2", colors2, len(colors2))
            
        fig = plt.figure(figsize= (13,8), layout="constrained")
    
        gs = plt.GridSpec(3, 4, figure=fig)
        ax2 = fig.add_subplot(gs[:, :-1])
        ax1 = fig.add_subplot(gs[0, -1])
        ax3 = fig.add_subplot(gs[1, -1])
        ax4 = fig.add_subplot(gs[2, -1])
        
        c1 = ax1.pcolor(x_val_plot, y_val_plot, meshDrug,shading="nearest", cmap="Greens", vmin=0, vmax=C0drug)
        c2 = ax2.pcolor(x_val_plot, y_val_plot, meshCellular,shading="nearest", cmap=cmap, vmin=0, vmax=9)
        c3 = ax3.pcolor(x_val_plot, y_val_plot, meshOx,shading="nearest", cmap="Blues", vmin=0, vmax=C0oxy)
        c4 = ax4.pcolor(x_val_plot, y_val_plot, meshInfected,shading="nearest", cmap=cmap2, vmin=0, vmax=Ncib)
    
        cbar1 = fig.colorbar(c1, ax=[ax1],location='right', ticks=[0, round(C0drug/6), round(C0drug/3), round(C0drug/2), round(2*C0drug/3), round(5*C0drug/6), C0drug])
        cbar1.ax.set_yticklabels([0, round(C0drug/6), round(C0drug/3), round(C0drug/2), round(2*C0drug/3), round(5*C0drug/6), f'> {C0drug}'], fontsize=15)
        cbar1.ax.set_title('ng/g', fontsize=15)
        
        cbar2 = fig.colorbar(c2, ax=[ax2],location='right',ticks=[0.45,1*0.9+ 0.45,2*0.9+ 0.45,3*0.9+ 0.45,4*0.9+ 0.45, \
                                                                  5*0.9+ 0.45,6*0.9+ 0.45,7*0.9+ 0.45,8*0.9+ 0.45,9*0.9+ 0.45])
        cbar2.ax.set_yticklabels(['Empty','TB slow','TB fast','T cell','Mr','Ma','Mi','Mci','Caseum', ''], fontsize=15)
        cbar3 = fig.colorbar(c3, ax=[ax3],location='right')
        cbar3.ax.tick_params(labelsize=15)
        cbar3.ax.set_title(r'$\mu$ mol/cell', fontsize=15)
        cbar4 = fig.colorbar(c4, ax=[ax4],location='right', ticks = [0, 5, 10, 15, 20])
        cbar4.ax.set_yticklabels([0, 5, 10, 15, 'Caseum'], fontsize=15)

    time = 0.
    i=0
    drug_sim_bool = True
    tIni = 0. 
    tic = timing.time()
    
    
    conc_meanDrug = np.zeros(n_don)
    for tEnd in tqdm(times):
        csv_row = [time_string, tEnd]
        if drug_sim_bool:
            if tIni > t_start_dose + t_stop_dose - 1e-5: #dt3
                paramsDrug = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshDrug, BC_index_tupla, dx, dt3, BC_conc, elim_rate]  
                paramsOx = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshOx, BC_index_tupla, dx, dt3, C0oxy, meshCellular, ke_tb, ke_t, ke_mr, ke_ma, ke_mi, ke_mci]
            else: #dt2
                paramsDrug = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshDrug, BC_index_tupla, dx, dt2, BC_conc, elim_rate]
                paramsOx = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshOx, BC_index_tupla, dx, dt2, C0oxy, meshCellular, ke_tb, ke_t, ke_mr, ke_ma, ke_mi, ke_mci]
        else:
            paramsOx = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshOx, BC_index_tupla, dx, dt3, C0oxy, meshCellular, ke_tb, ke_t, ke_mr, ke_ma, ke_mi, ke_mci]
        
        if drug_sim_bool:
            if tIni > t_start_dose - 1e-5:
                conc_meanDrug, meshDrug, mesh_meanDrug = simulation2D_multiple_D_numba(Ds_matrix, paramsDrug)
            if tIni > t_start_dose + t_stop_dose + t_dissipation_drug - 1e-5:
                drug_sim_bool = False
        
        conc_meanOx, meshOx, mesh_meanOx = simulation2D_oxy_numba(Dox, paramsOx)
        

        paramsSim = [tIni, tEnd, meshDrug, meshOx, BC_index_np,Nx, dx, dt,\
            meshCellular, meshMovTime, meshLifeTime, meshInfected, meshReplication,\
            meshProbN,meshProbR,meshProbS,meshProbL,\
            movT, movMr, movMa, movMi, movCaseum, lifeT, lifeMr, lifeMa, lifeMi,\
            mov_vec_np, life_vec_np, Nici, Ncib, NstartRec,\
            Drugkill_f, Drugkill_s, Drugkill_mi, Oslow_to_fast, Ofast_to_slow,\
            P_kill_T_Mi, P_activ_MrMa, Pnew_Mr, Pnew_T, P_Mi_move_Tb, P_Ma_deat_caseum, P_Ma_healt_caseum, P_MrMi, P_Mr_kill_tb, P_activ_MiMa,\
            replication_fast, replication_slow,\
                reduction_factor,reductionT, reductionMr,life_factorT,\
            seed]
                
            
        meshCellular, meshMovTime, meshLifeTime, meshInfected, meshReplication= cellular_movements_numba(\
             tIni, tEnd, meshDrug, meshOx, BC_index_np,Nx, dx, dt,\
            meshCellular, meshMovTime, meshLifeTime, meshInfected, meshReplication,\
            meshProbN,meshProbR,meshProbS,meshProbL,\
            movT, movMr, movMa, movMi, movCaseum, lifeT, lifeMr, lifeMa, lifeMi,\
            mov_vec_np, life_vec_np, Nici, Ncib, NstartRec,\
            Drugkill_f, Drugkill_s, Drugkill_mi, Oslow_to_fast, Ofast_to_slow,\
            P_kill_T_Mi, P_activ_MrMa, Pnew_Mr, Pnew_T, P_Mi_move_Tb, P_Ma_deat_caseum, P_Ma_healt_caseum, P_MrMi, P_Mr_kill_tb, P_activ_MiMa,\
            replication_fast, replication_slow,\
                reduction_factor,reductionT, reductionMr,life_factorT,\
            seed)
        
        
        Ntb += [np.shape(np.asarray(np.where(meshCellular == 1)))[1] + np.shape(np.asarray(np.where(meshCellular == 2)))[1]]
        Ncaseum += [np.shape(np.asarray(np.where(meshCellular == 8)))[1]]
        Nhealty += [np.shape(np.asarray(np.where(meshCellular == 0)))[1]]
        Ntcell += [np.shape(np.asarray(np.where(meshCellular == 3)))[1]]
        Nmr += [np.shape(np.asarray(np.where(meshCellular == 4)))[1]]
        Nma +=[np.shape(np.asarray(np.where(meshCellular == 5)))[1]]
        Nmi += [np.shape(np.asarray(np.where(meshCellular == 6)))[1] + np.shape(np.asarray(np.where(meshCellular == 7)))[1]]
        Perc_ox_centre += [meshOx[Nx2, Nx2]*100/C0oxy]
        Perc_ox_20 += [meshOx[round(Nx2*0.2), Nx2]*100/C0oxy]
        Perc_ox_40 += [meshOx[round(Nx2*0.4), Nx2]*100/C0oxy]
        Perc_ox_60 += [meshOx[round(Nx2*0.6), Nx2]*100/C0oxy]
        Perc_ox_80 += [meshOx[round(Nx2*0.8), Nx2]*100/C0oxy]
        Perc_ox_boundary += [meshOx[1, Nx2]*100/C0oxy]
        
        
        csv_row = np.concatenate((csv_row, [Perc_ox_boundary[-1], Perc_ox_20[-1], Perc_ox_40[-1],\
                                              Perc_ox_60[-1], Perc_ox_80[-1], Perc_ox_centre[-1],\
                                              Ntb[-1], Ncaseum[-1], Nhealty[-1], Ntcell[-1], Nmr[-1], Nma[-1], Nmi[-1]]))
        csv_row = np.concatenate((csv_row, conc_meanDrug))
        csv_row = np.concatenate((csv_row, [meshDrug[Nx2, Nx2]]))
        csv_row = [eval(i) for i in csv_row] #now we have int and not string

        csv_row = np.asarray(np.round(csv_row, 2), dtype=float).tolist()
    
        all_csv_rows.append(csv_row)
        
        if saveMatrix and tEnd==t_saveMatrix:
            meshCellular_copiaprova = meshCellular
            meshMovTime_copiaprova = meshMovTime
            meshLifeTime_copiaprova = meshLifeTime 
            meshInfected_copiaprova = meshInfected
            meshReplication_copiaprova = meshReplication
            save_matrix(meshCellular_copiaprova, folder + '/meshCel.npy')
            save_matrix(meshMovTime_copiaprova, folder + '/meshMov.npy')
            save_matrix(meshLifeTime_copiaprova, folder + '/meshLif.npy')
            save_matrix(meshInfected_copiaprova, folder + '/meshInf.npy')
            save_matrix(meshReplication_copiaprova, folder + '/meshRep.npy')
            # loaded_A = load_matrix('meshCel.npy')
        
        tIni = tEnd
        if doPlot and tEnd in snapshot_days:
            if drug_sim_bool:
                ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            
            if drug_sim_bool:
                c1 = ax1.pcolor(x_val_plot, y_val_plot, meshDrug,shading="nearest", cmap="Greens", vmin=0, vmax=C0drug)
                ax1.set_title('Drug', fontsize=18)
                ax1.xaxis.set_tick_params(labelbottom=False)
                ax1.set_xticks([])
                ax1.set_yticks([])
                
            c2 = ax2.pcolor(x_val_plot, y_val_plot, meshCellular,shading="nearest", cmap=cmap, vmin=0, vmax=9)
            c3 = ax3.pcolor(x_val_plot, y_val_plot, meshOx,shading="nearest", cmap="Blues", vmin=0, vmax=C0oxy)
            c4 = ax4.pcolor(x_val_plot, y_val_plot, meshInfected,shading="nearest", cmap=cmap2, vmin=0, vmax=Ncib)
            ax2.set_title('Cellular mesh',fontsize=20)
            ax3.set_title('Oxygen', fontsize=18)
            ax4.set_title('Mtb in Mi or Mci', fontsize=18)
            fig.suptitle(f"time: {tEnd}",fontsize=22, wrap = True)
            
            # Hide X and Y axes label marks
            ax3.xaxis.set_tick_params(labelbottom=False)
            ax3.yaxis.set_tick_params(labelleft=False)
            ax4.xaxis.set_tick_params(labelbottom=False)
            ax4.yaxis.set_tick_params(labelleft=False)
            
            # Hide X and Y axes tick marks
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax4.set_xticks([])
            ax4.set_yticks([])
            
            ax2.set_xlabel(r'x-space point [$\mu m$]', fontsize=18)
            ax2.set_ylabel(r'y-space point [$\mu m$]', fontsize=18)
            ax2.tick_params(axis='both',which='major', labelsize=15)
            
            refresh_figure(fig)

            if savePlot:
                figName = f"{time_string}_stochastic_agent_simulation_2D_time_{10000+tEnd}"
                figName = figName.translate(str.maketrans('', '', '.'))
                fig.savefig(folderPNG+'/'+figName+'.png',bbox_inches='tight')
