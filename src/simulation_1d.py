#%% import

# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import time as timing
from tqdm import tqdm
from functools import reduce
from matplotlib import path
import scipy.optimize as opt
import scipy.sparse.linalg as linalg
import scipy.stats
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy.random as rand
import csv
import os
import sys
from functions import *
from release_config import OUTPUT_DIR, project_path, selected_drug


plt.close('all')
# plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['savefig.transparent'] = False

#%% test parameters
#booleans
doVariabilityPK = os.environ.get("TB_PK_VARIABILITY", "0") == "1"
doPlot = 0 #boolean
savePlot = 0
doPlotFit = 0  # Save final fit plots without opening a live GUI
savePlotFit = 1
plotObs = os.environ.get("TB_PLOT_OBS", "0") == "1"
plotMBC90 = 1
doPlotDistributionVaribility = 0
savePlotDistributionVaribility = 0
saveCSV = 1


dx = None
Len_x = None
dt2 = 0.1 #for the Drug and Ox simulations
dt3 = 1 #for Drug and Oxygen after treatment

# drug_name = 'BDQ'
drug_name = selected_drug()


# #BDQ
if drug_name == 'BDQ':
    D_out = 450000
    D_out_change = 150000 #after 3 days reduce the dose from 400 to 40

    MBC90 = 2280 #drug concentration to kill TBslow  MW= 555g/mol x MBC90= 4,12 uM = 2300ug/l = 2300 ng/ml 
    D_max = 13330
    D_min = 5538
    shape = 37
    b = 32 #unitless
    elim_rate = 0.6397
    C0 = 3000 #boundary concentration
    Len_x_fit = 3400
    if Len_x == None:
        Len_x = Len_x_fit
    if dx == None:
        dx = 34
    out_don_thickness = 350 #um the thickness of the outer donuts
    
    F = 1
    ka1 = 1.0     # rapid gastrointestinal absorption
    ke  = 5.0     # rapid clearance driven by local blood flow
    ka2 = 15.0    # rapid, strong lysosomal uptake
    ka3 = 0.005   # very slow lysosomal release (5.5-month half-life)
    if doVariabilityPK: #coefficient of variability in a logNormal distribution
        CV_F = 0.23
        CV_ka1 = 0.45 #ka1 = absorption from Gastro Intestinal;
        CV_ka2 = 0.45 #ka2 = absorption from extracell to intracell
        CV_ka3 = 0.45 #ka2 = relase from intracell to extracell
        CV_ke = 0.3 # ke = clearance from extracell compartment
    n_don = 5
    
    concentrations = [\
                        0,0,0,0,38, \
                        0,0,0,185,289, \
                        179,0,19,19,457, \
                        457,77,95,95,557, \
                        5,33,10,51,847, \
                        0,25,44,44,1810, \
                        12,45,253,253,5300, \
                        0,8,26,165,4350, \
                        503,663,661,1010,1910, \
                        2050,1030,1060,1220,4040, \
                        325,699,745,1090,2600]
    
    obs_times = [1,1,1,1,3,3,4,4,26,26,28]


#TBAJ587
elif drug_name == 'TBAJ587':
    
    D_out = 700000
    D_out_change = 200000

    MBC90 = 332 #drug concentration to kill TBslow  MW= 615g/mol x MBC90= 0,54 uM = 332 ug/l = 332 ng/ml paper Bustion 2025
    D_max = 9170
    D_min = 5870
    shape = 83
    b = 35
    elim_rate = 0.19
    C0 = 7000
    Len_x_fit = 4000
    if Len_x == None:
        Len_x = Len_x_fit
    if dx == None:
        dx = 40
    out_don_thickness = 500 #um the thickness of the outer donuts
    
    F = 1
    ka1 = 1.2     # rapid gastrointestinal absorption
    ke  = 6.5     # rapid clearance driven by local blood flow
    ka2 = 8.0    # rapid, strong lysosomal uptake
    ka3 = 0.0025   # very slow lysosomal release (5.5-month half-life)
    if doVariabilityPK: #coefficient of variability in a logNormal distribution
        CV_F = 0.23
        CV_ka1 = 0.45 #ka1 = absorption from Gastro Intestinal;
        CV_ka2 = 0.45 #ka2 = absorption from extracell to intracell
        CV_ka3 = 0.45 #ka2 = relase from intracell to extracell
        CV_ke = 0.3 # ke = clearance from extracell compartment
    n_don = 4

    concentrations = [\
    0, 126, 761, 14800, \
    109, 313, 615, 7710, \
    23, 157, 1820, 14900, \
    87, 269, 1630, 19000, \
    96, 167, 1310, 18700, \
    42, 193, 1240, 14100, \
    
    427, 665, 854, 7480, \
    592, 1130, 1470, 4120, \
    198, 511, 946, 5510, \
    853, 1510, 3400, 8220, \
    53.3, 288, 1130, 6640, \
    68.4, 248, 426, 7240, \
    275, 12.3, 585, 4990, \
    2900, 2310, 2810, 6130 \
                      ]
    obs_times = [3,3,3,2,3,3,13,13,13,28,28,28,28,28]


Len_x = float(os.environ.get("TB_DIAMETER", Len_x))
Len_x = round(Len_x/dx)*dx
b_param = int(b*Len_x_fit/100) #b*Len_x_fit/100 in um
n_don = int(os.environ.get("TB_N_DON", n_don))
C0drug = C0
T = int(os.environ.get("TB_T_END", "350"))  # final time in days

#Drug
t_change_dose = 3
t_stop_dose = 180
t_start_dose = 0 #first dose after 12-16 weeks (day 84-112)
t_dissipation_drug = 170


if t_stop_dose <= T:
    times = np.arange(dt2,t_stop_dose + dt2, dt2)
    times = np.round(times, 3)
    tim_diss = np.arange(t_stop_dose+dt3,T + dt3, dt3)
    tim_diss = np.round(tim_diss, 3)
    times = np.concatenate((times, tim_diss))
    index_tstop = np.searchsorted(times, t_stop_dose)
else:
    times = np.arange(dt2,T + dt2, dt2)
    times = np.round(times, 3)
    index_tstop = -1    


if D_out_change != None and t_change_dose != None:
    doses = [(float(t), F*D_out) for t in range(t_change_dose)] + [(float(t), F*D_out_change) for t in range(t_change_dose, t_stop_dose)]
else:
    doses = [(float(t), F*D_out) for t in range(t_stop_dose)]

n_times = len(times)

params_pk = (ka1, ke, ka2, ka3)
BC_conc = create_bc_function(doses=doses, params=params_pk, t_final=T)
 
Nx = round(Len_x/dx)+1
x_val = np.arange(0,Len_x + dx, dx)
ratio = Len_x/Len_x_fit
Ds = hill_Ds(Nx, D_max, D_min, shape, b, ratio=ratio)


D_case = '2Dhill'
DDrug_title = f'{D_case} Dmax{D_max}_Dmin{D_min}_shape{shape}_b{b}_Dout{D_out}_elimrate{elim_rate}'

n_sim = int(os.environ.get("TB_N_SIM", "100"))  # PK variability replicates

scheme = 'BTCS_thomas'


#%% variance on PK Drug parameters ka and ke
seed=111111
# for a lognormal(mu, sigma), if we want for example 20% of variance coefficient
# variance sigma^2 = ln(var/(m^2) + 1) 
# mean mu = ln(m) - sigma^2/2
# sqrt(var) = 0.2*m
# => var = 0.04*m^2
# => sigma^2 = ln(1 + 0.04)
# => sigma = sqrt(ln(1 + 0.04))
#variability on PK
if doVariabilityPK:
    plot_trasparence = 1/(10*(1+np.log(n_sim))) #for the trasparence in the plotFit
    lw = 2.5 #line width in the plot

    seed = rand.randint(1000,10000000,1)[0]
    rand.seed(seed)
    sigma_F = np.sqrt(np.log(1 + CV_F**2))
    sigma_ka1 = np.sqrt(np.log(1 + CV_ka1**2))
    sigma_ka2 = np.sqrt(np.log(1 + CV_ka2**2))
    sigma_ka3 = np.sqrt(np.log(1 + CV_ka3**2))
    sigma_ke = np.sqrt(np.log(1 + CV_ke**2))
    
    F_array = rand.lognormal(mean = np.log(F) - (sigma_F**2)/2, sigma = sigma_F, size= n_sim)
    ka1_array = rand.lognormal(mean = np.log(ka1) - (sigma_ka1**2)/2, sigma = sigma_ka1, size= n_sim)
    ka2_array = rand.lognormal(mean = np.log(ka2) - (sigma_ka2**2)/2, sigma = sigma_ka2, size= n_sim)
    ka3_array = rand.lognormal(mean = np.log(ka3) - (sigma_ka3**2)/2, sigma = sigma_ka3, size= n_sim)
    ke_array = rand.lognormal(mean = np.log(ke) - (sigma_ke**2)/2, sigma = sigma_ke, size= n_sim)
    nbins = 50


#%% resolution 1D multi D

named_tuple = timing.localtime() # get struct_time
time_string = timing.strftime("%Y_%m_%d_%H%M%S", named_tuple)
folder = str(OUTPUT_DIR / (time_string+f'{drug_name}_Drug_simulation_1D_Tend_{T}'))
os.makedirs(folder, exist_ok=True)


    
#resolution
if doVariabilityPK:
    folder_base = folder
    
    if saveCSV and MBC90 != None:
        folderCSV_base = folder_base+'/csv'
        os.makedirs(folderCSV_base, exist_ok=True)
        if n_don == 4:
            header = ['time id', 'simulation n.', 'eradication time',\
                      'time to arrive MBC in don1', 'time to arrive MBC in don2',\
                      'time to arrive MBC in don3', 'time to arrive MBC in don4', 'time to arrive MBC centre', \
                        'time above MBC (% of treatment) in don1', 'time above MBC (% of treatment) in don2',\
                        'time above MBC (% of treatment) in don3', 'time above MBC (% of treatment) in don4', 'time above MBC (% of treatment) centre',\
                        'time above MBC after end in don1', 'time above MBC after end in don2',\
                        'time above MBC after end in don3', 'time above MBC after end in don4', 'time above MBC after end centre']
    
        elif n_don == 5:
            header = ['time id', 'simulation n.', 'eradication time',\
                      'time to arrive MBC in don1', 'time to arrive MBC in don2',\
                      'time to arrive MBC in don3', 'time to arrive MBC in don4', 'time to arrive MBC in don5', 'time to arrive MBC centre', \
                        'time above MBC (% of treatment) in don1', 'time above MBC (% of treatment) in don2',\
                        'time above MBC (% of treatment) in don3', 'time above MBC (% of treatment) in don4', 'time above MBC (% of treatment) in don5', 'time above MBC (% of treatment) centre',\
                        'time above MBC after end in don1', 'time above MBC after end in don2',\
                        'time above MBC after end in don3', 'time above MBC after end in don4', 'time above MBC after end in don5', 'time above MBC after end centre']
        csv_name_base = f"{time_string} MBC and eradication"
    
        with open(folderCSV_base+'/'+csv_name_base+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
    
    if doPlotDistributionVaribility:
        greenPlotDistributionVaribility = [] #for eradicated samples
        redPlotDistributionVaribility = [] #for samples without eradication
    
    # diameter
    R=Len_x/2
    dr = R/n_don #donut thickness
    r_circles = np.linspace(dr, R, n_don)
    r_circles[0] = round(out_don_thickness,3)
    if r_circles[0] >= r_circles[1]:
        print("WARNING: check r_circles and n_don")
    index = np.asarray(np.floor(r_circles/dx), dtype=int)
    mesh_inside_index_don_1d = []
    il=0
    for k in range(len(index)):
        ir = index[k] 
        mesh_inside_index_don_1d += [list(np.arange(il,ir)) + list(np.arange(Nx - ir, Nx - il))]
        il = ir
    centre_index = int(index[-1])
    mesh_inside_index_don_1d[-1] = np.sort(mesh_inside_index_don_1d[-1] + [centre_index]).tolist()
    mesh_inside_index_don_1d.reverse() #from inner to outer
    
    tic = timing.time()
    
    if doPlotFit:
        plt.ion() #plot interattivo
    else:
        plt.ioff()
    
    if doPlotFit or savePlotFit:
        #individual plots
        line_handles = [] #list of plotted lines
        fig2 = plt.figure(figsize= (12,8), layout='constrained')
        ax = fig2.add_subplot(111)
        ax.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        colors = ['g', 'r', 'b', 'c', 'm', 'y']
        for i in range(1, n_don + 1):
            line, = ax.plot(times, np.repeat(i, n_times), 
                            colors[(i - 1) % len(colors)] + '-', 
                            linewidth=2.5,
                            # label=f'Simulation in donut {i}')
                            label=f'Donut {i}')
            line_handles.append(line)
        ax.legend(loc = 'upper left', fontsize = 18)
        if plotMBC90:
            if MBC90 != None:
                ax.axhline(y=MBC90, color="red", alpha = 0.4, linewidth = 2)
                ax.annotate("CasMBC90", xy=(0.9*T, MBC90 + 250), color="r", size=22)
        ax.set_xlabel('Time (t) [day]', fontsize=18)
        ax.set_ylabel('Concentration [ng/g]', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(f"{drug_name} simulation", fontsize = 22, pad = 20, wrap = True)
        fig2.tight_layout()
        
        #plot with all simulations
        fig1 = plt.figure(figsize= (12,8), layout='constrained')
        ax1 = fig1.add_subplot(111)
        ax1.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
    
        #baseline simulation
        meshDrug = np.zeros(Nx)
        result_baseline=[] 
        drug_sim_bool = True
        tIni = 0
        for tEnd in times:
            if drug_sim_bool:
                if tIni > t_start_dose + t_stop_dose: #dt3
                    paramsDrug = [tIni - t_start_dose, tEnd - t_start_dose, mesh_inside_index_don_1d, meshDrug, Len_x, dx, dt3, BC_conc, elim_rate, t_stop_dose]  
                else: #dt2
                    paramsDrug = [tIni - t_start_dose, tEnd - t_start_dose, mesh_inside_index_don_1d, meshDrug, Len_x, dx, dt2, BC_conc, elim_rate, t_stop_dose]
            
            if drug_sim_bool:
                if tIni >= t_start_dose:
                    conc_meanDrug, meshDrug = simulation1D_polar_multiD_conservative_sigmoidal(Ds, paramsDrug) 
                if tIni >= t_start_dose + t_stop_dose + t_dissipation_drug:
                    drug_sim_bool = False
                    conc_meanDrug, meshDrug = np.zeros(n_don), np.zeros(Nx)
            
            tIni = tEnd
            result_baseline += [conc_meanDrug] #from don1(inner) to outer
        
        
        result_baseline = np.asarray(result_baseline).T
        for i in range(n_don):
            ax1.plot(times, result_baseline[i], colors[i]+'-',linewidth = 2.5, label = f'Donut {i+1}', zorder = n_don+10) #high z-order keeps this line in front
            ax1.plot(times, result_baseline[i], 'w-',linewidth = 4, zorder = n_don+9, alpha = 0.6)
        if plotObs:
            try:
                for i in range(len(concentrations)):
                    ax1.plot(obs_times[i//n_don], concentrations[i], colors[i%n_don]+'o', markersize=10, mec=(1, 1, 1, 0.6), mew=1.2, zorder = n_don+11) #translucent white border
            except Exception:
                pass
        
        legAx1 = ax1.legend(loc = 'upper left', fontsize = 18)
        legAx1.set_zorder(n_don + 12)
        if plotMBC90:
            if MBC90 != None:
                ax1.axhline(y=MBC90, color="red", alpha = 0.4, linewidth = 2, zorder = n_don+10)
                ax1.annotate("CasMBC90", xy=(0.9*T, MBC90 + 50), color="r", size=22)
        ax1.set_xlabel('Time (t) [day]', fontsize=18)
        ax1.set_ylabel('Concentration [ng/g]', fontsize=18)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.set_title(f"{drug_name} simulation", fontsize = 22, pad = 20, wrap = True)
        ax1.set_ylim(ax1.get_ylim())      # "Fissa" gli attuali ylim
        ax1.autoscale(False, axis='y')
        fig1.tight_layout()

            
    lastPointsAllSimulation = []
    eradicationTimes = []
    n_eradication = 0 
    for iteration in tqdm(range(n_sim)):
        folder = folder_base+f'/simulations/sim{10000001+iteration}'
        
        F_i = F_array[iteration]
        ka1_i = ka1_array[iteration]
        ka2_i = ka2_array[iteration]
        ka3_i = ka3_array[iteration]
        ke_i = ke_array[iteration]
        params_pk_i = (ka1_i, ke_i, ka2_i, ka3_i)
        doses_i = [(t, dose * F_i) for t, dose in doses]
        BC_conc_i = create_bc_function(doses=doses_i, params=params_pk_i, t_final=T)

        
        if saveCSV:
            folderCSV = folder+'/csv'
            os.makedirs(folderCSV, exist_ok=True)
            if n_don == 4:
                header = ['time id', 'time',\
                          'sim C in don1', 'sim C in don2', 'sim C in don3', 'sim C in don4', 'centre']
        
            elif n_don == 5:
                header = ['time id', 'time',\
                          'sim C in don1', 'sim C in don2', 'sim C in don3', 'sim C in don4', 'sim C in don5', 'centre']
            csv_name = f"{time_string} simulation 1D {scheme} and D {D_case}"
            csv_path = os.path.join(folderCSV, csv_name + '.csv')
            with open(csv_path, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
        
        
        
        meshDrug = np.zeros(Nx)
        
        if saveCSV: #close this file after the loop
            f_csv = open(csv_path, 'a', encoding='UTF8', newline='')
            writer = csv.writer(f_csv)                
            
        drug_sim_bool = True
        
        
        result = []
        result_centre = []
        tIni = 0
        for tEnd in times:

            if drug_sim_bool:
                if tIni > t_start_dose + t_stop_dose: #dt3
                    paramsDrug = [tIni - t_start_dose, tEnd - t_start_dose, mesh_inside_index_don_1d, meshDrug, Len_x, dx, dt3, BC_conc_i, elim_rate, t_stop_dose]  
                else: #dt2
                    paramsDrug = [tIni - t_start_dose, tEnd - t_start_dose, mesh_inside_index_don_1d, meshDrug, Len_x, dx, dt2, BC_conc_i, elim_rate, t_stop_dose]
            
            if drug_sim_bool:
                if tIni >= t_start_dose:
                    conc_meanDrug, meshDrug = simulation1D_polar_multiD_conservative_sigmoidal(Ds, paramsDrug) 
                if tIni >= t_start_dose + t_stop_dose + t_dissipation_drug:
                    drug_sim_bool = False
                    conc_meanDrug, meshDrug = np.zeros(n_don), np.zeros(Nx)
                    
            result += [conc_meanDrug] #from don1(inner) to outer [time x n_don]
            result_centre += [meshDrug[centre_index]]
 
            if saveCSV:
                csv_row = [time_string, tEnd]
                csv_row = np.concatenate((csv_row, conc_meanDrug))
                csv_row = np.concatenate((csv_row, [meshDrug[centre_index]]))
                csv_row = [eval(i) for i in csv_row] #now we have int and not string
                csv_row = np.asarray(np.round(csv_row, 2),dtype=float)
                writer.writerow(csv_row)
            
         
            tIni = tEnd

        if saveCSV:
            f_csv.close()
        
        result = np.asarray(result).T #[n_don x time]
        result_centre = np.asarray(result_centre) #[time]
        lastPointsAllSimulation += [result[-1,index_tstop]] #[n sim]
        t_plot = times
        if doPlotFit:
            for i in range(n_don):
                # update y data; x positions remain fixed
                y_new = result[i]  # or the appropriate source
                ax.set_ylim(0,1.1*np.max(y_new))
                line_handles[i].set_ydata(y_new)
                # if i == 0:
                    # ax1.plot(t_plot, y_new, colors[i]+'-', alpha = plot_trasparence, lw= lw, zorder = 1)
                ax1.plot(t_plot, y_new, colors[i]+'-', alpha = plot_trasparence, lw= lw, zorder = 1)
        
            refresh_figure(fig1)
            
            figName = f"{time_string}simulation_with_D_{DDrug_title}"
            if savePlotFit:
                fig2.savefig(folder+'/'+figName+'.png',bbox_inches='tight')
        elif savePlotFit:
            for i in range(n_don):
                y_new = result[i]  # or the appropriate source
                ax.set_ylim(0,1.1*np.max(y_new))
                line_handles[i].set_ydata(y_new)
                ax1.plot(t_plot, y_new, colors[i]+'-', alpha = plot_trasparence, lw = lw, zorder =1)
            figName = f"{time_string}simulation_with_D_{DDrug_title}"
            fig2.savefig(folder+'/'+figName+'.png',bbox_inches='tight')

        
        if MBC90 != None:
            MBC90_time_above = [] #time above MBC90 in % in tratement period in each donut from inner to outer and last value is the centre
            MBC90_start_time_above = []
            MBC90_end_time_above = [] #time above after treatment end
            for i in range(n_don):
                conditionMBC = result[i] >= MBC90 
                indexMBC = np.where(conditionMBC)[0]
                if len(indexMBC) > 0:
                    ind1 = indexMBC[0]
                    ind2 = indexMBC[-1]
                    t1 = t_plot[ind1]
                    t2 = t_plot[ind2]
                    MBC90_start_time_above += [t1]
                    MBC90_end_time_above += [t2-t_stop_dose]
                    MBC90_time_above += [np.round((t_stop_dose - t1)/t_stop_dose,4)]
                else:
                    # print("CasMBC90 was not reached")
                    MBC90_start_time_above += ['CasMBC90 not reached']
                    MBC90_end_time_above += ['CasMBC90 not reached']
                    MBC90_time_above += ['CasMBC90 not reached']
            
            conditionMBC = result_centre >= MBC90
            indexMBC = np.where(conditionMBC)[0]
            if len(indexMBC) > 0:
                n_eradication += 1
                ind1 = indexMBC[0]
                ind2 = indexMBC[-1]
                t1 = t_plot[ind1]
                t2 = t_plot[ind2]
                MBC90_start_time_above += [t1]
                eradicationTimes += [t1]
                MBC90_end_time_above += [t2-t_stop_dose]
                MBC90_time_above += [np.round((t_stop_dose - t1)/t_stop_dose,4)]
                if doPlotDistributionVaribility:
                    greenPlotDistributionVaribility += [[ka1_i,ke_i,ka2_i,ka3_i]]
            else:
                print("CasMBC90 was not reached")
                MBC90_start_time_above += ['CasMBC90 not reached']
                MBC90_end_time_above += ['CasMBC90 not reached']
                MBC90_time_above += ['CasMBC90 not reached']
                if doPlotDistributionVaribility:
                    redPlotDistributionVaribility += [[ka1_i,ke_i,ka2_i,ka3_i]]
            
            if saveCSV:
                csv_row = [eval(time_string), iteration+1, MBC90_start_time_above[-1]] + MBC90_start_time_above + MBC90_time_above + MBC90_end_time_above
                csv_row = [round(x, 2) if isinstance(x, (int, float)) else x for x in csv_row]
                csv_row = np.asarray(csv_row, dtype=object)
                with open(folderCSV_base+'/'+csv_name_base+'.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    # write multiple rows
                    writer.writerows([csv_row])
                
        # plt.close() 
        with open(folder+f'/{time_string}_Drug__params.txt', 'w', encoding='UTF8', newline='') as f:
            f.write(f'{time_string} Drug PK parameters: \n')
            f.write(f'{drug_name} \n')
            f.write(f'ka1 = {ka1_i}; ke = {ke_i}; ka2 = {ka2_i}; ka3 = {ka3_i}\n\n')
            if MBC90 != None:
                f.write(f'Time to arrive MBC90 in each donut from inner to outer and the last value in the centre\n')
                f.write(f'{MBC90_start_time_above}\n')
                f.write(f'Time above MBC90 (in %) during the tratment in each donut from inner to outer and the last value in the centre\n')
                f.write(f'{MBC90_time_above}\n')
                f.write(f'Time above MBC90 (in day) after tratment end in each donut from inner to outer and the last value in the centre\n')
                f.write(f'{MBC90_end_time_above}\n')
                f.write(f'Eradication time (same of time to arrive MBC90 in centre)\n')
                f.write(f'{MBC90_start_time_above[-1]}\n')
    
    
    # Fit a shifted lognormal only when enough distinct threshold times exist.
    fit_available = len(eradicationTimes) >= 3 and np.ptp(eradicationTimes) > 0
    if fit_available:
        try:
            sigma, shift, scale = scipy.stats.lognorm.fit(eradicationTimes)
            sigma2 = sigma**2
            meanLogN = scale*np.exp(sigma2/2) + shift
            varLogN = (np.exp(sigma2)-1)*np.exp(sigma2)*scale**2
            ks_stat, p_value_ks = scipy.stats.kstest(
                eradicationTimes, 'lognorm', args=(sigma, shift, scale))
            log_erT = np.log(np.array(eradicationTimes) - shift)
            sw_stat, p_value_sw = scipy.stats.shapiro(log_erT)
        except Exception as exc:
            fit_available = False
            print(f"Skipping shifted-lognormal diagnostics: {exc}")

    if doPlotDistributionVaribility and MBC90 != None and fit_available:
        if np.size(greenPlotDistributionVaribility) != 0:
            greenPlotDistributionVaribilityT = np.asarray(greenPlotDistributionVaribility).T #[4 x kas(green)]
            ka1Green = greenPlotDistributionVaribilityT[0]
            keGreen = greenPlotDistributionVaribilityT[1]
            ka2Green = greenPlotDistributionVaribilityT[2]
            ka3Green = greenPlotDistributionVaribilityT[3]
        else:
            ka1Green = np.array([])
            keGreen = np.array([])
            ka2Green = np.array([])
            ka3Green = np.array([])
        if np.size(redPlotDistributionVaribility) != 0:
            redPlotDistributionVaribilityT = np.asarray(redPlotDistributionVaribility).T #[4 x kas(red)]
            ka1Red = redPlotDistributionVaribilityT[0]
            keRed = redPlotDistributionVaribilityT[1]
            ka2Red = redPlotDistributionVaribilityT[2]
            ka3Red = redPlotDistributionVaribilityT[3]
        else:
            ka1Red = np.array([])
            keRed = np.array([])
            ka2Red = np.array([])
            ka3Red = np.array([])
        ka1Plot = [ka1Green, ka1Red]
        kePlot = [keGreen, keRed]
        ka2Plot = [ka2Green, ka2Red]
        ka3Plot = [ka3Green, ka3Red]
        
        colors = ['green', 'red']
        
        fig10 = plt.figure(figsize= (12,16), layout='constrained')
        ax10 = fig10.add_subplot(411)
        ax20 = fig10.add_subplot(412)
        ax30 = fig10.add_subplot(413)
        ax40 = fig10.add_subplot(414)
        ax10.hist(ka1Plot, nbins, density=True, histtype='bar', stacked=True, facecolor = colors, label=('CasMBC90 achieved', 'CasMBC90 not achieved'))
        ax20.hist(kePlot, nbins, density=True, histtype='bar', stacked=True, facecolor = colors)
        ax30.hist(ka2Plot, nbins, density=True, histtype='bar', stacked=True, facecolor = colors)
        ax40.hist(ka3Plot, nbins, density=True, histtype='bar', stacked=True, facecolor = colors)
        ax10.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax20.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax30.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax40.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax10.legend(loc = 'upper right', fontsize = 18)
        ax10.set_xlabel('$k_a1$ value', fontsize=18)
        ax20.set_xlabel('$k_e$ value', fontsize=18)
        ax30.set_xlabel('$k_a2$ value', fontsize=18)
        ax40.set_xlabel('$k_a3$ value', fontsize=18)
        ax10.set_ylabel('Density', fontsize=18)
        ax20.set_ylabel('Density', fontsize=18)
        ax30.set_ylabel('Density', fontsize=18)
        ax40.set_ylabel('Density', fontsize=18)
        ax10.tick_params(axis='both', labelsize=18)
        ax20.tick_params(axis='both', labelsize=18)
        ax30.tick_params(axis='both', labelsize=18)
        ax40.tick_params(axis='both', labelsize=18)
        # ax1.set_title(f"{drug_name} simulation", fontsize = 22, pad = 20, wrap = True)
        
        fig20 = plt.figure(figsize= (12,8), layout='constrained')
        ax10 = fig20.add_subplot(411)
        ax20 = fig20.add_subplot(412)
        ax30 = fig20.add_subplot(413)
        ax40 = fig20.add_subplot(414)
        ax10.hist(ka1Plot, nbins, histtype='bar', color=colors, label=('CasMBC90 achieved', 'CasMBC90 not achieved'))
        ax20.hist(kePlot, nbins, histtype='bar', color=colors)
        ax30.hist(ka2Plot, nbins, histtype='bar', color=colors)
        ax40.hist(ka3Plot, nbins, histtype='bar', color=colors)
        ax10.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax20.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax30.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax40.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax10.legend(loc = 'upper right', fontsize = 18)
        ax10.set_xlabel('$k_a1$ value', fontsize=18)
        ax20.set_xlabel('$k_e$ value', fontsize=18)
        ax30.set_xlabel('$k_a2$ value', fontsize=18)
        ax40.set_xlabel('$k_a3$ value', fontsize=18)
        ax10.set_ylabel('Quantity', fontsize=18)
        ax20.set_ylabel('Quantity', fontsize=18)
        ax30.set_ylabel('Quantity', fontsize=18)
        ax40.set_ylabel('Quantity', fontsize=18)
        ax10.tick_params(axis='both', labelsize=18)
        ax20.tick_params(axis='both', labelsize=18)
        ax30.tick_params(axis='both', labelsize=18)
        ax40.tick_params(axis='both', labelsize=18)
        # ax1.set_title(f"{drug_name} simulation", fontsize = 22, pad = 20, wrap = True)
        
        fig30 = plt.figure(figsize= (12,8), layout='constrained')
        # # Stima dei parametri lognormali (fit)
        # # sigma, shift, mu = scipy.stats.lognorm.fit(eradicationTimes, floc=0)  # fix shift at zero because lognormal is defined for positive values
        # sigma, shift, scale = scipy.stats.lognorm.fit(eradicationTimes) #scale = exp(mu), where mu and sigma describe the normal distribution
        # sigma2 = sigma**2
        # meanLogN = scale*np.exp((sigma2)/2) + shift
        # varLogN = (np.exp(sigma2)-1)*np.exp(sigma2)*scale**2
        x_vals = np.linspace(min(eradicationTimes), max(eradicationTimes), 500)
        pdf_vals = scipy.stats.lognorm.pdf(x_vals, sigma, loc=shift, scale=scale)
        
        ax50 = fig30.add_subplot(111)
        ax50.hist(eradicationTimes, nbins, density=True, histtype='bar')
        ax50.plot(x_vals, pdf_vals, 'b-', lw=2, label='PDF lognormal')
        ax50.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        ax50.legend(loc = 'upper right', fontsize = 18)
        ax50.set_xlabel('Time', fontsize=18)
        ax50.set_ylabel('Quantity', fontsize=18)
        ax50.tick_params(axis='both', labelsize=18)
        
        fig10.tight_layout()
        fig20.tight_layout()
        fig30.tight_layout()
        plt.show()
        if savePlotDistributionVaribility:
            fig10.savefig(folder_base+'/distribution_ka_ke.png',bbox_inches='tight')
            fig20.savefig(folder_base+'/distribution_ka_ke2.png',bbox_inches='tight')
            fig30.savefig(folder_base+'/distribution_eradicationTime.png',bbox_inches='tight')
    
    if savePlotFit:
        fig1.savefig(folder_base+'/cumulative_fit.png',bbox_inches='tight')
    
    # stop tracking computational cost
    toc = timing.time()     
    # compute computational cost
    wallClockTime = toc - tic
    print(wallClockTime)
    print(f'eradication prob: {np.round(n_eradication/n_sim,3)}')
    with open(folder_base+f'/{time_string}_Drug_{drug_name}_params.txt', 'w', encoding='UTF8', newline='') as f:
        f.write(f'{time_string} Drug simulation parameters: \n')
        f.write(f'T end:{T}; seed: {seed} \n')
        f.write(f'Drug parameters: \n')
        f.write(f'{drug_name} \n')
        f.write(f'CV ka1: {CV_ka1}, ke: {CV_ke}, ka2: {CV_ka2}, ka3: {CV_ka3};\n')
        f.write(f'time step dt:{dt2} day; \n')
        f.write(f'time start {t_start_dose} day, time stop {t_stop_dose} day, dissipation period {t_dissipation_drug} day\n')
        f.write(f'size:{Len_x} um; \n')
        f.write(f'space step dx:{dx} um\n')
        f.write(f'D_out:{D_out}; (dose in the 2 compartment absorption model) \n')
        f.write(f'D_max:{D_max}; um^2/day \n')
        f.write(f'D_min{D_min}; um^2/day \n')
        f.write(f'k:{shape}; (for the shape) \n')
        f.write(f'b:{b} unitless; {b_param} um; (b*Len_x_fit/100) \n')
        f.write(f'elimination rate:{elim_rate} 1/day; \n')
        f.write(f'n_donuts:{n_don}; \n')
        f.write(f'Len_x fit:{Len_x_fit} um \n\n')
        f.write(f'outer donut thickness:{out_don_thickness} um (it is only there the consumption of drug)\n\n')
        
        if MBC90 != None and fit_available:
            f.write(f'Eradication time distribution is a Lognormal with parameters\n')
            f.write(f'scale: {scale} = e^mu, sigma: {sigma}, shift: {shift}\n')
            f.write(f'hence the mean and variance of the LogNormal are:\n')
            f.write(f'E[X]: {meanLogN}, Var(X): {varLogN}\n\n')
            
            f.write(f"Kolmogorov-Smirnov test:\n")
            f.write(f"KS stat: {ks_stat:.4f}\n")
            f.write(f"p-value: {p_value_ks:.4f}\n")
            if p_value_ks > 0.05:
                f.write("Result: the shifted lognormal null hypothesis was not rejected.\n")
            else:
                f.write("Result: the shifted lognormal null hypothesis was rejected.\n")
            
            f.write(f"Shapiro-Wilk test:\n")
            f.write(f"SW stat: {sw_stat:.4f}, P-value: {p_value_sw:.4f}\n")

            if p_value_sw > 0.05:
                f.write("The null hypothesis was not rejected: data are consistent with a shifted lognormal distribution.\n")
            else:
                f.write("The null hypothesis was rejected: data are inconsistent with a lognormal distribution.\n")

        if MBC90 is not None and not fit_available:
            f.write('Shifted-lognormal diagnostics skipped: fewer than three distinct threshold times.\n')

    
        f.write(f'Execution time:{round(wallClockTime,2)} sec\n')
        f.write(f'Threshold-attainment fraction: {n_eradication}/{n_sim} = {np.round(n_eradication/n_sim,3)}')





else:
    if savePlot:
        folderPNG = folder+'/png'
        os.makedirs(folderPNG, exist_ok=True)
        
    if saveCSV:
        folderCSV = folder+'/csv'
        os.makedirs(folderCSV, exist_ok=True)
        if n_don == 4:
            header = ['time id', 'time',\
                      'sim C in don1', 'sim C in don2', 'sim C in don3', 'sim C in don4', 'centre']

        elif n_don == 5:
            header = ['time id', 'time',\
                      'sim C in don1', 'sim C in don2', 'sim C in don3', 'sim C in don4', 'sim C in don5', 'centre']
        csv_name = f"{time_string} simulation 1D {scheme} and D {D_case}"

        with open(folderCSV+'/'+csv_name+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
    

    # diameter
    R=Len_x/2
    dr = R/n_don #donut thickness
    r_circles = np.linspace(dr, R, n_don)
    r_circles[0] = round(out_don_thickness,3)
    if r_circles[0] >= r_circles[1]:
        print("WARNING: check r_circles and n_don")
    index = np.asarray(np.floor(r_circles/dx), dtype=int)
    mesh_inside_index_don_1d = []
    il=0
    for k in range(len(index)):
        ir = index[k] 
        mesh_inside_index_don_1d += [list(np.arange(il,ir)) + list(np.arange(Nx - ir, Nx - il))]
        il = ir
    centre_index = int(index[-1])
    mesh_inside_index_don_1d[-1] = np.sort(mesh_inside_index_don_1d[-1] + [centre_index]).tolist()
    mesh_inside_index_don_1d.reverse() #from inner to outer

    if doPlot or savePlot:
        # 'gridding' the 1D arrays into 2D arrays so they can be used by pcolor
        x_val_plot, y_val_plot = np.meshgrid(x_val, [1,2])
        plt.close('all')
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax.pcolor(x_val_plot, y_val_plot, [meshDrug, meshDrug], edgecolors='k', linewidths=0, cmap="Oranges", vmin=0, vmax=C0)
        plt.tight_layout()
        # plt.pause(1)
    
    
    
    time = 0.
    meshDrug = np.zeros(Nx)
    result=[]
    result_centre = []
    drug_sim_bool = True
    tIni = 0
    tic = timing.time()
    for tEnd in times:
        if drug_sim_bool:
            if tIni > t_start_dose + t_stop_dose: #dt3
                paramsDrug = [tIni - t_start_dose, tEnd - t_start_dose, mesh_inside_index_don_1d, meshDrug, Len_x, dx, dt3, BC_conc, elim_rate, t_stop_dose]  
            else: #dt2
                paramsDrug = [tIni - t_start_dose, tEnd - t_start_dose, mesh_inside_index_don_1d, meshDrug, Len_x, dx, dt2, BC_conc, elim_rate, t_stop_dose]
        
        if drug_sim_bool:
            if tIni >= t_start_dose:
                conc_meanDrug, meshDrug = simulation1D_polar_multiD_conservative_sigmoidal(Ds, paramsDrug)
            if tIni >= t_start_dose + t_stop_dose + t_dissipation_drug:
                drug_sim_bool = False
                conc_meanDrug, meshDrug = np.zeros(n_don), np.zeros(Nx)
        
        if saveCSV:
            csv_row = [time_string, tEnd]        
            csv_row = np.concatenate((csv_row, conc_meanDrug))
            csv_row = np.concatenate((csv_row, [meshDrug[centre_index]]))
            csv_row = [eval(i) for i in csv_row] #now we have int and not string
            csv_row = np.asarray(np.round(csv_row, 2),dtype=float)
            with open(folderCSV+'/'+csv_name+'.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([csv_row])

        result += [conc_meanDrug] #from don1(inner) to outer [time x n_don]
        result_centre += [meshDrug[centre_index]]
     
        tIni = tEnd

        if doPlot or savePlot:
            for don,i in zip(mesh_inside_index_don_1d, range(n_don)):
                mesh_mean = meshDrug.copy()
                mesh_mean[don] = conc_meanDrug[i]

            ax = plt.subplot(221)
            ax2 = plt.subplot(222)
            c = ax.pcolor(x_val_plot, y_val_plot ,[mesh, mesh], edgecolors='k', linewidths=0, cmap="Greens", vmin=0, vmax=C0)
            plt.suptitle(f"time: {tEnd}", wrap = True)
            ax2.pcolor(x_val_plot, y_val_plot, [mesh_mean, mesh_mean], shading="nearest", edgecolors='k', linewidths=0, cmap="Greens", vmin=0, vmax=C0)
            plt.colorbar(c, ax=ax2)
            plt.tight_layout()
            refresh_figure(fig)
            
            if savePlot:
                figName = f"{time_string}simulation_1D_with_D_sigmoidal_time_{10000+tEnd}"
                figName = figName.translate(str.maketrans('', '', '.'))
                plt.savefig(folderPNG+'/'+figName+'.png',bbox_inches='tight')

    
    
    result = np.asarray(result).T #[n_don x time]
    result_centre = np.asarray(result_centre) #[time]
    t_plot = times
    if doPlotFit or savePlotFit:
        fig2 = plt.figure(figsize= (12,8), layout='constrained')
        ax = fig2.add_subplot(111)
        ax.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        colors = ['g', 'r', 'b', 'c', 'm', 'y']
        for i in range(n_don):
            ax.plot(t_plot, result[i], colors[i]+'-',linewidth = 2.5, label = f'Donut {i+1}') #high z-order keeps this line in front
        if plotObs:
            try:
                for i in range(len(concentrations)):
                    ax.plot(obs_times[i//n_don], concentrations[i], colors[i%n_don]+'o', markersize=10, mec=(1, 1, 1, 0.6), mew=1.2, zorder = n_don+11) #translucent white border
            except Exception:
                pass
        
        ax.legend(loc = 'upper right', fontsize = 18)
        if plotMBC90:
            if MBC90 != None:
                ax.axhline(y=MBC90, color="red", alpha = 0.4, linewidth = 2)
                ax.annotate("CasMBC90", xy=(0.88*T, MBC90 + 150), color="r", size=20)
        ax.set_xlabel('Time (t) [day]', fontsize=18)
        ax.set_ylabel('Concentration [ng/g]', fontsize=18)
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(f"{drug_name} simulation", fontsize = 22, pad = 20, wrap = True)
        plt.tight_layout()
        
        figName = f"{time_string}simulation_with_D_{DDrug_title}"
        if savePlotFit:
            fig2.savefig(folder+'/'+figName+'.png',bbox_inches='tight')
        if doPlotFit:
            plt.show()

    if MBC90 != None:
        MBC90_time_above = [] #time above MBC90 in % in tratement period in each donut from inner to outer and last value is the centre
        MBC90_start_time_above = []
        MBC90_end_time_above = [] #time above after treatment end
        
        for i in range(n_don):
            conditionMBC = result[i] >= MBC90 
            indexMBC = np.where(conditionMBC)[0]
            if len(indexMBC) > 0:
                ind1 = indexMBC[0]
                ind2 = indexMBC[-1]
                t1 = t_plot[ind1]
                t2 = t_plot[ind2]
                MBC90_start_time_above += [t1]
                MBC90_end_time_above += [t2-t_stop_dose]
                MBC90_time_above += [np.round((t_stop_dose - t1)/t_stop_dose,4)]
            else:
                print("CasMBC90 was not reached")
                MBC90_start_time_above += ['CasMBC90 not reached']
                MBC90_end_time_above += ['CasMBC90 not reached']
                MBC90_time_above += ['CasMBC90 not reached']
        
        conditionMBC = result_centre >= MBC90
        indexMBC = np.where(conditionMBC)[0]
        if len(indexMBC) > 0:
            ind1 = indexMBC[0]
            ind2 = indexMBC[-1]
            t1 = t_plot[ind1]
            t2 = t_plot[ind2]
            MBC90_start_time_above += [t1]
            MBC90_end_time_above += [t2-t_stop_dose]
            MBC90_time_above += [np.round((t_stop_dose - t1)/t_stop_dose,4)]
        else:
            print("CasMBC90 was not reached")
            MBC90_start_time_above += ['CasMBC90 not reached']
            MBC90_end_time_above += ['CasMBC90 not reached']
            MBC90_time_above += ['CasMBC90 not reached']
            if doPlotDistributionVaribility:
                redPlotDistributionVaribility += [[ka_i,ke_i]]
        
                
    
    
    # stop tracking computational cost
    toc = timing.time()     
    # compute computational cost
    wallClockTime = toc - tic
    print(wallClockTime)
        
    with open(folder+f'/{time_string}_Drug_{drug_name}_params.txt', 'w', encoding='UTF8', newline='') as f:
        f.write(f'{time_string} Drug simulation parameters: \n')
        f.write(f'T end:{T}')
        f.write(f'Drug parameters: \n')
        f.write(f' {drug_name}, dt:{dt2} day, size:{Len_x}, Drug start:{t_start_dose} day, Drug change dose after: {t_change_dose} day; \n')
        f.write(f' ka1: {ka1}, ke: {ke}, ka2: {ka2}, ka3: {ka3}, D_out:{D_out}, D_max:{D_max}, D_min{D_min}, shape:{shape}, b:{b}, elimination rate:{elim_rate}, n_donuts:{n_don} \n \n')
        f.write(f'size:{Len_x} um; \n')
        f.write(f'space step dx:{dx} um\n')
        f.write(f'Len_x fit:{Len_x_fit} um \n\n')
        f.write(f'outer donut thickness:{out_don_thickness} um (it is only there the consumption of drug)\n\n')
        if MBC90 != None:
            f.write(f'Time to arrive MBC90 in each donut from inner to outer and the last value in the centre\n')
            f.write(f'{MBC90_start_time_above}\n')
            f.write(f'Time above MBC90 (in %) during the tratment in each donut from inner to outer and the last value in the centre\n')
            f.write(f'{MBC90_time_above}\n')
            f.write(f'Time above MBC90 (in day) after tratment end in each donut from inner to outer and the last value in the centre\n')
            f.write(f'{MBC90_end_time_above}\n')
            f.write(f'Eradication time (same of time to arrive MBC90 in centre)\n')
            f.write(f'{MBC90_start_time_above[-1]}\n')
        f.write(f'Execution time:{round(wallClockTime,2)} sec')
