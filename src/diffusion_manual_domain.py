#%% import
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import imageio
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import matplotlib.colors as mpcolors
import seaborn as sns
import time as timing
from tqdm import tqdm
from functools import reduce
from matplotlib import path
import scipy.optimize as opt
import SelectFromCollection as SelectFromCollection
import shapely as shapely
from ast import literal_eval
import pandas as pd
import csv
import os
import sys


from functions import *
from release_config import OUTPUT_DIR, project_path, selected_drug



plt.close('all')
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['savefig.transparent'] = False

#%% test parameters
doVariabilityPK = os.environ.get("TB_PK_VARIABILITY", "0") == "1"
doPlot = 0 #boolean
savePlot = 0
doPlotFit = 0  # Save final fit plots without opening a live GUI
savePlotFit = 1
doPlotDistributionVaribility = 0
savePlotDistributionVaribility = 0
saveCSV = 1

image_cover = bool(os.environ.get("TB_IMAGE"))  # optional background image for point selection
image_name = str(project_path(os.environ.get("TB_IMAGE", "data/geometry/image.png")))  # optional background image
size_point = 1 #point size in plot
name_csv_file = os.environ.get("TB_GEOMETRY")
if name_csv_file:
    name_csv_file = str(project_path(name_csv_file).with_suffix(""))




dx = 20 #space step (in both x and y)
dt2 = 0.2 #for the Drug and Ox simulations
dt3 = 1 #for Drug and Oxygen after treatment
Len_x= 4000 #mesh width (rounded to a multiple of dx)
Len_y= 5000#mesh height

# R=round(min(Len_x, Len_y)/2)
R=2500
centre_cerchio = [2500,2500]


# drug_name = 'BDQ'
drug_name = selected_drug()



# #BDQ
if drug_name == 'BDQ':
    D_out = 450000
    D_out_change = 150000 #after 3 days reduce the dose from 400 to 40

    # MBC90 = 2600 #drug concentration to kill TBslow  MW= 555g/mol x MBC90= 4,6 uM = 2600ug/l = 2600 ng/ml 
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


n_don = 5
F = 1  # baseline bioavailability for both drugs

Len_x = round(Len_x/dx)*dx
Len_y = round(Len_y/dx)*dx
b_param = int(b*Len_x_fit/100) #b*Len_x_fit/100 in um
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
    doses = [(float(t), D_out) for t in range(t_change_dose)] + [(float(t), D_out_change) for t in range(t_change_dose, t_stop_dose)]
else:
    doses = [(float(t), D_out) for t in range(t_stop_dose)]

n_times = len(times)

params_pk = (ka1, ke, ka2, ka3)
BC_conc = create_bc_function(doses=doses, params=params_pk, t_final=T)
 
Nx = round(Len_x/dx)+1
Ny = round(Len_y/dx)+1
x_val = np.arange(0,Len_x + dx, dx)
y_val = np.arange(0,Len_y + dx, dx)
ratio = min(Len_x, Len_y)/Len_x_fit
Ds = hill_Ds(2*min(Nx,Ny)-1, D_max, D_min, shape, b, ratio=ratio)


D_case = '2Dhill'
DDrug_title = f'{D_case} Dmax{D_max}_Dmin{D_min}_shape{shape}_b{b}_Dout{D_out}_elimrate{elim_rate}'

#%% creation of the mesh, boundary conditions and initial conditions

if name_csv_file == None:

    if image_cover==True:
        im = plt.imread(image_name)
        fig, ax = plt.subplots()
        im = ax.imshow(im, extent=[0, Nx-1, 0, Ny-1])
    else:
        fig, ax = plt.subplots()
    
    grid_size_y = Ny
    grid_size_x = Nx
    grid_x = np.tile(np.arange(grid_size_x), grid_size_y)
    grid_y = np.repeat(np.arange(grid_size_y), grid_size_x)
    pts = ax.scatter(grid_x, grid_y, s=size_point)
    
    selector = SelectFromCollection.SelectFromCollection(ax, pts)
    selected_points = []
    
    def accept(event):
        if event.key == "enter":
            global selected_points
            selected_points = np.asarray(selector.xys[selector.ind], dtype=int)
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()
    
    fig.canvas.mpl_connect("key_press_event", accept)
    
    
    
    x_cerchio = np.linspace(0, 2*np.pi, 10**5)
    x_cerchio = R*np.cos(x_cerchio) + centre_cerchio[0]
    y_cerchio = np.linspace(0, 2*np.pi, 10**5)
    y_cerchio = R*np.sin(y_cerchio) + centre_cerchio[1]
    
    BC_index = []
    ind_x_BC = round(x_cerchio[0]/dx) #find the index of the mesh for x
    ind_y_BC = round(y_cerchio[0]/dx)
    BC_index = BC_index + [[ind_x_BC,ind_y_BC]]
    
    for i in range(1,len(x_cerchio)):
        ind_x_BC = round(x_cerchio[i]/dx) #find the index of the mesh for x
        ind_y_BC = round(y_cerchio[i]/dx)
        if [ind_x_BC,ind_y_BC] != BC_index[-1]:
            BC_index = BC_index + [[ind_x_BC,ind_y_BC]]
    
    ax.plot(np.transpose(BC_index)[0], np.transpose(BC_index)[1],'ro' )
    
    
    
    ax.set_title("Press enter to accept selected points.")
    
    plt.show()

#%%
if name_csv_file == None:
    BC_selected_points = []
    
    sel_p_list = selected_points.tolist()
    for el in sel_p_list:
        if ([el[0],el[1]+1] not in sel_p_list) or ([el[0],el[1]-1] not in sel_p_list)\
            or ([el[0]+1,el[1]] not in sel_p_list) or ([el[0]-1,el[1]] not in sel_p_list)\
            or ([el[0]+1,el[1]+1] not in sel_p_list) or ([el[0]-1,el[1]-1] not in sel_p_list)\
            or ([el[0]-1,el[1]+1] not in sel_p_list) or ([el[0]+1,el[1]-1] not in sel_p_list):
                BC_selected_points += [el]
    
    #discard isolated points; each boundary point needs at least two neighbors
    #this also removes single-cell branches
    remove_bol = True
    while remove_bol:
        remove_bol = False
        for el in BC_selected_points:
            vicini = 0
            if [el[0],el[1]+1] in BC_selected_points:
                vicini += 1
            if [el[0],el[1]-1] in BC_selected_points:
                vicini += 1
            if [el[0]+1,el[1]] in BC_selected_points:
                vicini += 1
            if [el[0]-1,el[1]] in BC_selected_points:
                vicini += 1
            if vicini < 2:
                BC_selected_points.remove(el)
                remove_bol=True
                # print(el)
    # print(BC_selected_points)


#%% visualize the circle and boundary
if name_csv_file == None:
    fig2, ax2 = plt.subplots()
    
    x_cerchio = np.linspace(0, 2*np.pi, 10**5)
    x_cerchio = R*np.cos(x_cerchio) + centre_cerchio[0]
    y_cerchio = np.linspace(0, 2*np.pi, 10**5)
    y_cerchio = R*np.sin(y_cerchio) + centre_cerchio[1]
    
    BC_index_cerchio = []
    ind_x_BC = round(x_cerchio[0]/dx) #find the index of the mesh for x
    ind_y_BC = round(y_cerchio[0]/dx)
    BC_index_cerchio = BC_index_cerchio + [[ind_x_BC,ind_y_BC]]
    
    for i in range(1,len(x_cerchio)):
        ind_x_BC = round(x_cerchio[i]/dx) #find the index of the mesh for x
        ind_y_BC = round(y_cerchio[i]/dx)
        if [ind_x_BC,ind_y_BC] != BC_index_cerchio[-1]:
            BC_index_cerchio = BC_index_cerchio + [[ind_x_BC,ind_y_BC]]
    
    ax2.plot(np.transpose(BC_index_cerchio)[0], np.transpose(BC_index_cerchio)[1],'bo-' )
    ax2.plot(np.transpose(BC_selected_points)[0], np.transpose(BC_selected_points)[1],'ro' )
    
    tellme('You will select the start and stop points (anti clockwise), click to begin')
    
    plt.waitforbuttonpress() #true key, false mouse click
    
    max_xlim = ax2.get_xlim() # get current x_limits to set max zoom out
    max_ylim = ax2.get_ylim() # get current y_limits to set max zoom out
    f = zoom_factory(ax2, max_xlim, max_ylim, base_scale=1.1)
    plt.show()
    
    
    while True:
        pts = []
        while len(pts) < 2:
            tellme('Select 2 points with mouse, left click for select and right click for delete (middle for stop), scroll to zoom NB: do not click to zoom')
            pts = np.asarray(plt.ginput(2, timeout=-1))
            if len(pts) < 2:
                tellme('Too few points, starting over')
                timing.sleep(1)  # Wait a second
        pts = np.asarray(np.around(pts), dtype=int)
        ph = ax2.plot(pts[:, 0], pts[:, 1], 'go', lw=2)
    
        tellme('Happy? Key click for yes, mouse click for no')
    
        if plt.waitforbuttonpress():
            break
    
        # Get rid of fill
        for p in ph:
            p.remove()


#%%
if name_csv_file == None:
    start = pts[0].tolist()
    stop = pts[1].tolist()
    start_circle = start
    stop_circle = stop
    BC_selected_points_sorted = sorted_BC_index(BC_selected_points, start,stop)
    BC_index = combine_BC_circle_and_other(BC_selected_points_sorted, start_circle, stop_circle, dx, Nx, Ny,R,centre_cerchio)
    
    named_tuple = timing.localtime() # get struct_time
    time_string = timing.strftime("%Y%m%d%H%M%S", named_tuple)
    header = ['time id', 'points']
    csv_name = f"{time_string}_BC_points"
    with open(project_path('data/geometry') / (csv_name+'.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        writer.writerow([time_string, BC_index])

#%% start there if you want to import the csv

if name_csv_file != None :
    csv_name = name_csv_file
    df = pd.read_csv(csv_name+'.csv')
    BC_index = df['points'][0]
    BC_index = literal_eval(BC_index)
    BC_ind_T = np.asarray(BC_index).T
    Nx=np.max(BC_ind_T[0])+1
    Ny=np.max(BC_ind_T[1])+1
    Len_x=(Nx-1)*dx
    Len_y=(Ny-1)*dx
    x_val = np.arange(0,Len_x + dx, dx)
    y_val = np.arange(0,Len_y + dx, dx)
    ratio = min(Len_x, Len_y)/Len_x_fit
    Ds = hill_Ds(2*min(Nx,Ny)-1, D_max, D_min, shape, b, ratio=ratio)
    fig_BC = plt.figure(figsize= (12,8), layout='constrained')
    ax_BC = fig_BC.add_subplot(111)
    ax_BC.plot(BC_ind_T[0], BC_ind_T[1], 'o')
    BC_index_tupla = tuple(BC_ind_T) # ([arrray row index], [array colum index])
    BC_index_np = np.asarray(BC_index_tupla).T



else:
    df = pd.read_csv(csv_name+'.csv')
    BC_index = df['points'][0]
    BC_index = literal_eval(BC_index)


#interior points
p = path.Path(BC_index)
mesh_inside_index = []

for i in range(Nx):
    for j in range(Ny):
        if p.contains_point([i,j], radius=1e-5) and [i,j] not in BC_index:
            mesh_inside_index += [[i,j]]
            
mesh_inside_index_np = np.array(mesh_inside_index)
mesh_inside_index_tupla = tuple(mesh_inside_index_np.T)



#%% creation of the mesh, boundary conditions and initial conditions multi D

meshDrug = np.zeros((Nx,Ny))

meshDistance = np.zeros((Nx,Ny)) #represent the distance of a point from the centre
meshDistanceSecondPoint = np.zeros((Nx,Ny)) #represent the distance of a point from the second infection point
meshDistance_fromBoundary2 = np.zeros((2*Nx-1,2*Ny-1)) #represent the distance of a point from the boundary 
meshDistance_fromBoundary = np.zeros((Nx,Ny))


## %%distance matrix

Ds_matrix = np.zeros((2*Nx-1, 2*Ny-1))
Ds_matrix.fill(Ds[0])
x = np.linspace(0, Nx-1, Nx, dtype=int)
y = np.linspace(0, Ny-1, Ny, dtype=int)
x, y = np.meshgrid(x, y)

dr = R/n_don

polygon = shapely.Polygon(BC_index)
for el in mesh_inside_index:
    el0 = el[0]
    el1 = el[1]
    point = shapely.Point(el0, el1)
    dist = round(polygon.exterior.distance(point),3)
    meshDistance_fromBoundary[el0,el1] = dist #distance in grid indices


# --- Thicken the boundary ---
spessore_bordo_celle = 2  # boundary thickness in cells

nuovo_BC_index = list(BC_index) # start from the original boundary
nuovo_mesh_inside_index = []

for el in mesh_inside_index:
    el0 = el[0]
    el1 = el[1]
    distanza = meshDistance_fromBoundary[el0, el1]
    
    # points near the original boundary become boundary cells
    if distanza <= spessore_bordo_celle:
        nuovo_BC_index.append([el0, el1])
    else:
        # otherwise the point stays interior
        nuovo_mesh_inside_index.append([el0, el1])

# update the original variables
BC_index = nuovo_BC_index
mesh_inside_index = nuovo_mesh_inside_index

# rebuild NumPy and tuple forms because they are used 
# by Numba functions below (e.g. simulation2D_multiple_D_numba)
BC_index_np = np.asarray(BC_index)
BC_index_tupla = tuple(BC_index_np.T)

mesh_inside_index_np = np.array(mesh_inside_index)
mesh_inside_index_tupla = tuple(mesh_inside_index_np.T)

#interior points
BC_index2 = 2*np.asarray(BC_index)
p = path.Path(BC_index2)
mesh_inside_index2 = []

for i in range(2*Nx):
    for j in range(2*Ny):
        if p.contains_point([i,j], radius=1e-5) and [i,j] not in BC_index2.tolist():
            mesh_inside_index2 += [[i,j]]

polygon = shapely.Polygon(BC_index2)
for el in mesh_inside_index2:
    el0 = el[0]
    el1 = el[1]
    point = shapely.Point(el0, el1)
    dist = round(polygon.exterior.distance(point)/2,3)
    meshDistance_fromBoundary2[el0,el1] = dist #distance in grid indices

max_dist = np.max(meshDistance_fromBoundary)
centre = np.asarray(np.where(meshDistance_fromBoundary == max_dist)).T[0]



#%%creation Ds_matrix

r_circles = np.linspace(0, round((min(Nx,Ny)-1)/2) - 1, round((min(Nx,Ny)-1)))

for k in range(len(r_circles)):
    D = Ds[k]
    r = r_circles[k]
    index_distance = np.asarray(np.where(meshDistance_fromBoundary2 >= r))
    if np.shape(index_distance)[1]!=0:
        Ds_matrix[index_distance[0], index_distance[1]] = D


#%%mesh_inside_index_don

dr = max_dist/n_don
r_circles = np.linspace(0, max_dist-dr, n_don)
r_circles[1] = round(out_don_thickness/dx,3)
if len(r_circles)>2:
    if r_circles[1] >= r_circles[2]:
        print("WARNING: check r_circles and n_don")

mesh_inside_index_don = []
mesh_ausiliaria = np.ones_like(meshDrug)*100
for k in range(len(r_circles)):
    r = r_circles[k]
    ind_don = np.array(np.where(meshDistance_fromBoundary > r+1e-5))
    mesh_ausiliaria[ind_don[0], ind_don[1]] = k
    
for k in range(len(r_circles)):
    ind_don = np.array(np.where(mesh_ausiliaria == k)).T.tolist()
    mesh_inside_index_don = [ind_don] + mesh_inside_index_don


#likely unnecessary
last_don = []
for el in mesh_inside_index_don[-1]:
    if el not in BC_index:
        last_don += [el]
mesh_inside_index_don[-1] = last_don

mesh_inside_index_don_tupla = [tuple(np.array(d).T) for d in mesh_inside_index_don]



#%% variance on PK Drug parameters ka and ke
seed=111111
# for a lognormal(mu, sigma), if we want for example 20% of variance coefficient
# variance sigma^2 = ln(var/(m^2) + 1) 
# mean mu = ln(m) - sigma^2/2
# sqrt(var) = 0.2*m
# => var = 0.04*m^2
# => sigma^2 = ln(1 + 0.04)
# => sigma = sqrt(ln(1 + 0.04))

if doVariabilityPK:
    n_sim = int(os.environ.get("TB_N_SIM", "100"))  # PK variability replicates
    plot_trasparence = 0.1 #for the trasparence in the plotFit
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



#%% resolution 2D multi D

#resolution
if doVariabilityPK:
    named_tuple = timing.localtime() # get struct_time
    time_string = timing.strftime("%Y_%m_%d_%H%M%S", named_tuple)
    folder = str(OUTPUT_DIR / (time_string+f'_{drug_name}_Drug_simulation_manual_domain_Tend_{T}_variability'))
    os.makedirs(folder, exist_ok=True)
    if name_csv_file != None :
        plt.ioff()
        colors = ['green', 'red', 'blue', 'aqua', 'aqua', 'magenta', (0,0,0,0)] #the last color is transparent
        cmap = mpcolors.LinearSegmentedColormap.from_list("MyCmap", colors, len(colors))
        mesh_plot_don = np.ones_like(meshDrug)*100     #from inner to outer   
        for i in range(n_don):
            don = np.asarray(mesh_inside_index_don[i]).T
            mesh_plot_don[don[0], don[1]] = i
        
        fig_don, axs = plt.subplots(1,2,figsize= (round(10*(Nx/np.max((Nx,Ny))),3)+1,round(10*(Ny/np.max((Nx,Ny))),3)/2 ), layout='constrained')
        ax1 = axs[0]
        ax2 = axs[1]
        
        ax1.plot(BC_ind_T[0], BC_ind_T[1], 'o', color = 'blue')
        
        c2 = ax2.pcolor(mesh_plot_don.T,shading="nearest", cmap=cmap, vmin=0, vmax=n_don)
        ax1.set_title('BC points')
        ax2.set_title('Donuts')
        plt.savefig(folder+'/geometria_dominio.png',bbox_inches='tight')
    
    
    folder_base = folder
    if saveCSV:
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
    
    tic = timing.time()
    
    if doPlotFit:
        plt.ion() #plot interattivo
    else:
        plt.ioff()
    
    if doPlotFit or savePlotFit:
        line_handles = [] #list of plotted lines
        fig2 = plt.figure(figsize= (12,8), layout='constrained')
        ax = fig2.add_subplot(111)
        ax.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
        colors = ['g', 'r', 'b', 'c', 'm', 'y']
        for i in range(1, n_don + 1): #initialize the plot
            line, = ax.plot(times, np.repeat(i, n_times), 
                            colors[(i - 1) % len(colors)] + '-', 
                            linewidth=2.5,
                            label=f'Donut {i}')
            line_handles.append(line)
        ax.legend(loc = 'upper left', fontsize = 18)
        ax.axhline(y=MBC90, color="red", alpha = 0.4, linewidth = 2)
        plt.annotate("MBC90", xy=(0.9*T, MBC90 + 50), color="r", size=22)
        plt.xlabel('Time (t) [day]', fontsize=18)
        plt.ylabel('Concentration [ng/g]', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(f"{drug_name} simulation", fontsize = 22, pad = 20, wrap = True)
        plt.tight_layout()
        
    lastPointsAllSimulation = []
    eradicationTimes = []
    n_eradication = 0
    for iteration in tqdm(range(n_sim)):
        print(f'iteration n:{iteration}')
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
            csv_name = f"{time_string} simulation manual domain"
        
            with open(folderCSV+'/'+csv_name+'.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)      
        
        meshDrug = np.zeros((Nx,Ny))
            
        drug_sim_bool = True
        result = []
        result_centre = []
        tIni = 0
        # for tEnd in times:
        for tEnd in tqdm(times):
            csv_row = [time_string, tEnd]
            if drug_sim_bool:
                if tIni > t_start_dose + t_stop_dose: #dt3
                    paramsDrug = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshDrug, BC_index_tupla, dx, dt3, BC_conc_i, elim_rate]
                else: #dt2
                    paramsDrug = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshDrug, BC_index_tupla, dx, dt2, BC_conc_i, elim_rate]
            
            if drug_sim_bool:
                if tIni >= t_start_dose:
                    conc_meanDrug, meshDrug, mesh_meanDrug = simulation2D_multiple_D_numba(Ds_matrix, paramsDrug)
                if tIni >= t_start_dose + t_stop_dose + t_dissipation_drug:
                    drug_sim_bool = False
            
            result += [conc_meanDrug] #from don1(inner) to outer [time x n_don]
            result_centre += [meshDrug[centre[0], centre[1]]]
            
            csv_row = np.concatenate((csv_row, conc_meanDrug))
            csv_row = np.concatenate((csv_row, [meshDrug[centre[0], centre[1]]]))
            csv_row = [eval(i) for i in csv_row] #now we have int and not string
            csv_row = np.asarray(np.round(csv_row, 2),dtype=float)
         
            tIni = tEnd
        
            if saveCSV:
                with open(folderCSV+'/'+csv_name+'.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows([csv_row])        
        
        
        result = np.asarray(result).T #[n_don x time]
        result_centre = np.asarray(result_centre) #[time]
        lastPointsAllSimulation += [result[-1,index_tstop]] #[n sim]
        t_plot = times
        
        if doPlotFit:
            for i in range(n_don):
                y_new = result[i]  # or the appropriate source
                ax.set_ylim(0,1.1*np.max(y_new))
                line_handles[i].set_ydata(y_new)


            refresh_figure(fig2)
            
        elif savePlotFit:
            for i in range(n_don):
                y_new = result[i]  # or the appropriate source
                ax.set_ylim(0,1.1*np.max(y_new))
                line_handles[i].set_ydata(y_new)
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
            else:
                print("CasMBC90 was not reached")
                MBC90_start_time_above += ['CasMBC90 not reached']
                MBC90_end_time_above += ['CasMBC90 not reached']
                MBC90_time_above += ['CasMBC90 not reached']
            
        if saveCSV:
            csv_row = [eval(time_string), iteration+1, MBC90_start_time_above[-1]] + MBC90_start_time_above + MBC90_time_above + MBC90_end_time_above
            csv_row = [round(x, 2) if isinstance(x, (int, float)) else x for x in csv_row]
            csv_row = np.asarray(csv_row, dtype=object)
            with open(folderCSV_base+'/'+csv_name_base+'.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([csv_row])
        
        with open(folder+f'/{time_string}_Drug__params.txt', 'w', encoding='UTF8', newline='') as f:
            f.write(f'{time_string} Drug PK parameters: \n')
            f.write(f'{drug_name} \n')
            f.write(f'F = {F_i}; ka1 = {ka1_i}; ke = {ke_i}; ka2 = {ka2_i}; ka3 = {ka3_i}\n\n')
            if MBC90 != None:
                f.write(f'Time to arrive MBC90 in each donut from inner to outer and the last value in the centre\n')
                f.write(f'{MBC90_start_time_above}\n')
                f.write(f'Time above MBC90 (in %) during the tratment in each donut from inner to outer and the last value in the centre\n')
                f.write(f'{MBC90_time_above}\n')
                f.write(f'Time above MBC90 (in day) after tratment end in each donut from inner to outer and the last value in the centre\n')
                f.write(f'{MBC90_end_time_above}\n')
                f.write(f'Eradication time (same of time to arrive MBC90 in centre)\n')
                f.write(f'{MBC90_start_time_above[-1]}\n')
    
    # stop tracking computational cost
    toc = timing.time()     
    # compute computational cost
    wallClockTime = toc - tic
    print(wallClockTime)
    
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
        
        fig30 = plt.figure(figsize= (12,8), layout='constrained')
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
        
        fig30.tight_layout()
        plt.show()
        if savePlotDistributionVaribility:
            fig30.savefig(folder_base+'/distribution_eradicationTime.png',bbox_inches='tight')


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
    print('no variability')
    
    named_tuple = timing.localtime() # get struct_time
    time_string = timing.strftime("%Y_%m_%d_%H%M%S", named_tuple)
    folder = str(OUTPUT_DIR / (time_string+f'_{drug_name}_Drug_simulation_manual_domain_Tend_{T}'))
    os.makedirs(folder, exist_ok=True)
    if name_csv_file != None :
        plt.ioff()
        colors = ['green', 'red', 'blue', 'aqua', 'aqua', 'magenta', (0,0,0,0)] #the last color is transparent
        cmap = mpcolors.LinearSegmentedColormap.from_list("MyCmap", colors, len(colors))
        mesh_plot_don = np.ones_like(meshDrug)*100     #from inner to outer   
        for i in range(n_don):
            don = np.asarray(mesh_inside_index_don[i]).T
            mesh_plot_don[don[0], don[1]] = i
        
        fig_don, axs = plt.subplots(1,2,figsize= (round(10*(Nx/np.max((Nx,Ny))),3)+1,round(10*(Ny/np.max((Nx,Ny))),3)/2 ), layout='constrained')
        ax1 = axs[0]
        ax2 = axs[1]
        
        ax1.plot(BC_ind_T[0], BC_ind_T[1], 'o', color = 'blue')
        
        c2 = ax2.pcolor(mesh_plot_don.T,shading="nearest", cmap=cmap, vmin=0, vmax=n_don)
        ax1.set_title('BC points')
        ax2.set_title('Donuts')
        plt.savefig(folder+'/geometria_dominio.png',bbox_inches='tight')
    
    
    for iteration in range(1):
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
            csv_name = f"{time_string} simulation with manual domain"
            csv_path = os.path.join(folderCSV, csv_name + '.csv')
            
            with open(csv_path, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
           
            
        meshDrug = np.zeros((Nx,Ny))
        y_val_plot, x_val_plot = np.meshgrid(y_val, x_val)
        if doPlot:
            plt.ion()
            fig, axs = plt.subplots(1,2,figsize= (round(10*(Nx/np.max((Nx,Ny))),3)+1,round(10*(Ny/np.max((Nx,Ny))),3)/2 ), layout='constrained')
            ax1 = axs[0]
            ax2 = axs[1]
            
            c1 = ax1.pcolor(x_val_plot, y_val_plot, meshDrug,shading="nearest", cmap="Greens", vmin=0, vmax=C0drug)
            cbar1 = fig.colorbar(c1, ax=[ax2],location='right', ticks=[0, round(C0drug/6), round(C0drug/3), round(C0drug/2), round(2*C0drug/3), round(5*C0drug/6), C0drug])
            cbar1.ax.set_yticklabels([0, round(C0drug/6), round(C0drug/3), round(C0drug/2), round(2*C0drug/3), round(5*C0drug/6), f'> {C0drug}'])
            cbar1.ax.set_title('ng/g')
        
        time = 0.
        
        i=0
        drug_sim_bool = True
        tIni = 0.
        tic = timing.time()

        if saveCSV: #close this file after the loop
            f_csv = open(csv_path, 'a', encoding='UTF8', newline='')
            writer = csv.writer(f_csv)
            
        for tEnd in tqdm(times): #tqdm to see the progression on console
            csv_row = [time_string, tEnd]

            if drug_sim_bool:
                if tIni > t_start_dose + t_stop_dose: #dt3
                    paramsDrug = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshDrug, BC_index_tupla, dx, dt3, BC_conc, elim_rate] 
                else: #dt2
                    paramsDrug = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshDrug, BC_index_tupla, dx, dt2, BC_conc, elim_rate]

            if drug_sim_bool:
                if tIni >= t_start_dose:
                    conc_meanDrug, meshDrug, mesh_meanDrug = simulation2D_multiple_D_numba(Ds_matrix, paramsDrug)
                if tIni >= t_start_dose + t_stop_dose + t_dissipation_drug:
                    drug_sim_bool = False
                    
            if saveCSV:
                csv_row = np.concatenate((csv_row, conc_meanDrug))
                csv_row = np.concatenate((csv_row, [meshDrug[centre[0], centre[1]]]))
                csv_row = [eval(i) for i in csv_row] #now we have int and not string
                csv_row = np.asarray(np.round(csv_row, 2), dtype=float)
                writer.writerow(csv_row)

            tIni = tEnd
        
            if doPlot:
                ax1 = axs[0]
                ax2 = axs[1]
                if drug_sim_bool:
                    ax1.cla()
                ax2.cla()
                
                if drug_sim_bool:
                    c1 = ax1.pcolor(x_val_plot, y_val_plot, meshDrug,shading="nearest", cmap="Greens", vmin=0, vmax=C0drug)
                    ax1.set_title('Drug', fontsize=18)

                ax2.pcolor(x_val_plot, y_val_plot, mesh_meanDrug,shading="nearest", cmap="Greens", vmin=0, vmax=C0)
                ax2.set_title('Mean drug in donuts')
                plt.suptitle(f"time: {tEnd}", wrap = True)
                
                if plt.get_fignums():  # figures are open
                    fig = plt.gcf()
                    fig.canvas.draw_idle()  # aggiorna senza forzare apertura
                else:
                    plt.pause(0.1)
                    plt.show()
                # plt.pause(0.1)
                if savePlot:
                    figName = f"{time_string}_Drug_simulation_manual_domain_2D_time_{10000+tEnd}"
                    figName = figName.translate(str.maketrans('', '', '.'))
                    plt.savefig(folderPNG+'/'+figName+'.png',bbox_inches='tight')
        
        if saveCSV:
            f_csv.close()
        
        
        df = pd.read_csv(folderCSV+'/'+csv_name+'.csv')
        t_plot = df["time"]
        if doPlotFit:
            plt.ion() #plot interattivo
        else:
            plt.ioff()
            
        if doPlotFit:
            plt.ion()
            fig2 = plt.figure(figsize= (12,8), layout='constrained')
            ax = fig2.add_subplot(111)
            ax.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
            colors = ['g', 'r', 'b', 'c', 'm', 'y']
            for i in range(1,n_don+1):
                ax.plot(t_plot, df["sim C in don"+str(i)], colors[i-1]+'-',linewidth = 2.5, label = f'Donut {i}')
            ax.legend(loc = 'upper right', fontsize = 18)
            ax.axhline(y=MBC90, color="red", alpha = 0.4, linewidth = 2)
            plt.annotate("MBC90", xy=(0.9*T, MBC90 + 50), color="r", size=22)
            plt.xlabel('Time (t) [day]', fontsize=18)
            plt.ylabel('Concentration [ng/g]', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.title(f"{drug_name} simulation", fontsize = 22, pad = 20, wrap = True)
            plt.tight_layout()
            figName = f"{time_string}simulation_with_D_{DDrug_title}"
            if savePlotFit:
                plt.savefig(folder+'/'+figName+'.png',bbox_inches='tight')
            plt.show()

        elif savePlotFit:
            fig2 = plt.figure(figsize= (12,8), layout='constrained')
            ax = fig2.add_subplot(111)
            ax.grid(axis='y', color='k', alpha = 0.1, linewidth = 2)
            colors = ['g', 'r', 'b', 'c', 'm', 'y']
            for i in range(1, n_don + 1): #initialize the plot
                y_plot = df[f"sim C in don{i}"]
                ax.plot(times, y_plot, colors[(i - 1) % len(colors)] + '-', linewidth=2.5, label=f'Donut {i}')
            ax.legend(loc = 'upper right', fontsize = 18)
            ax.axhline(y=MBC90, color="red", alpha = 0.4, linewidth = 2)
            plt.annotate("MBC90", xy=(0.9*T, MBC90 + 50), color="r", size=22)
            plt.xlabel('Time (t) [day]', fontsize=18)
            plt.ylabel('Concentration [ng/g]', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.title(f"{drug_name} simulation", fontsize = 22, pad = 20, wrap = True)
            plt.tight_layout()
            figName = f"{time_string}simulation_with_D_{DDrug_title}"
            fig_path = os.path.join(folder, figName + '.png')
            os.makedirs(folder, exist_ok=True)
            plt.savefig(fig_path,bbox_inches='tight')
            
        MBC90_time_above = [] #time above MBC90 in % in tratement period in each donut from inner to outer and last value is the centre
        MBC90_start_time_above = []
        MBC90_end_time_above = [] #time above after treatment end
        for i in range(1,n_don+1):
            conditionMBC = df["sim C in don"+str(i)] >= MBC90 
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
        
        conditionMBC = df["centre"] >= MBC90 
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
            
        # stop tracking computational cost
        toc = timing.time()     
        # compute computational cost
        wallClockTime = toc - tic
        print(wallClockTime)
        
    with open(folder+f'/{time_string}_Drug__params.txt', 'w', encoding='UTF8', newline='') as f:
        f.write(f'{time_string} Drug simulation parameters: \n')
        f.write(f'T end:{T}')
        f.write(f'Drug parameters: \n')
        f.write(f'{drug_name} \n')
        f.write(f'time step dt:{dt2} day; \n')
        f.write(f'time start {t_start_dose} day, time stop {t_stop_dose} day, dissipation period {t_dissipation_drug} day\n')
        f.write(f'size:{Len_x} x {Len_y} um; \n')
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
        if MBC90 != None:
            f.write(f'Time to arrive MBC90 in each donut from inner to outer and the last value in the centre\n')
            f.write(f'{str(MBC90_start_time_above)}\n')
            f.write(f'Time above MBC90 (in %) during the tratment in each donut from inner to outer and the last value in the centre\n')
            f.write(f'{str(MBC90_time_above)}\n')
            f.write(f'Time above MBC90 (in day) after tratment end in each donut from inner to outer and the last value in the centre\n')
            f.write(f'{str(MBC90_end_time_above)}\n')
            f.write(f'Eradication time (same of time to arrive MBC90 in centre)\n')
            f.write(f'{MBC90_start_time_above[-1]}\n')
        f.write(f'Execution time:{round(wallClockTime/60,1)} min')
    


