#%% import
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:32:55 2023

@author: ismaele
"""
#import
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import time as timing
from functools import reduce
from matplotlib import path
import scipy.optimize as opt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
import csv
import os
import sys
from numba import njit

#%% functions


def refresh_figure(fig):
    """Update a live plot when supported; also work with non-GUI backends."""
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


    
def cg(Ax, b, x, tol=1e-6, kmax=10**4, Ax_param=None):
    """
    CG as a matrix-free method for the resolution of Ax=b.

    Parameters
    ----------
    Ax : function that return the matrix product of Ax.
    b : vector (or matrix that will be transform in a vector).
    x : start value.
    tol : stop value
        The default is 1e-6.
    kmax : maximum iterations
        The default is 10**4.
    Ax_param = parameters needed for the function Ax
    
    
    Returns
    -------
    the number of iterations due to convergence and x became the solution

    """
    # print('start')
    r = b - Ax(x,Ax_param)
    p = r.copy()        
    rho = np.dot(r.flat, r.flat) 
    for k in range(kmax):
        if np.sqrt(rho) < tol: 
            # print(k)
            return k
        Ap = Ax(p, Ax_param)
        alpha = rho / np.dot(p.flat, Ap.flat)
        x += alpha * p
        r -= alpha * Ap
        rhonew = np.dot(r.flat, r.flat)
        p = r + rhonew / rho * p
        rho = rhonew
    return -1



def Ax_op2(x, param):
    """ Return (I + lambda A_h) x. """
    dx = param[0]
    dt = param[1]
    # print(dt)
    D = param[2]
    BC_index = param[3]
    mesh_inside_index = param[4]
    
    mii = np.asarray(mesh_inside_index).T
    mii0 = mii[0]
    mii1 = mii[1]
    BC = np.asarray(BC_index).T
    
    lamb = D*dt / dx**2
    n = len(x) - 1
    Ax = np.zeros_like(x)
    #fix BC
    Ax[BC[0],BC[1]] = x[BC[0],BC[1]]

    Ax[mii0, mii1] = x[mii0, mii1] + lamb * (4 * x[mii0, mii1] -\
                x[mii0-1, mii1] - x[mii0+1, mii1] -\
                x[mii0, mii1-1] - x[mii0, mii1+1])
    
    return Ax

def Ax_op_oxy(x, param):
    """ Return (I + lambda A_h) x. """
    dx = param[0]
    dt = param[1]
    D = param[2]
    BC_index = param[3] #tupla
    mii = param[4] # mesh_inside_index tupla
    mcell = param[5].copy() # meshCellular 2D array
    ke_tb = param[6]
    ke_t = param[7]
    ke_mr = param[8]
    ke_ma = param[9]
    ke_mi = param[10]
    ke_mci = param[11]

    r, c = mii #row and column array

    # mii = np.asarray(mesh_inside_index).T
    # BCcell = np.asarray(BC_index).T
    #remove BC_index from meshCellular setting it to 0 (empty)
    # mcell[BCcell[0], BCcell[1]] = 0
    mcell[BC_index] = 0
    
    # Array of elimination rate where index = value of the cell (0-7)
    #0 empty, 1 TB slow, 2 Tb fast, 3 T, 4 Mr, 5 Ma, 6 Mi, 7 Mci, 8 caseum
    rates = np.array([0, ke_tb, ke_tb, ke_t, ke_mr, ke_ma, ke_mi, ke_mci, 0])
    
    # TB_slow = np.asarray(np.where(mcell == 1))
    # TB_fast = np.asarray(np.where(mcell == 2))
    # Tcell = np.asarray(np.where(mcell == 3))
    # Mr = np.asarray(np.where(mcell == 4))
    # Ma = np.asarray(np.where(mcell == 5))
    # Mi = np.asarray(np.where(mcell == 6))
    # Mci = np.asarray(np.where(mcell == 7))
    
    lamb = D*dt / dx**2
    # n = len(x) - 1
    Ax = np.zeros_like(x)
    #fix BC
    Ax[BC_index] = x[BC_index]
    
    # #operate only on interior cells
    # Ax[mii[0], mii[1]] = x[mii[0], mii[1]] + lamb * (4 * x[mii[0], mii[1]] -\
    #             x[mii[0]-1, mii[1]] - x[mii[0]+1, mii[1]] -\
    #             x[mii[0], mii[1]-1] - x[mii[0], mii[1]+1])
    
    Ax[r, c] = x[r, c] + lamb * (4 * x[r, c] -\
             x[r-1, c] - x[r+1, c] - x[r, c-1] - x[r, c+1])
    
    # apply reaction only away from the boundary (using mii)
    Ax[mii] += dt * rates[mcell[mii]] * x[mii]

    # equivalent to:
    # #aggiungo gli elim rate
    # Ax[TB_slow[0], TB_slow[1]] += dt*ke_tb * x[TB_slow[0], TB_slow[1]]
    # Ax[TB_fast[0], TB_fast[1]] += dt*ke_tb * x[TB_fast[0], TB_fast[1]]
    # Ax[Tcell[0], Tcell[1]] += dt*ke_t * x[Tcell[0], Tcell[1]]
    # Ax[Mr[0], Mr[1]] += dt*ke_mr * x[Mr[0], Mr[1]]
    # Ax[Ma[0], Ma[1]] += dt*ke_ma * x[Ma[0], Ma[1]]
    # Ax[Mi[0], Mi[1]] += dt*ke_mi * x[Mi[0], Mi[1]]
    # Ax[Mci[0], Mci[1]] += dt*ke_mci * x[Mci[0], Mci[1]]
    
    return Ax



def Ax_op_2D_multi_D_simplified(x, param):
    """ Return (I + lambda A_h) x. 
    diven domain
    """    
    
    dx = param[0]
    dt = param[1]
    # print(dt)
    D = param[2] # matrix with dimension 2*Nx-1, 2*Nx-1
    BC_index = param[3]
    mesh_inside_index = param[4] #from inner to outer (without the boundary)
    mesh_inside_index_don_2d = param[5] #from inner to outer (without the boundary)
    elim_rate = param[6]
    
    n = len(x) - 1
    Ax = np.zeros_like(x)
    BC = np.asarray(BC_index).T
    Ax[BC[0],BC[1]] = x[BC[0],BC[1]]
    lamb = D*dt / (dx**2) #matrix
    #operate only on interior cells
    mii = np.asarray(mesh_inside_index).T
    mii0 = mii[0]
    mii1 = mii[1]
    Ax[mii0, mii1] = x[mii0, mii1] +\
        (lamb[2*mii0 +1,2*mii1] + lamb[2*mii0-1, 2*mii1] +\
         lamb[2*mii0,2*mii1 +1] + lamb[2*mii0, 2*mii1-1]) * x[mii0, mii1] -\
        lamb[2*mii0 -1, 2*mii1] * x[mii0-1, mii1] -\
        lamb[2*mii0 +1, 2*mii1] * x[mii0+1, mii1] -\
        lamb[2*mii0, 2*mii1 -1] * x[mii0, mii1-1] -\
        lamb[2*mii0, 2*mii1 +1] * x[mii0, mii1+1]
    don = np.asarray(mesh_inside_index_don_2d[-1]).T
    Ax[don[0],don[1]] += dt*elim_rate*x[don[0], don[1]]
    return Ax

def Ax_op_2D_multi_D(x, param):
    # mimetic version of the function
    """ Return (I + dt * A_mimetic) x
        Conservativo, isotropo (MPFA-O)
    """

    dx = param[0]
    dt = param[1]
    D  = param[2]   # staggered (2N-1 x 2N-1)
    BC = param[3]   #BC_index tupla
    mii = param[4]  #mesh_inside_index tupla
    mii_don = param[5] #mesh_inside_index_don_2d tupla
    elim_rate = param[6]
        
    # Scompatta i parametri pre-ottimizzati
    # dx, dt, D, BC, mii, don, elim_rate = param_opt
    Ax = np.zeros_like(x)

    # -----------------------
    # Boundary conditions
    # -----------------------
    # BC = np.asarray(BC_index).T
    # Ax[BC[0], BC[1]] = x[BC[0], BC[1]]
    Ax[BC] = x[BC]
    # -----------------------
    # Precompute lambda
    # -----------------------
    lamb = D * dt / dx**2 #Matrix like D

    # -----------------------
    # Work only Inside nodes
    # -----------------------
    
    # mii = np.asarray(mesh_inside_index).T
    # i = mii[0]
    # j = mii[1]
    i, j = mii # mii must be a tuple (array_i, array_j)
    #precompute straggered avoiding 2*i+1 many times
    i2, j2 = 2*i, 2*j
    i2_p, i2_m = i2 + 1, i2 - 1
    j2_p, j2_m = j2 + 1, j2 - 1

    # -----------------------
    # Coefficients (trapezoidal integration rule)
    # -----------------------
    alpha = 0.5
    beta  = 0.25

    # -----------------------
    # D on faces
    # -----------------------
    D_E = alpha * lamb[i2_p, j2]
    D_W = alpha * lamb[i2_m, j2]
    D_N = alpha * lamb[i2, j2_p]
    D_S = alpha * lamb[i2, j2_m]
    # D_E  = alpha * lamb[2*i+1, 2*j]
    # D_W  = alpha * lamb[2*i-1, 2*j]
    # D_N  = alpha * lamb[2*i,   2*j+1]
    # D_S  = alpha * lamb[2*i,   2*j-1]

    # D on vertices
    D_NE = beta * lamb[i2_p, j2_p]
    D_SE = beta * lamb[i2_p, j2_m]
    D_NW = beta * lamb[i2_m, j2_p]
    D_SW = beta * lamb[i2_m, j2_m]
    # D_NE = beta * lamb[2*i+1, 2*j+1]
    # D_SE = beta * lamb[2*i+1, 2*j-1]
    # D_NW = beta * lamb[2*i-1, 2*j+1]
    # D_SW = beta * lamb[2*i-1, 2*j-1]
    
    # -----------------------
    # Mimetic flux
    # -----------------------
    # qE = (
    #     D_E  * (x[i+1, j]   - x[i, j]) +
    #     D_NE * (x[i+1, j+1] - x[i, j+1]) +
    #     D_SE * (x[i+1, j-1] - x[i, j-1])
    # )

    # qW = (
    #     D_W  * (x[i, j]   - x[i-1, j]) +
    #     D_NW * (x[i, j+1] - x[i-1, j+1]) +
    #     D_SW * (x[i, j-1] - x[i-1, j-1])
    # )

    # qN = (
    #     D_N  * (x[i, j+1]   - x[i, j]) +
    #     D_NE * (x[i+1, j+1] - x[i+1, j]) +
    #     D_NW * (x[i-1, j+1] - x[i-1, j])
    # )

    # qS = (
    #     D_S  * (x[i, j]   - x[i, j-1]) +
    #     D_SE * (x[i+1, j] - x[i+1, j-1]) +
    #     D_SW * (x[i-1, j] - x[i-1, j-1])
    # )
    
    # -----------------------
    # Operator BTCS
    # -----------------------
    # Ax[i, j] = x[i, j] - (qE - qW + qN - qS)
    
    #equivalently
    # Ax[i, j] = x[i, j] +\
    #     (D_E + D_W + D_N + D_S) * x[i, j] +\
    #     (D_NW + D_SW - D_W) * x[i-1, j] +\
    #     (D_NE + D_SE - D_E) * x[i+1, j] +\
    #     (D_SE + D_SW - D_S) * x[i, j-1] +\
    #     (D_NE + D_NW - D_N) * x[i, j+1] -\
    #     2* (D_NE * x[i+1, j+1] + D_SE * x[i+1, j-1] + D_SW * x[i-1, j-1] + D_NW * x[i-1, j+1])
    
    # flusso + ortogonale + vertici
    Ax[i, j] = x[i, j] * (1.0 + D_E + D_W + D_N + D_S + 1*(D_NE + D_NW + D_SE + D_SW)) -\
        (D_E * x[i+1, j] + D_W * x[i-1, j] + D_N * x[i, j+1] + D_S * x[i, j-1]) -\
        1*(D_NE * x[i+1, j+1] + D_NW * x[i-1, j+1] + D_SE * x[i+1, j-1] + D_SW * x[i-1, j-1]) #flusso vertici

    # -----------------------
    # Elimination term
    # -----------------------
    last_don = mii_don[-1]
    Ax[last_don] += dt * elim_rate * x[last_don]
    # don = np.asarray(mesh_inside_index_don_2d[-1]).T
    # Ax[don[0], don[1]] += dt * elim_rate * x[don[0], don[1]]

    return Ax



def save_matrix(A, file_name):
    with open(file_name, 'wb') as f:
        np.save(f, A)

def load_matrix(file_name):
    with open(file_name, 'rb') as f:
        A = np.load(f)
    return (A)



#%% ABM Numba
@njit
def cg_oxy_numba(old_mesh, mesh, dx, dt, D, BC_r, BC_c, mii_r, mii_c, mcell, rates, tol=1e-2, kmax=10000):
    # old_mesh is right-hand side b; mesh is initial guess x
    x = mesh.copy()
    b = old_mesh.copy()
    lamb = D * dt / (dx**2)
    
    # --- inner function computing Ax ---
    def compute_Ax(v):
        Ax = np.zeros_like(v)
        # 1. Condizioni al contorno (BC)
        for i in range(len(BC_r)):
            r, c = BC_r[i], BC_c[i]
            Ax[r, c] = v[r, c]
            
        # 2. interior points (diffusion and consumption)
        for i in range(len(mii_r)):
            r, c = mii_r[i], mii_c[i]
            # consumption rate based on cell type
            cell_type = mcell[r, c]
            rate = rates[cell_type]
            
            # Laplaciano e reazione
            Ax[r, c] = v[r, c] + lamb * (4 * v[r, c] - v[r-1, c] - v[r+1, c] - v[r, c-1] - v[r, c+1]) + dt * rate * v[r, c]
        return Ax

    # --- Inizio del Gradiente Coniugato ---
    r = b - compute_Ax(x)
    p = r.copy()
    rho = np.sum(r * r) 
    
    iters = -1
    for k in range(kmax):
        if np.sqrt(rho) < tol:
            iters = k
            break
            
        Ap = compute_Ax(p)
        alpha = rho / np.sum(p * Ap)
        
        x += alpha * p
        r -= alpha * Ap
        
        rhonew = np.sum(r * r)
        p = r + (rhonew / rho) * p
        rho = rhonew

    return iters, x

@njit
def cg_drug_numba(old_mesh, mesh, dx, dt, D_mat, BC_r, BC_c, mii_r, mii_c, don_r, don_c, elim_rate, tol=1e-3, kmax=100000):
    x = mesh.copy()
    b = old_mesh.copy()
    lamb = D_mat * dt / (dx**2)
    
    alpha = 0.5
    beta = 0.25

    def compute_Ax(v):
        Ax = np.zeros_like(v)
        
        # BC
        for idx in range(len(BC_r)):
            r, c = BC_r[idx], BC_c[idx]
            Ax[r, c] = v[r, c]

        # interior (MPFA-O)
        for idx in range(len(mii_r)):
            i, j = mii_r[idx], mii_c[idx]
            i2, j2 = 2*i, 2*j
            i2_p, i2_m = i2 + 1, i2 - 1
            j2_p, j2_m = j2 + 1, j2 - 1

            D_E = alpha * lamb[i2_p, j2]
            D_W = alpha * lamb[i2_m, j2]
            D_N = alpha * lamb[i2, j2_p]
            D_S = alpha * lamb[i2, j2_m]

            D_NE = beta * lamb[i2_p, j2_p]
            D_SE = beta * lamb[i2_p, j2_m]
            D_NW = beta * lamb[i2_m, j2_p]
            D_SW = beta * lamb[i2_m, j2_m]
            
            Ax[i, j] = v[i, j] * (1.0 + D_E + D_W + D_N + D_S + (D_NE + D_NW + D_SE + D_SW)) \
                     - (D_E * v[i+1, j] + D_W * v[i-1, j] + D_N * v[i, j+1] + D_S * v[i, j-1]) \
                     - (D_NE * v[i+1, j+1] + D_NW * v[i-1, j+1] + D_SE * v[i+1, j-1] + D_SW * v[i-1, j-1])

        # elimination in the outermost donut
        for idx in range(len(don_r)):
            r, c = don_r[idx], don_c[idx]
            Ax[r, c] += dt * elim_rate * v[r, c]

        return Ax

    # mimetic version of the function
    """ Return (I + dt * A_mimetic) x
        Conservativo, isotropo (MPFA-O)
    """

    # --- Inizio del Gradiente Coniugato ---
    r = b - compute_Ax(x)
    p = r.copy()
    rho = np.sum(r * r)
    
    iters = -1
    for k in range(kmax):
        if np.sqrt(rho) < tol:
            iters = k
            break
        Ap = compute_Ax(p)
        alpha_cg = rho / np.sum(p * Ap)
        x += alpha_cg * p
        r -= alpha_cg * Ap
        rhonew = np.sum(r * r)
        p = r + (rhonew / rho) * p
        rho = rhonew
        
    return iters, x











def simulation2D_oxy_numba(D, params):
    # paramsOx = [tIni, tEnd, mesh_inside_index_tupla, mesh_inside_index_don_tupla, meshOx, BC_index_tupla, dx, dt3, C0oxy, meshCellular, ke_tb, ke_t, ke_mr, ke_ma, ke_mi, ke_mci]
    '''
    
    Parameters
    ----------
    D : scalar
    params : 
        tIni,                   scalar
        tEnd,                   scalar
        mesh_inside_index,      tuple
        mesh_inside_index_don,  tuple
        initial_mesh,           2D np array
        BC_index,               tuple
        dx,                     saclar
        dt,                     scalar
        C0                      scalar
    mesh_inside_index_don_1d is from inner to outer region (it contains also the boundary)

    Returns
    -------
    None.

    '''
    tIni, tEnd, mesh_inside_index, mesh_inside_index_don, mesh, BC_index, dx, dt, C0,\
        meshCellular, ke_tb, ke_t, ke_mr, ke_ma, ke_mi, ke_mci = params
    
    time = tIni
    dt_copy = dt
    # mesh = initial_mesh
    rates = np.array([0, ke_tb, ke_tb, ke_t, ke_mr, ke_ma, ke_mi, ke_mci, 0], dtype=np.float64)
    
    while time<tEnd - 1e-10:
        # match exit time
        if time+dt_copy>tEnd:
            dt_copy = tEnd-time     
        # update time
        time += dt_copy 
        
        if math.isnan(time):
            print("TIME IS NAN")
            sys.exit("Exiting program")   
        #update interior cells
        old_mesh = mesh.copy()
        
        # Prepare arrays with types Numba can compile.
        BC_r, BC_c = BC_index[0], BC_index[1]
        mii_r, mii_c = mesh_inside_index[0], mesh_inside_index[1]

        
        # 2. call the JIT-compiled solver
        k, mesh = cg_oxy_numba(old_mesh, mesh, dx, dt_copy, D, 
                               BC_r, BC_c, mii_r, mii_c, meshCellular, rates, tol=1e-2)
                               
        assert k >= 0, "CG did not converge"

    mesh_mean = mesh.copy()
    conc_mean = []

    for don in mesh_inside_index_don:
        mean = np.mean(mesh[don])
        conc_mean.append(mean)
        mesh_mean[don] = mean
    conc_mean = np.asarray(conc_mean)
    
    return(conc_mean, mesh, mesh_mean)

def simulation2D_multiple_D_numba(Ds, params):
    '''
    
    Parameters
    ----------
    Ds : matrix of dimension (2Nx-1)x(2Nx-1) with the value of D_{i+1/2,j} in position (2i+1,2j) and D_{i,j+1/2} in (2i,2j+1)
    params : 
        tIni,                   scalar
        tEnd,                   scalar
        mesh_inside_index,      tuple
        mesh_inside_index_don,  tuple
        initial_mesh,           2D np array
        BC_index,               tuple
        dx,                     saclar
        dt,                     scalar
        D_out                   scalar
        elim_rate,              scalar
        t_stop_dose             scalar
        mesh_inside_index_don_1d is from inner to outer region (it NOT contains boundary)
    mesh_inside_index (it NOT contains boundary)
    
    ka and ke: absorption and elimination rate (clearance) for drug BC 
    
    NB. for the moment only for BTCS
    
    Returns
    -------
    conc_mean: array with the mean in each donut
    mesh: updated mesh
    mesh_mean: mesh with the mean value in each donut (useful for plot)

    '''

    tIni, tEnd, mesh_inside_index, mesh_inside_index_don_2d, initial_mesh, BC_index, dx, dt, BC_conc, elim_rate = params
    len_BC_ind = len(BC_index[0])
    time = tIni
    dt_copy = dt
    mesh = initial_mesh.copy()
    
    mii_r, mii_c = mesh_inside_index[0], mesh_inside_index[1]
    
    while time<tEnd - 1e-10:
        #fix BC
        BC = BC_conc(time)
        
        # Extract coordinates for Numba.
        BC_r, BC_c = BC_index[0], BC_index[1]
        
        mesh[BC_index] = np.repeat(BC,len_BC_ind)
        # match exit time
        if time+dt_copy>tEnd:
            dt_copy = tEnd-time     
        # update time
        time += dt_copy 
        
        if math.isnan(time):
            print("TIME IS NAN")
            sys.exit("Exiting program")   
        #update interior cells
        old_mesh = mesh.copy()
        
        last_don_r, last_don_c = mesh_inside_index_don_2d[-1][0], mesh_inside_index_don_2d[-1][1]

        k, mesh = cg_drug_numba(old_mesh, mesh, dx, dt_copy, Ds, 
                                BC_r, BC_c, mii_r, mii_c, last_don_r, last_don_c, 
                                elim_rate, tol=1e-5)

        assert k >= 0, "CG did not converge"
        # update exact solution
        
    mesh_mean = mesh.copy()
    conc_mean = []    
    for don in mesh_inside_index_don_2d:
        mean = np.mean(mesh[don])
        conc_mean.append(mean)
        mesh_mean[don] = mean
    conc_mean = np.asarray(conc_mean)
    
    return(conc_mean, mesh, mesh_mean)







@njit
def numba_weighted_choice(choices, probs, count):
    """Numba-compatible weighted choice among the first count entries."""
    if count == 0:
        return -1
    if count == 1:
        return choices[0]
    
    # sum probabilities for normalization
    total = 0.0
    for i in range(count):
        total += probs[i]
        
    r = np.random.rand() * total
    cum = 0.0
    for i in range(count):
        cum += probs[i]
        if r <= cum:
            return choices[i]
            
    return choices[count - 1] # fallback di sicurezza




@njit
def cellular_movements_numba(tIni, tEnd, meshDrug, meshOx, BC_index,Nx, dx, dt,\
    meshCellular, meshMovTime, meshLifeTime, meshInfected, meshReplication,\
    meshProbN,meshProbR,meshProbS,meshProbL,\
    movT, movMr, movMa, movMi, movCaseum, lifeT, lifeMr, lifeMa, lifeMi,\
    mov_vec, life_vec, Nici, Ncib, NstartRec,\
    Drugkill_f, Drugkill_s, Drugkill_mi, Oslow_to_fast, Ofast_to_slow,\
    P_kill_T_Mi, P_activ_MrMa, Pnew_Mr, Pnew_T, P_Mi_move_Tb, P_Ma_deat_caseum, P_Ma_healt_caseum, P_MrMi, P_Mr_kill_tb, P_activ_MiMa,\
    replication_fast, replication_slow,\
        reduction_factor,reductionT, reductionMr,life_factorT,\
    seed):
    '''
    
    Parameters
    ----------
    params : 
        tIni        scalar
        tEnd        scalar
        initial_mesh
        BC_index
        dx          scalar
        dt          scalar
        meshCellular
        meshMovTime
        meshLifeTime
        meshInfected
        meshReplication
        meshProbN, meshProbR, meshProbS, meshProbL,
        movT, movMr, movMa, movMi, movCaseum, 
        lifeT, lifeMr, lifeMa, lifeMi,\
        mov_vec
        life_vec
        Nici        scalar
        Ncib        scalar
        NstartRec   scalar
        Drugkill_f, Drugkill_s, Drugkill_mi     scalar
        Oslow_to_fast, Ofast_to_slow            scalar
        P_kill_T_Mi, P_activ_MrMa, Pnew_Mr, Pnew_T, P_Mi_move_Tb, P_Ma_deat_caseum, P_Ma_healt_caseum, P_MrMi, P_Mr_kill_tb, P_activ_MiMa,
        rep_fast, rep_slow                      scalar
        reduction_factor                        scalar
        reductionT, reductionMr                 scalar
        life_factorT                            scalar
        seed        scalar
    mesh_inside_index_don_1d is from inner to outer region (it contains also the boundary)

    Returns
    -------
    Updated meshCellular

    '''
    
    np.random.seed(seed) # initialize Numba's random generator separately
    
    time = tIni
    dt_copy = dt
    meshD = meshDrug # avoid unnecessary copies to save memory
    meshO = meshOx
    Nx, Ny = meshCellular.shape

    # explicit NumPy arrays for Numba instead of Python lists
    Direction_x = np.array([0, 1, 0, -1], dtype=np.int32)
    Direction_y = np.array([1, 0, -1, 0], dtype=np.int32)

    while time < tEnd - 1e-10:
        if time + dt_copy > tEnd:
            dt_copy = tEnd - time     
        time += dt_copy 
        
        if np.isnan(time):
            raise RuntimeError("TIME IS NAN") # sys.exit is unsupported in Numba
        
        # --- RIMOZIONE CELLE MORTE ---
        # Numba optimizes boolean masks
        dead_mask = meshLifeTime < time
        dead_ma_mask = dead_mask & (meshCellular == 5)
        
        dead_cells_Ma = np.argwhere(dead_ma_mask)
        
        dead_indices = np.argwhere(dead_mask)
        for i in range(len(dead_indices)):
            r = dead_indices[i, 0]
            c = dead_indices[i, 1]
            meshCellular[r, c] = 0
            meshMovTime[r, c] = 1e10 
            meshLifeTime[r, c] = 1e10
            meshInfected[r, c] = 0
            meshReplication[r, c] = 1e10
        

       # Ma dead -> caseum
        n_ma_dead = len(dead_cells_Ma)
        if n_ma_dead > 0:
            to_caseum = np.random.binomial(1, P_Ma_deat_caseum, size=n_ma_dead)
            for idx in range(n_ma_dead):
                if to_caseum[idx]:
                    r, c = dead_cells_Ma[idx, 0], dead_cells_Ma[idx, 1]
                    meshCellular[r, c] = 8
                    meshMovTime[r, c] = np.random.exponential(movCaseum) + time
                    meshLifeTime[r, c] = 1e10
                    meshInfected[r, c] = 0
                    meshReplication[r, c] = 1e10


        # --- MOVIMENTI ---
        mov_mask = meshMovTime < time
        mov_mask_cas = mov_mask & (meshCellular == 8)
        mov_mask_not_cas = mov_mask & (meshCellular != 8)
        
        possible_mov_cas = np.argwhere(mov_mask_cas)
        possible_mov_not_cas = np.argwhere(mov_mask_not_cas)        

        # use basic boolean logic instead of isin in Numba
        has_tb = ((meshCellular == 1) | (meshCellular == 2)).any()
        
        if has_tb: 
            if len(possible_mov_not_cas) > 0:
                np.random.shuffle(possible_mov_not_cas) # Shuffle in place
                
                for idx in range(len(possible_mov_not_cas)):
                    el0, el1 = possible_mov_not_cas[idx, 0], possible_mov_not_cas[idx, 1]
                    cellula = meshCellular[el0, el1]
                    
                    p_values = np.array([meshProbN[el0, el1], meshProbR[el0, el1], meshProbS[el0, el1], meshProbL[el0, el1]])

                    # --- Numba-compatible implementation ---
                    # preallocate space for up to four empty directions
                    dir_empty = np.empty(4, dtype=np.int32)
                    prob_empty = np.empty(4, dtype=np.float64)
                    c_empty = 0 # Contatore elementi validi
                    
                    dir_tb = np.empty(4, dtype=np.int32)
                    prob_tb = np.empty(4, dtype=np.float64)
                    c_tb = 0


                    if cellula == 5: # Ma
                        life_cellula = meshLifeTime[el0, el1]
                        
                        
                        for d in range(4):
                            newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                            
                            if 0 <= newx < Nx and 0 <= newy < Ny:
                                val_vicino = meshCellular[newx, newy]
                                if val_vicino == 0:
                                    dir_empty[c_empty] = d
                                    prob_empty[c_empty] = p_values[d]
                                    c_empty += 1
                                elif val_vicino == 1 or val_vicino == 2:
                                    dir_tb[c_tb] = d
                                    prob_tb[c_tb] = p_values[d]
                                    c_tb += 1
    
                        direzione = -1
                        
                        if c_empty > 0 or c_tb > 0:
                            if c_tb > 0 and c_empty > 0:
                                if np.random.binomial(1, P_Mi_move_Tb):
                                    direzione = numba_weighted_choice(dir_tb, prob_tb, c_tb)
                                else:
                                    direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
                            elif c_tb > 0:
                                direzione = numba_weighted_choice(dir_tb, prob_tb, c_tb)
                            else:
                                direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
        
                        if direzione != -1:
                            newx, newy = el0 + Direction_x[direzione], el1 + Direction_y[direzione]
                            
                            # check whether the chosen direction was empty
                            is_empty_dir = False
                            for i in range(c_empty):
                                if dir_empty[i] == direzione:
                                    is_empty_dir = True
                                    break
                            
                            if is_empty_dir:
                                meshCellular[newx, newy] = cellula
                                meshMovTime[newx, newy] = np.random.exponential(mov_vec[cellula]) + time
                                meshLifeTime[newx, newy] = life_cellula
                                meshInfected[newx, newy] = 0
                                meshReplication[newx, newy] = 1e10
                            else:
                                meshCellular[newx, newy] = cellula + 1
                                meshMovTime[newx, newy] = np.random.exponential(mov_vec[cellula+1]) + time
                                meshLifeTime[newx, newy] = np.random.exponential(life_vec[cellula+1]) + time
                                meshInfected[newx, newy] = 1
                                meshReplication[newx, newy] = 1e10
        
                            meshCellular[el0, el1] = 0
                            meshLifeTime[el0, el1] = 1e10
                            meshMovTime[el0, el1] = 1e10
                            meshInfected[el0, el1] = 0
                            meshReplication[el0, el1] = 1e10
                        else:
                            meshMovTime[el0, el1] = np.random.exponential(mov_vec[cellula]) + time
                    
                    elif cellula == 6:
                        life_cellula = meshLifeTime[el0, el1]

                        tb_mangiati = meshInfected[el0, el1]
                        if tb_mangiati == 0:
                            print('no Mtb at this location; check agent state')
                        
                        for d in range(4):
                            newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                            
                            if 0 <= newx < Nx and 0 <= newy < Ny:
                                val_vicino = meshCellular[newx, newy]
                                if val_vicino == 0:
                                    dir_empty[c_empty] = d
                                    prob_empty[c_empty] = p_values[d]
                                    c_empty += 1
                                elif val_vicino == 1 or val_vicino == 2:
                                    dir_tb[c_tb] = d
                                    prob_tb[c_tb] = p_values[d]
                                    c_tb += 1
                        
                        direzione = -1
                        
                        if c_empty > 0 or c_tb > 0:
                            if c_tb > 0 and c_empty > 0:
                                if np.random.binomial(1, P_Mi_move_Tb):
                                    direzione = numba_weighted_choice(dir_tb, prob_tb, c_tb)
                                else:
                                    direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
                            elif c_tb > 0:
                                direzione = numba_weighted_choice(dir_tb, prob_tb, c_tb)
                            else:
                                direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)

                        
                        if direzione != -1:
                            newx, newy = el0 + Direction_x[direzione], el1 + Direction_y[direzione]
                            
                            # check whether the chosen direction was empty
                            is_empty_dir = False
                            for i in range(c_empty):
                                if dir_empty[i] == direzione:
                                    is_empty_dir = True
                                    break
                            
                            if is_empty_dir:
                                meshCellular[newx, newy] = cellula
                                meshMovTime[newx, newy] = np.random.exponential(mov_vec[cellula]) + time
                                meshLifeTime[newx, newy] = life_cellula
                                meshInfected[newx, newy] = tb_mangiati
                                meshReplication[newx, newy] = 1e10
                            else:
                                # move Mi and eat TB -> increase infection
                                meshInfected[newx, newy] = tb_mangiati + 1 #first bacterial eated
                                meshReplication[newx, newy] = 1e10
                                if tb_mangiati + 1 >= Nici: # ==, but for safe code >=
                                    meshCellular[newx, newy] = cellula + 1 #became Mci
                                    meshMovTime[newx, newy] = np.random.exponential(scale=mov_vec[cellula+1]) + time
                                    meshLifeTime[newx, newy] = np.random.exponential(scale=life_vec[cellula+1]) + time
                                else:
                                    # Remain Mi (6)
                                    meshCellular[newx, newy] = cellula
                                    meshLifeTime[newx, newy] = np.random.exponential(scale=life_vec[cellula]) + time
                                    meshMovTime[newx, newy] = np.random.exponential(scale=mov_vec[cellula]) + time
        
                            meshCellular[el0, el1] = 0
                            meshLifeTime[el0, el1] = 1e10
                            meshMovTime[el0, el1] = 1e10
                            meshInfected[el0, el1] = 0
                            meshReplication[el0, el1] = 1e10
                        else:
                            meshMovTime[el0, el1] = np.random.exponential(mov_vec[cellula]) + time

                    

                    
                    elif cellula == 7:
                        life_cellula = meshLifeTime[el0, el1]
                    
                        tb_mangiati = meshInfected[el0, el1]
                        if tb_mangiati == 0:
                            print('no Mtb at this location; check agent state')
                        
                        for d in range(4):
                            newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                            
                            if 0 <= newx < Nx and 0 <= newy < Ny:
                                val_vicino = meshCellular[newx, newy]
                                if val_vicino == 0:
                                    dir_empty[c_empty] = d
                                    prob_empty[c_empty] = p_values[d]
                                    c_empty += 1
                                elif val_vicino == 1 or val_vicino == 2:
                                    dir_tb[c_tb] = d
                                    prob_tb[c_tb] = p_values[d]
                                    c_tb += 1
                        
                        direzione = -1
                        
                        if c_empty > 0 or c_tb > 0:
                            if c_tb > 0 and c_empty > 0:
                                if np.random.binomial(1, P_Mi_move_Tb):
                                    direzione = numba_weighted_choice(dir_tb, prob_tb, c_tb)
                                else:
                                    direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
                            elif c_tb > 0:
                                direzione = numba_weighted_choice(dir_tb, prob_tb, c_tb)
                            else:
                                direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
                    
                        
                        if direzione != -1:
                            newx, newy = el0 + Direction_x[direzione], el1 + Direction_y[direzione]
                            
                            # check whether the chosen direction was empty
                            is_empty_dir = False
                            for i in range(c_empty):
                                if dir_empty[i] == direzione:
                                    is_empty_dir = True
                                    break
                            
                            if is_empty_dir:
                                meshCellular[newx, newy] = cellula
                                meshMovTime[newx, newy] = np.random.exponential(mov_vec[cellula]) + time
                                meshLifeTime[newx, newy] = life_cellula
                                meshInfected[newx, newy] = tb_mangiati
                                meshReplication[newx, newy] = 1e10
                                
                                meshCellular[el0, el1] = 0
                                meshLifeTime[el0, el1] = 1e10
                                meshMovTime[el0, el1] = 1e10
                                meshInfected[el0, el1] = 0
                                meshReplication[el0, el1] = 1e10

                            else:
                                # --- SOGLIA DI BURST RAGGIUNTA ---
                                if tb_mangiati + 1 >= Ncib:
                                    # 1. the destination becomes caseum (8)
                                    # Mci has code 7; code 8 denotes caseum
                                    meshCellular[newx, newy] = cellula + 1
                                    meshInfected[newx, newy] = Ncib
                                    meshLifeTime[newx, newy] = 1e10
                                    # Usiamo np.random invece di rand
                                    meshMovTime[newx, newy] = time + np.random.exponential(movCaseum)
                                    meshReplication[newx, newy] = 1e10
                                    
                                    # 2. released bacterium occupies the original position (el0, el1)
                                    meshCellular[el0, el1] = 1 # Batterio (TB)
                                    meshLifeTime[el0, el1] = 1e10
                                    meshMovTime[el0, el1] = 1e10
                                    meshInfected[el0, el1] = 0
                                    meshReplication[el0, el1] = time + np.random.exponential(replication_slow)
                                    
                                    # 3. RILASCIO BATTERI NEL VICINATO DI MOORE (8 celle intorno a newx, newy)
                                    # prefer simple loops over complex slicing in Numba
                                    for di in range(-1, 2):    # da -1 a 1
                                        for dj in range(-1, 2):
                                            if di == 0 and dj == 0:
                                                continue
                                            
                                            mx, my = newx + di, newy + dj
                                            
                                            # boundary check
                                            if 0 <= mx < Nx and 0 <= my < Ny:
                                                val_vicino = meshCellular[mx, my]
                                                
                                                # direct comparisons are faster and more stable in Numba 
                                                # invece di "not in [8, 1000]"
                                                if val_vicino != 8 and val_vicino != 1000:
                                                    meshCellular[mx, my] = 1
                                                    meshReplication[mx, my] = time + np.random.exponential(replication_slow)
                                                    meshLifeTime[mx, my] = 1e10
                                                    meshMovTime[mx, my] = 1e10
                                                    meshInfected[mx, my] = 0
                                else:
                                    # eat TB without burst: remain Mci (7)
                                    meshCellular[newx, newy] = cellula
                                    meshInfected[newx, newy] = tb_mangiati + 1
                                    meshLifeTime[newx, newy] = life_cellula
                                    meshMovTime[newx, newy] = np.random.exponential(scale=mov_vec[cellula]) + time
                                    meshReplication[newx, newy] = 1e10
                                    
                                    meshCellular[el0, el1] = 0
                                    meshLifeTime[el0, el1] = 1e10
                                    meshMovTime[el0, el1] = 1e10
                                    meshInfected[el0, el1] = 0
                                    meshReplication[el0, el1] = 1e10

                        else:
                            meshMovTime[el0, el1] = np.random.exponential(mov_vec[cellula]) + time


                    elif cellula == 4:
                        life_cellula = meshLifeTime[el0, el1]
                    
                        for d in range(4):
                            newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                            
                            if 0 <= newx < Nx and 0 <= newy < Ny:
                                val_vicino = meshCellular[newx, newy]
                                if val_vicino == 0:
                                    dir_empty[c_empty] = d
                                    prob_empty[c_empty] = p_values[d]
                                    c_empty += 1
                                elif val_vicino == 1 or val_vicino == 2:
                                    dir_tb[c_tb] = d
                                    prob_tb[c_tb] = p_values[d]
                                    c_tb += 1
                        
                        direzione = -1
                        
                        if c_empty > 0 or c_tb > 0:
                            if c_tb > 0 and c_empty > 0:
                                if np.random.binomial(1, P_Mi_move_Tb):
                                    direzione = numba_weighted_choice(dir_tb, prob_tb, c_tb)
                                else:
                                    direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
                            elif c_tb > 0:
                                direzione = numba_weighted_choice(dir_tb, prob_tb, c_tb)
                            else:
                                direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
                    
                        
                        if direzione != -1:
                            newx, newy = el0 + Direction_x[direzione], el1 + Direction_y[direzione]
                            
                            # check whether the chosen direction was empty
                            is_empty_dir = False
                            for i in range(c_empty):
                                if dir_empty[i] == direzione:
                                    is_empty_dir = True
                                    break
                            
                            if is_empty_dir:
                                meshCellular[newx, newy] = cellula
                                meshMovTime[newx, newy] = np.random.exponential(mov_vec[cellula]) + time
                                meshLifeTime[newx, newy] = life_cellula
                                meshInfected[newx, newy] = 0
                                meshReplication[newx, newy] = 1e10
                                                                
                                # Clear original cell
                                meshCellular[el0, el1] = 0
                                meshLifeTime[el0, el1] = 1e10
                                meshMovTime[el0, el1] = 1e10
                                meshInfected[el0, el1] = 0
                                meshReplication[el0, el1] = 1e10
                                
                            else: # Movement into TB cell
                                if np.random.binomial(1, P_Mr_kill_tb): # Kill TB but don't become infected
                                    meshCellular[newx, newy] = cellula #Mr
                                    meshMovTime[newx, newy] = np.random.exponential(scale=mov_vec[cellula]) + time
                                    meshLifeTime[newx, newy] = life_cellula
                                    meshInfected[newx, newy] = 0
                                    meshReplication[newx, newy] = 1e10
                                    
                                    # Clear original cell
                                    meshCellular[el0, el1] = 0
                                    meshLifeTime[el0, el1] = 1e10
                                    meshMovTime[el0, el1] = 1e10
                                    meshInfected[el0, el1] = 0
                                    meshReplication[el0, el1] = 1e10
                                    
                                elif np.random.binomial(1, P_MrMi): # Become infected (Mi)
                                    meshCellular[newx, newy] = cellula + 2 
                                    meshMovTime[newx, newy] = np.random.exponential(scale=mov_vec[cellula + 2]) + time
                                    meshLifeTime[newx, newy] = np.random.exponential(scale=life_vec[cellula + 2]) + time
                                    meshInfected[newx, newy] = 1
                                    meshReplication[newx, newy] = 1e10
                                    
                                    # Clear original cell
                                    meshCellular[el0, el1] = 0
                                    meshLifeTime[el0, el1] = 1e10
                                    meshMovTime[el0, el1] = 1e10
                                    meshInfected[el0, el1] = 0
                                    meshReplication[el0, el1] = 1e10
                                    
                                else: # Doesn't move but update time
                                    meshMovTime[el0, el1] = np.random.exponential(scale=mov_vec[cellula]) + time
                        else: # No movement possible
                            meshMovTime[el0, el1] = np.random.exponential(scale=mov_vec[cellula]) + time
                    



                    # --- bacterial logic (TB = 1, 2) ---
                    elif cellula == 1 or cellula == 2 or cellula == 3:
                        life_cellula = meshLifeTime[el0, el1]
                        
                        for d in range(4):
                            newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                            
                            if 0 <= newx < Nx and 0 <= newy < Ny:
                                val_vicino = meshCellular[newx, newy]
                                if val_vicino == 0:
                                    dir_empty[c_empty] = d
                                    prob_empty[c_empty] = p_values[d]
                                    c_empty += 1
                        
                        direzione = -1
                        
                        if c_empty > 0:
                            direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
                            nx, ny = el0 + Direction_x[direzione], el1 + Direction_y[direzione]
                            # Sposta batterio
                            meshCellular[nx, ny] = cellula
                            meshMovTime[nx, ny] = np.random.exponential(mov_vec[cellula]) + time
                            meshLifeTime[nx, ny] = meshLifeTime[el0, el1]
                            # Reset vecchia
                            meshCellular[el0, el1] = 0
                            meshMovTime[el0, el1] = 1e10
                            meshLifeTime[el0, el1] = 1e10                            
                            meshInfected[el0, el1] = 0 
                            meshReplication[el0, el1] = 1e10

                        else: # Nessun movimento possibile
                            meshMovTime[el0, el1] = np.random.exponential(scale=mov_vec[cellula]) + time


        else: # Ntot_tb = 0
            if len(possible_mov_not_cas) > 0:
                # shuffle rows for random movement order
                np.random.shuffle(possible_mov_not_cas)
                for idx in range(len(possible_mov_not_cas)):
                    el0, el1 = possible_mov_not_cas[idx, 0], possible_mov_not_cas[idx, 1]
                    cellula = meshCellular[el0, el1]
                    
                    p_values = np.array([meshProbN[el0, el1], meshProbR[el0, el1], meshProbS[el0, el1], meshProbL[el0, el1]])

                    # --- Numba-compatible implementation ---
                    # preallocate space for up to four empty directions
                    dir_empty = np.empty(4, dtype=np.int32)
                    prob_empty = np.empty(4, dtype=np.float64)
                    c_empty = 0 # Contatore elementi validi
                
                    # 0 empty, 1 TB slow, 2 Tb fast, 3 T, 4 Mr, 5 Ma, 6 Mi, 7 Mci, 8 caseum
                    if 1 <= cellula <= 7:
                        life_cellula = meshLifeTime[el0, el1]
                        
                        for d in range(4):
                            newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                            
                            if 0 <= newx < Nx and 0 <= newy < Ny:
                                val_vicino = meshCellular[newx, newy]
                                if val_vicino == 0:
                                    dir_empty[c_empty] = d
                                    prob_empty[c_empty] = p_values[d]
                                    c_empty += 1

                        
                        if c_empty > 0:
                            direzione = numba_weighted_choice(dir_empty, prob_empty, c_empty)
                            newx, newy = el0 + Direction_x[direzione], el1 + Direction_y[direzione]
                            # Move cell to new position
                            meshCellular[newx, newy] = cellula
                            meshLifeTime[newx, newy] = life_cellula
                            meshMovTime[newx, newy] = np.random.exponential(scale=mov_vec[cellula]) + time
                            meshInfected[newx, newy] = 0
                            meshReplication[newx, newy] = 1e10
                            
                            # Clear original cell position
                            meshCellular[el0, el1] = 0
                            meshLifeTime[el0, el1] = 1e10
                            meshMovTime[el0, el1] = 1e10
                            meshInfected[el0, el1] = 0
                            meshReplication[el0, el1] = 1e10
                        else: # Nessun movimento possibile
                            meshMovTime[el0, el1] = np.random.exponential(scale=mov_vec[cellula]) + time

                    
        # --- CASEUM MOVEMENT (CELL TYPE 8) ---
        if len(possible_mov_cas) > 0:
            np.random.shuffle(possible_mov_cas)
            
            for idx in range(len(possible_mov_cas)):
                el0, el1 = possible_mov_cas[idx, 0], possible_mov_cas[idx, 1]
                cellula = meshCellular[el0, el1]
            
                # probabilities drawn for this cell
                # (N=0, R=1, S=2, L=3)
                p_values = np.array([meshProbN[el0, el1], meshProbR[el0, el1], meshProbS[el0, el1], meshProbL[el0, el1]])

                # preallocate space for up to four empty directions
                dir_available = np.empty(4, dtype=np.int32)
                prob_available = np.empty(4, dtype=np.float64)
                c_available = 0 # Contatore elementi validi
            
                # 1. IDENTIFY AVAILABLE DIRECTIONS (Avoid Caseum and mesh boundaries)
                for d in range(4):
                    newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                    
                    if 0 <= newx < Nx and 0 <= newy < Ny:
                        val_vicino = meshCellular[newx, newy]
                        if val_vicino != 8 and val_vicino != 1000:
                            dir_available[c_available] = d
                            prob_available[c_available] = p_values[d]
                            c_available += 1

                
                # 2. CHOOSE DIRECTION
                if c_available > 0:
                    direzione = numba_weighted_choice(dir_available, prob_available, c_available)
                    newx, newy = el0 + Direction_x[direzione], el1 + Direction_y[direzione]
                    # Backup data of the target cell (to be swapped)
                    cell = meshCellular[newx, newy]
                    lif  = meshLifeTime[newx, newy]
                    mov  = meshMovTime[newx, newy]
                    inf  = meshInfected[newx, newy]
                    rep  = meshReplication[newx, newy]
                    
                    # Move Caseum to new position (with specific Caseum attributes)
                    meshCellular[newx, newy] = cellula
                    meshLifeTime[newx, newy] = 1e10
                    meshMovTime[newx, newy]  = np.random.exponential(scale=movCaseum) + time
                    meshInfected[newx, newy] = Ncib # Caseum always carries Ncib
                    meshReplication[newx, newy] = 1e10
                    
                    # Move target cell's content back to original position (The Swap)
                    meshCellular[el0, el1] = cell
                    meshLifeTime[el0, el1] = lif
                    meshMovTime[el0, el1]  = mov
                    meshInfected[el0, el1] = inf
                    meshReplication[el0, el1] = rep
                else: # Nessun movimento possibile
                    meshMovTime[el0, el1] = np.random.exponential(scale=movCaseum) + time





        # ==========================================
        # REPLICATION TB
        # ==========================================
        repli_Tb = np.argwhere(meshReplication <= time) 
        
        if len(repli_Tb) > 0:
            np.random.shuffle(repli_Tb) # Usa np.random invece di rand
            
            for idx in range(len(repli_Tb)):
                el0, el1 = repli_Tb[idx, 0], repli_Tb[idx, 1]
                cellula = meshCellular[el0, el1]
                
                # Sostituisce dir_empty = []
                dir_empty = np.empty(4, dtype=np.int32)
                c_empty = 0
                
                for d in range(4):
                    newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                    if 0 <= newx < Nx and 0 <= newy < Ny:
                        if meshCellular[newx, newy] == 0:
                            dir_empty[c_empty] = d
                            c_empty += 1
                
                if c_empty > 0:
                    # choose uniformly among empty directions
                    r_idx = np.random.randint(0, c_empty)
                    direzione = dir_empty[r_idx]
                    
                    newx, newy = el0 + Direction_x[direzione], el1 + Direction_y[direzione]
                    
                    if cellula == 1:
                        rep_scale = replication_slow
                    elif cellula == 2: # cellula == 2
                        rep_scale = replication_fast
                    else:
                        print(f'error: Not TB, cellula == {cellula}')
                    
                    # Figlia (Nuova posizione)
                    meshCellular[newx, newy] = cellula
                    meshLifeTime[newx, newy] = 1e10 
                    meshMovTime[newx, newy]  = 1e10 
                    meshInfected[newx, newy] = 0
                    meshReplication[newx, newy] = time + np.random.exponential(rep_scale)
                    
                    # Genitore (Vecchia posizione)
                    meshCellular[el0, el1] = cellula
                    meshLifeTime[el0, el1] = 1e10
                    meshMovTime[el0, el1]  = 1e10
                    meshInfected[el0, el1] = 0
                    meshReplication[el0, el1] = time + np.random.exponential(rep_scale)











        # ==========================================
        # 2. BACTERIAL DEATH (DRUGS) 
        # ==========================================
        mask_tbf = meshCellular == 2
        mask_tbf_kill = (meshD >= Drugkill_f) & mask_tbf
        
        mask_tbs = meshCellular == 1
        mask_tbs_kill = (meshD >= Drugkill_s) & mask_tbs
        
        # Sostituisce np.isin(meshCellular, [6, 7])
        mask_mi = (meshCellular == 6) | (meshCellular == 7)
        mask_mi_kill = (meshD >= Drugkill_mi) & mask_mi
        
        # Morte TBF
        kill_f_idx = np.argwhere(mask_tbf_kill)
        for i in range(len(kill_f_idx)):
            r, c = kill_f_idx[i, 0], kill_f_idx[i, 1]
            meshCellular[r, c] = 0
            meshMovTime[r, c] = 1e10
            meshLifeTime[r, c] = 1e10
            meshInfected[r, c] = 0
            meshReplication[r, c] = 1e10
            
        # Morte TBS
        kill_s_idx = np.argwhere(mask_tbs_kill)
        for i in range(len(kill_s_idx)):
            r, c = kill_s_idx[i, 0], kill_s_idx[i, 1]
            meshCellular[r, c] = 0
            meshMovTime[r, c] = 1e10
            meshLifeTime[r, c] = 1e10
            meshInfected[r, c] = 0
            meshReplication[r, c] = 1e10
            
        # Cura Macrofagi (Mi/Mci -> Mr)
        kill_mi_idx = np.argwhere(mask_mi_kill)
        for i in range(len(kill_mi_idx)):
            r, c = kill_mi_idx[i, 0], kill_mi_idx[i, 1]
            meshCellular[r, c] = 4
            meshInfected[r, c] = 0
            meshReplication[r, c] = 1e10
            # Niente argomenti 'size', lo facciamo inline
            meshMovTime[r, c] = time + np.random.exponential(movMr)
            meshLifeTime[r, c] = time + np.random.exponential(lifeMr)






        # ==========================================
        # IMMUNE SYSTEM INTERACTIONS (T-Cells & Macrophages)
        # ==========================================
        t_cells = np.argwhere(meshCellular == 3)
        for idx in range(len(t_cells)):
            el0, el1 = t_cells[idx, 0], t_cells[idx, 1]
            for d in range(4):
                newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                if 0 <= newx < Nx and 0 <= newy < Ny:
                    target = meshCellular[newx, newy]
                    
                    if target == 6:
                        if np.random.binomial(1, dt_copy * P_activ_MiMa):
                            meshCellular[newx, newy] = 5
                            meshMovTime[newx, newy] = time + np.random.exponential(movMa)
                            meshLifeTime[newx, newy] = time + np.random.exponential(lifeMa)
                            meshInfected[newx, newy] = 0
                            meshReplication[newx, newy] = 1e10
                            target = 5
                            
                    if target == 6 or target == 7: # Evita l'operatore 'in'
                        if np.random.binomial(1, dt_copy * P_kill_T_Mi):
                            meshCellular[newx, newy] = 8
                            meshMovTime[newx, newy] = time + np.random.exponential(movCaseum)
                            meshLifeTime[newx, newy] = 1e10
                            meshInfected[newx, newy] = Ncib
                            meshReplication[newx, newy] = 1e10
                            
                    if target == 4:
                        if np.random.binomial(1, dt_copy * P_activ_MrMa):
                            meshCellular[newx, newy] = 5
                            meshMovTime[newx, newy] = time + np.random.exponential(movMa)
                            meshLifeTime[newx, newy] = time + np.random.exponential(lifeMa)
                            meshInfected[newx, newy] = 0
                            meshReplication[newx, newy] = 1e10

        ma_cells = np.argwhere(meshCellular == 5)
        for idx in range(len(ma_cells)):
            el0, el1 = ma_cells[idx, 0], ma_cells[idx, 1]
            for d in range(4):
                newx, newy = el0 + Direction_x[d], el1 + Direction_y[d]
                if 0 <= newx < Nx and 0 <= newy < Ny:
                    if meshCellular[newx, newy] == 8:
                        if np.random.binomial(1, dt_copy * P_Ma_healt_caseum):
                            meshCellular[newx, newy] = 0
                            meshMovTime[newx, newy] = 1e10
                            meshLifeTime[newx, newy] = 1e10
                            meshInfected[newx, newy] = 0
                            meshReplication[newx, newy] = 1e10



        # ==========================================
        # OXYGEN THRESHOLDS
        # ==========================================
        n_tbs = np.count_nonzero(mask_tbs)
        if n_tbs != 0:
            mask_slow_to_fast = (meshO >= Oslow_to_fast) & mask_tbs
            stf_idx = np.argwhere(mask_slow_to_fast)
            for i in range(len(stf_idx)):
                r, c = stf_idx[i, 0], stf_idx[i, 1]
                meshCellular[r, c] = 2
                meshReplication[r, c] = time + np.random.exponential(replication_fast)

        n_tbf = np.count_nonzero(mask_tbf)
        if n_tbf != 0:
            mask_fast_to_slow = (meshO <= Ofast_to_slow) & mask_tbf
            fts_idx = np.argwhere(mask_fast_to_slow)
            for i in range(len(fts_idx)):
                r, c = fts_idx[i, 0], fts_idx[i, 1]
                meshCellular[r, c] = 1
                meshReplication[r, c] = time + np.random.exponential(replication_slow)

        # ==========================================
        # RECRUITMENT (Mr AND T CELLS)
        # ==========================================
        Ntb = n_tbs + n_tbf
        Ncas = np.count_nonzero(meshCellular == 8)
        
        ProbT = 0.0
        ProbM = 0.0
        
        if Ntb >= NstartRec:
            ProbT = dt_copy * Pnew_T
            ProbM = dt_copy * Pnew_Mr
        elif Ncas >= 0 and Ntb == 0:
            ProbT = dt_copy * Pnew_T / reductionT
            ProbM = dt_copy * Pnew_Mr / reductionMr
            
        if ProbT > 0 or ProbM > 0:
            # BC_index must be a 2D NumPy array such as np.array([[x,y], [x,y]])
            for i in range(len(BC_index)):
                el0, el1 = BC_index[i, 0], BC_index[i, 1]
                
                if meshCellular[el0, el1] == 0:
                    if np.random.binomial(1, ProbT):
                        meshCellular[el0, el1] = 3
                        meshMovTime[el0, el1]  = time + np.random.exponential(movT)
                        if Ntb == 0:
                            meshLifeTime[el0, el1] = time + np.random.exponential(life_factorT * lifeT)                            
                        else:
                            meshLifeTime[el0, el1] = time + np.random.exponential(lifeT)
                        meshInfected[el0, el1] = 0
                        meshReplication[el0, el1] = 1e10
                        
                    elif np.random.binomial(1, ProbM):
                        meshCellular[el0, el1] = 4
                        meshMovTime[el0, el1]  = time + np.random.exponential(movMr)
                        meshLifeTime[el0, el1] = time + np.random.exponential(lifeMr)
                        meshInfected[el0, el1] = 0
                        meshReplication[el0, el1] = 1e10

    return meshCellular, meshMovTime, meshLifeTime, meshInfected, meshReplication    
                    
                    
                    
 




#%%


# 1D

def Ax_op_1D(x, param):
    """ 
    Return (I + lambda A_h) x. 
    given domain
    """
    dx = param[0]
    dt = param[1]
    D = param[2]
    
    lamb = D*dt / dx**2
    n = len(x) - 1
    Ax = np.zeros_like(x)
    #fix BC
    Ax[0], Ax[-1] = x[0], x[-1]
    #operate only on interior cells
    for i in range(1,n):
        Ax[i] = x[i] + lamb * (2 * x[i] - x[i-1] - x[i+1])
    return Ax

def thomas_algorithm(a, b, c, r):
    """ Compute LU factorization of tridiagonal matrix insitu and solve Tx=r
    NB a,b,c are of the same lenght, but the first element of a and the last element of b are not used
    """
    n = len(b)
    for j in range(1,n):
        a[j] /= b[j-1]
        b[j] -= a[j] * c[j-1]
        r[j] -= a[j] * r[j-1]
    r[-1] /= b[-1]
    for j in reversed(range(n-1)):
        r[j] = (r[j] - c[j] * r[j+1]) / b[j]








def simulation1D_polar_multiD_conservative_sigmoidal(D, params):
    '''
    
    Parameters
    ----------
    D : array of lenght Nx-1 (D_1/2, D_1+1/2, ..., D_Nx-2+1/2)
    params : tIni, tEnd, mesh_inside_index_don_1d, initial_mesh, Len_x, dx, dt, D_out, D_out_change, elim_rate, t_stop_dose, t_change_dose 
    mesh_inside_index_don_1d is from inner to outer region (it contains also the boundary)
    ka and ke: absorption and elimination rate (clearance) for drug BC 

    Returns
    -------
    None.

    '''
    tIni, tEnd, mesh_inside_index_don_1d, initial_mesh, Len_x, dx, dt, BC_conc, elim_rate, t_stop_dose = params
    n_don = len(mesh_inside_index_don_1d)
    Nx = len(initial_mesh)
    centre = round((Nx-1)/2)
    indexl = np.linspace(1, centre-1, centre -1, dtype=int) #array of length (Nx-1)/2 -1
    indexr = np.linspace(centre+1,Nx-2, centre -1, dtype=int)
    Dlm = D[:centre-1]
    Dlp = D[1:centre]
    Drm = D[centre:-1]
    Drp = D[centre+1:]
    R = Len_x/2.
    ind = np.linspace(0,(Nx-1)/2,round((Nx-1)/2) +1, dtype=int)
    rs = np.abs(dx*ind - R)
    rs = np.concatenate((rs, np.flip(rs[:-1])))
    # rs[centre] = 1e-5
    rs[centre] = 0
    time = tIni
    dt_copy = dt
    mesh = initial_mesh.copy()
    
    lamb1_don_bis_lm = Dlm/(dx**2)
    lamb1_don_bis_lp = Dlp/(dx**2)#array of length (Nx-1)/2 -1(then i will multiply for dt)
    lamb1_don_bis_rm = Drm/(dx**2)
    lamb1_don_bis_rp = Drp/(dx**2)    
    
    while time<tEnd - 1e-10:
        # BC = BC_conc2(time, D = D_out, ka=ka, ke=ke, t_stop=t_stop_dose, t_change=t_change_dose, D_change=D_out_change)
        BC = BC_conc(time)
        mesh[0], mesh[-1] = BC, BC
        # match exit time
        if time+dt_copy>tEnd:
            dt_copy = tEnd-time     
        # update time
        time += dt_copy         
        if math.isnan(time):
            print("TIME IS NAN")
            sys.exit("Exiting program")
        
        lamb1_don_lm = lamb1_don_bis_lm*dt_copy
        lamb1_don_lp = lamb1_don_bis_lp*dt_copy
        lamb1_don_rm = lamb1_don_bis_rm*dt_copy
        lamb1_don_rp = lamb1_don_bis_rp*dt_copy
        
        
        #update interior cells (solution of Ax = b, with A tridiagonal (ld,d,ud))
        # since we are working insitu b=mesh
        ld = np.zeros(Nx) #lower diagonal
        d = np.ones(Nx) #diagonal
        ud = np.zeros(Nx) #upper diagonal
        
        ld[indexl] = -lamb1_don_lm*((rs[indexl] + (dx/2))/rs[indexl])        
        ld[indexr] = -lamb1_don_rm*((rs[indexr] - (dx/2))/rs[indexr])
        d[indexl] = 1 + lamb1_don_lp*((rs[indexl] - (dx/2))/rs[indexl]) + lamb1_don_lm*((rs[indexl] + (dx/2))/rs[indexl])
        d[indexr] = 1 + lamb1_don_rp*((rs[indexr] + (dx/2))/rs[indexr]) + lamb1_don_rm*((rs[indexr] - (dx/2))/rs[indexr])
        ud[indexl] = -lamb1_don_lp*((rs[indexl] - (dx/2))/rs[indexl])
        ud[indexr] = -lamb1_don_rp*((rs[indexr] + (dx/2))/rs[indexr]) 
        
        
        #centre with r=0      
        #radial diffusion: (∂t u)_0 = 2 D (u_1 - u_0) / dx^2
        lambda_c = D[centre] * dt_copy / dx**2

        ld[centre] = -lambda_c
        d[centre]  = 1.0 + 2.0 * lambda_c
        ud[centre] = -lambda_c

        #consumo
        d[mesh_inside_index_don_1d[-1]] += dt*elim_rate
        
        
        # ld[centre] = 0
        # d[centre] = 1
        # ud[centre] = 0
        
        
        #update with the Thomas algorithm
        thomas_algorithm(ld, d, ud, mesh)
        # mesh[centre] = (mesh[centre-1] + mesh[centre +1])*0.5
        
    res = []
    for don in mesh_inside_index_don_1d:
        mean = np.mean(mesh[don])
        res += [mean]
    res = np.asarray(res)
    
    # res is the mean in each donuts from inner to outer
    # print(res)
    return(res, mesh)




def tridiag2matrix(a, b, c):
    return np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)


def BC_pk_model(t, y, ka1, ke, ka2, ka3):
    """
    ODE system for the three-compartment pharmacokinetic model.
    
    y[0]: Tratto G.I. (Dose da assorbire)
    y[1]: Polmone Extracellulare (La Boundary Condition)
    y[2]: Polmone Intracellulare (Macrofagi/Lisosomi - Il serbatoio)
    """
    y1, y2, y3 = y
    
    dy1 = -ka1 * y1
    dy2 = ka1 * y1 - (ke + ka2) * y2 + ka3 * y3
    dy3 = ka2 * y2 - ka3 * y3
    
    return [dy1, dy2, dy3]


def create_bc_function(doses, params, t_final, C0 = 0.0, resolution_per_day=100):
    """
    Simulate the PK profile with multiple doses using stop and restart.
    Simulate the complete PK profile once and return a callable 
    boundary concentration at time t.
    
    doses: List of (time, dose) tuples
    params: Tuple (ka1, ke, ka2, ka3)
    t_final: Final time in days
    resolution_per_day: Samples per day (higher gives more precise interpolation)
    
    return: Callable `bc(t)`
    """
    y0 = np.array([C0, 0.0, 0.0])
    t_all, y_all = [], []
    # Doses after the simulated interval must not create a backwards ODE step.
    doses = sorted((time, dose) for time, dose in doses if time < t_final)
    if not doses:
        raise ValueError("At least one dose must occur before t_final")
    
    for i, (t_dose, dose) in enumerate(doses):
        y0[0] += dose
        t_start = t_dose
        t_end = doses[i+1][0] if (i + 1 < len(doses)) else t_final
            
        if t_start == t_end:
            continue
            
        num_points = max(2, int((t_end - t_start) * resolution_per_day))
        t_eval = np.linspace(t_start, t_end, num=num_points)
        
        sol = solve_ivp(
            fun= BC_pk_model,
            t_span=[t_start, t_end],
            y0=y0,
            t_eval=t_eval,
            args=params,
            method='RK45',
            rtol=1e-6, atol=1e-8
        )
        
        if i + 1 < len(doses):
            t_all.append(sol.t[:-1])
            y_all.append(sol.y[:, :-1])
        else:
            t_all.append(sol.t)
            y_all.append(sol.y)
            
        y0 = sol.y[:, -1].copy()

    t_concat = np.concatenate(t_all)
    y_concat = np.concatenate(y_all, axis=1)
    
    # y_concat[1, :] is the extracellular compartment (BC)
    bc_array = y_concat[1, :]
    
    # Interpolate boundary concentration and define values outside the sampled range.
    bc_interpolator = interp1d(
        t_concat, bc_array, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(0.0, bc_array[-1])
    )
    
    return bc_interpolator



def hill(t, D_max, D_min, shape, b):
    'return D_max(1 - x^shape/b^shape+x^shape) + D_min, shape>1'
    return((D_max-D_min)*(1-t**shape/(b**shape + t**shape)) + D_min)

def hill_Ds(Nx,D_max=11000, D_min=7000,shape=2,b=1, ratio=1):
    'D : array of lenght Nx-1 (D_1/2, D_1+1/2, ..., D_Nx-2+1/2) (es. 0.5-99.5)'
    #ratio = Len_x for the simulation/Len_x used in fit
    #larger shape makes the transition steeper
    #larger b moves the transition inward
    D_values = []
    times = np.linspace(50/(Nx-1), ratio*50 - 50/(Nx-1), round((Nx-1)/2))
    Val = hill(times,D_max,D_min,shape,b)
    D_values = np.concatenate((Val, np.flip(Val)))
    D_values = np.asarray(D_values, dtype=float)
    return(D_values)






#%% functions diffusion_manual_domain



def sorted_BC_index(BC_selected_points, start, stop):
    """
    

    Parameters
    ----------
    BC_selected_points : BC points
    start : starting poin
    stop : end point
    
    NB. sorted in counterclock wise

    Returns
    -------
    BC_selected_points sorted from start to end

    """
    direction = 1
    #direction of the previous step
    # 1 = destra
    # 2 = su
    # 3 = sinistra
    # 4 = down
    
    sort = [BC_selected_points[0]]
    elem = sort[0]
    if [elem[0]+1,elem[1]] in BC_selected_points:
        elem = [elem[0]+1,elem[1]]
        sort += [elem]
        direction = 1
    elif [elem[0],elem[1]+1] in BC_selected_points:
        elem = [elem[0],elem[1]+1]
        sort += [elem]
        direction = 2
    else:
        print("controlla1")
    #the two points are now ordered
    for i in range(len(BC_selected_points)-1): #adjust endpoint below
        # directions in order: up, right, left, down
        if direction == 1:
            if  (([elem[0],elem[1]-1] in BC_selected_points) and ([elem[0],elem[1]-1] not in sort)):
                    elem = [elem[0],elem[1]-1] #down
                    sort += [elem]
                    direction = 4
            
            elif  (([elem[0]+1,elem[1]] in BC_selected_points) and ([elem[0]+1,elem[1]] not in sort)):
                    elem = [elem[0]+1,elem[1]] #destra
                    sort += [elem]
                    direction = 1
            
            elif  (([elem[0],elem[1]+1] in BC_selected_points) and ([elem[0],elem[1]+1] not in sort)):
                    elem = [elem[0],elem[1]+1] #su
                    sort += [elem]
                    direction = 2

            elif  (([elem[0]-1,elem[1]] in BC_selected_points) and ([elem[0]-1,elem[1]] not in sort)):
                    elem = [elem[0]-1,elem[1]] #sinistra
                    sort += [elem]
                    direction = 3
            else:
                print("controlla2")
        
        elif direction == 2:
                       
            if  (([elem[0]+1,elem[1]] in BC_selected_points) and ([elem[0]+1,elem[1]] not in sort)):
                    elem = [elem[0]+1,elem[1]] #destra
                    sort += [elem]
                    direction = 1
            
            elif  (([elem[0],elem[1]+1] in BC_selected_points) and ([elem[0],elem[1]+1] not in sort)):
                    elem = [elem[0],elem[1]+1] #su
                    sort += [elem]
                    direction = 2

            elif  (([elem[0]-1,elem[1]] in BC_selected_points) and ([elem[0]-1,elem[1]] not in sort)):
                    elem = [elem[0]-1,elem[1]] #sinistra
                    sort += [elem]
                    direction = 3
                    
            elif  (([elem[0],elem[1]-1] in BC_selected_points) and ([elem[0],elem[1]-1] not in sort)):
                    elem = [elem[0],elem[1]-1] #down
                    sort += [elem]
                    direction = 4
            else:
                print("controlla2")
        
        elif direction == 3:            
            
            if  (([elem[0],elem[1]+1] in BC_selected_points) and ([elem[0],elem[1]+1] not in sort)):
                    elem = [elem[0],elem[1]+1] #su
                    sort += [elem]
                    direction = 2

            elif  (([elem[0]-1,elem[1]] in BC_selected_points) and ([elem[0]-1,elem[1]] not in sort)):
                    elem = [elem[0]-1,elem[1]] #sinistra
                    sort += [elem]
                    direction = 3
                    
            elif  (([elem[0],elem[1]-1] in BC_selected_points) and ([elem[0],elem[1]-1] not in sort)):
                    elem = [elem[0],elem[1]-1] #down
                    sort += [elem]
                    direction = 4
            elif  (([elem[0]+1,elem[1]] in BC_selected_points) and ([elem[0]+1,elem[1]] not in sort)):
                    elem = [elem[0]+1,elem[1]] #destra
                    sort += [elem]
                    direction = 1
            else:
                print("controlla2")
        
        elif direction == 4:
            if  (([elem[0]-1,elem[1]] in BC_selected_points) and ([elem[0]-1,elem[1]] not in sort)):
                    elem = [elem[0]-1,elem[1]] #sinistra
                    sort += [elem]
                    direction = 3
                    
            elif  (([elem[0],elem[1]-1] in BC_selected_points) and ([elem[0],elem[1]-1] not in sort)):
                    elem = [elem[0],elem[1]-1] #down
                    sort += [elem]
                    direction = 4
            elif  (([elem[0]+1,elem[1]] in BC_selected_points) and ([elem[0]+1,elem[1]] not in sort)):
                    elem = [elem[0]+1,elem[1]] #destra
                    sort += [elem]
                    direction = 1
            elif  (([elem[0],elem[1]+1] in BC_selected_points) and ([elem[0],elem[1]+1] not in sort)):
                    elem = [elem[0],elem[1]+1] #su
                    sort += [elem]
                    direction = 2
            else:
                print("controlla2")
        else:
            print("controlla3")
        
        
        

    # print(sort)
    if start == stop:
        return(sort)
    
    ind_start = sort.index(start)
    ind_stop = sort.index(stop)
    
    if ind_start > ind_stop: #include the starting point when appropriate
        return(sort[ind_start:] + sort[:ind_stop+1])
    return(sort[ind_start:ind_stop+1])



def combine_BC_circle_and_other(BC_selected_points, start, stop, dx, Nx, Ny, R, centre_cerchio=None):
    """
    

    Parameters
    ----------
    BC_selected_points : new points to bind with the circle
    start : start point binding
    stop : end ppoint binding

    Returns
    -------
    BC_index

    """    
    #BC ceerchio
    if np.shape(centre_cerchio)[0] == 2:
        x_cerchio = np.linspace(0, 2*np.pi, 10**5)
        x_cerchio = R*np.cos(x_cerchio) + centre_cerchio[0]
        y_cerchio = np.linspace(0, 2*np.pi, 10**5)
        y_cerchio = R*np.sin(y_cerchio) + centre_cerchio[1]
    else:
        x_cerchio = np.linspace(0, 2*np.pi, 10**5)
        x_cerchio = R*np.cos(x_cerchio) + R
        y_cerchio = np.linspace(0, 2*np.pi, 10**5)
        y_cerchio = R*np.sin(y_cerchio) + R

    BC_index = []
    # ind_x_BC = int(np.floor(x_cerchio[0]/dx)) #find the index of the mesh for x
    # ind_y_BC = int(np.floor(y_cerchio[0]/dx))
    ind_x_BC = round(x_cerchio[0]/dx) #find the index of the mesh for x
    ind_y_BC = round(y_cerchio[0]/dx)
    BC_index = BC_index + [[ind_x_BC,ind_y_BC]]

    for i in range(1,len(x_cerchio)):
        # ind_x_BC = int(np.floor(x_cerchio[i]/dx)) #find the index of the mesh for x
        # ind_y_BC = int(np.floor(y_cerchio[i]/dx))
        ind_x_BC = round(x_cerchio[i]/dx) #find the index of the mesh for x
        ind_y_BC = round(y_cerchio[i]/dx)
        if [ind_x_BC,ind_y_BC] != BC_index[-1]:
            BC_index = BC_index + [[ind_x_BC,ind_y_BC]]
    #remove duplicates defensively:  
    # BC_index[:] = reduce(lambda l, x: l if x in l else l + [x], BC_index, [])

    if start == stop:
        return(BC_selected_points)
    
    ind_start = BC_index.index(start)
    ind_stop = BC_index.index(stop)
    
    if ind_start > ind_stop: #include the starting point when appropriate
        return(BC_index[ind_stop:ind_start+1] + BC_selected_points)
    return(BC_index[:ind_start+1] + BC_selected_points + BC_index[ind_stop:])


def tellme(s):
    print(s)
    plt.title(s, fontsize=16, wrap = True)
    plt.draw()

def zoom_factory(ax, max_xlim, max_ylim, base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
            # x_scale = scale_factor / 2
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
            # x_scale = scale_factor * 2
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        # new_width = (cur_xlim[1] - cur_xlim[0]) * x_scale
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        if xdata - new_width * (1 - relx) > max_xlim[0]:
            x_min = xdata - new_width * (1 - relx)
        else:
            x_min = max_xlim[0]
        if xdata + new_width * (relx) < max_xlim[1]:
            x_max = xdata + new_width * (relx)
        else:
            x_max = max_xlim[1]
        if ydata - new_height * (1 - rely) > max_ylim[0]:
            y_min = ydata - new_height * (1 - rely)
        else:
            y_min = max_ylim[0]
        if ydata + new_height * (rely) < max_ylim[1]:
            y_max = ydata + new_height * (rely)
        else:
            y_max = max_ylim[1]
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.figure.canvas.draw()

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun
