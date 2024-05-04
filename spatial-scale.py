import numpy as np
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
import cvxpy as cp
import utils
import pickle
# from tempfile import TemporaryFile

def run_max_weight( time_steps, N, pmf_cust, pmf_serv, W, alpha_values, gurobi_flag = 1 ):
    
    arrival_ids = np.arange(0,N,1)
    cust_process = rv_discrete(name='cust_arr', values=(arrival_ids, pmf_cust))
    serv_process = rv_discrete(name='serv_arr', values=(arrival_ids, pmf_serv))
    
    Q_paths = []
    C_paths = []
    for alpha in alpha_values:
        if gurobi_flag:
            res = utils.max_weight_gurobi( time_steps, N, cust_process, serv_process, W, alpha )
        else:
            res = utils.max_weight( time_steps, N, cust_process, serv_process, W, alpha )
        Q_paths.append(res["QP"])
        C_paths.append(res["CP"])
    return({"Q_paths": Q_paths,"C_paths":C_paths})

def run_batching( time_steps, N, pmf_cust, pmf_serv, W, Tbatches, gurobi_flag = 1 ):
    
    arrival_ids = np.arange(0,N,1)
    cust_process = rv_discrete(name='cust_arr', values=(arrival_ids, pmf_cust))
    serv_process = rv_discrete(name='serv_arr', values=(arrival_ids, pmf_serv))
    
    Q_paths = []
    C_paths = []
    for T in Tbatches:
        if gurobi_flag:
            res = utils.batching_gurobi( time_steps, N, cust_process, serv_process, W, T )
        else:
            res = utils.batching( time_steps, N, cust_process, serv_process, W, T )
        Q_paths.append(res["QP"])
        C_paths.append(res["CP"])
    return({"Q_paths": Q_paths,"C_paths":C_paths})

def plot_from_sim_paths(file_name):
    with open(file_name,'rb') as f:
        M_res,B_res, N, W, pmf_cust, pmf_serv = pickle.load(f)
    print(M_res)
    Q_paths_M = M_res["Q_paths"]
    C_paths_M = M_res["C_paths"]
    Q_paths_B = B_res["Q_paths"]
    C_paths_B = B_res["C_paths"]
    (average_cost_M, average_queue_M) = utils.average_paths(C_paths_M,Q_paths_M)
    (average_cost_B, average_queue_B) = utils.average_paths(C_paths_B,Q_paths_B)
    [Ex,fluid_sol] = run_extremes(N,W,pmf_cust,pmf_serv)
    utils.qc_plot(Ex,fluid_sol,average_queue_B,average_cost_B,average_queue_M,average_cost_M)

def generate_weights(N_grid, is_spatial = 0):
    Weights = np.zeros(shape=(N_grid,N_grid))
    if is_spatial:
        X = np.arange(0.5,N_grid,1)
        Y = np.arange(0.5,N_grid,1)
        cell_locs = []
        for i in range(N_grid):
            for j in range(N_grid):
                cell_locs.append([X[i],Y[j]])
        N_cells = N_grid*N_grid
        Weights = np.zeros(shape=(N_cells,N_cells))

        for i in range(N_cells):
            for j in range(N_cells):
                Weights[i,j] = np.sqrt((cell_locs[i][0] - cell_locs[j][0])**2 + (cell_locs[i][1] - cell_locs[j][1])**2)
        print("Spatial",Weights)
    else:
        rand_weights = np.random.rand(N_grid, N_grid)
        N_cells = N_grid*N_grid
        for counter in range(N_cells):
            for i in range(N_grid):
                for j in range(N_grid):
                    for k in range(N_grid):
                        rand_weights[i,j] = min(rand_weights[i,j],rand_weights[i,k] + rand_weights[k,j] )
        print("Random",rand_weights)
        Weights = rand_weights
    return(Weights)

# Problem setup
factors = [10,100]
time_steps_M = 1#00000
time_steps_B = 1#00000

Tbatches = np.asarray( [1,2, 3, 4, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60, 100  ] )
# alpha_values = np.asarray( [0, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8,0.9,0.95] )
# alpha_values = np.asarray( [0, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.825, 0.85, 0.875,  0.9, 0.925, 0.95] )
alpha_values = np.asarray( [0, 0.2, 0.25, 0.35, 0.4, 0.45,0.46, 0.47, 0.48, 0.49, 0.5, 0.55, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.825, 0.85, 0.875,  0.9, 0.925, 0.95] )
N = 16
pmf_cust = np.random.rand(N)
pmf_cust = pmf_cust/(np.sum(pmf_cust))

pmf_serv = np.random.rand(N)
pmf_serv = pmf_serv/(np.sum(pmf_serv))
# pmf_cust = [0.19,0.31,0.42,0.08]
# pmf_serv = [0.09,0.23,0.41,0.27]
print(pmf_cust,pmf_serv)

for factor in factors:
    N = 4
    is_spatial = 1
    W = generate_weights(N,is_spatial)*factor
    print("scaled weights",W)
    
    if is_spatial == 1:
        N = N*N

    M_res = run_max_weight( time_steps_M, N, pmf_cust, pmf_serv, W, alpha_values,0 )
    Q_paths_M = M_res["Q_paths"]
    C_paths_M = M_res["C_paths"]


    B_res = run_batching( time_steps_B, N, pmf_cust, pmf_serv, W, Tbatches,0 )
    Q_paths_B = B_res["Q_paths"]
    C_paths_B = B_res["C_paths"]
    (average_cost_M, average_queue_M) = utils.average_paths(C_paths_M,Q_paths_M)
    (average_cost_B, average_queue_B) = utils.average_paths(C_paths_B,Q_paths_B)

    with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/sim_paths_spatial_scale_' + str(N) + '.pkl', 'wb') as f:  
    # with open('./sim_paths_spatial_scale_' + str(factor) + '.pkl', 'wb') as f: 
        pickle.dump([M_res,B_res, N, W, pmf_cust, pmf_serv], f)

    print("Run completed successfully")


