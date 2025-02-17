import numpy as np
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
import cvxpy as cp
import utils_mwm_sc as utils
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
            res = utils.max_weight_matching_scaled( time_steps, N, cust_process, serv_process, W, alpha )
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


def gen_long_weights(N_grid):
    Weights = np.zeros(shape=(N_grid,N_grid))
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
            xpoints = [ cell_locs[i][0] + 0.5, cell_locs[i][0] - 0.5, cell_locs[j][0] + 0.5,  cell_locs[j][0] - 0.5 ]
            ypoints = [ cell_locs[i][1] + 0.5, cell_locs[i][1] - 0.5, cell_locs[j][1] + 0.5,  cell_locs[j][1] - 0.5 ]
            Weights[i,j] = np.sqrt((max(xpoints) - min(xpoints))**2 
                                    + (max(ypoints) - min(ypoints))**2)
            
    print("Spatial: longest",Weights)

    return Weights

# Problem setup
Nvalues = [5]
for N in Nvalues:
    is_spatial = 1
    W = gen_long_weights(N)
    time_steps_M = 10#0000
    time_steps_B = 100000

    #Tbatches = np.asarray( [1,2,3, 4, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60, 100  ] )
    # alpha_values = np.asarray( [0, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8,0.9,0.95] )
    # Tbatches = np.array([10, 50, 100, 1000, 1500, 2500, 4000])
    # Tbatches = np.append(Tbatches,np.arange(5000, 100001, 5000))
    # alpha_values = np.array([0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999])
    alpha_values = np.asarray([0.84])
    Tbatches = np.asarray([4, 8, 12, 15, 18, 20, 21, 24, 30, 40, 50, 75, 90])
    #Tbatches = np.asarray( [5] )
    if is_spatial == 1:
        N = N*N
    p = 0.1
    pmf_cust = np.asarray([0.10773422, 0.0969608, 0.08726472, 0.07853824, 0.07068442, 0.06361598,
          0.05725438, 0.05152894, 0.04637605, 0.04173844, 0.0375646, 0.03380814,
          0.03042733, 0.02738459, 0.02464613, 0.02218152, 0.01996337, 0.01796703,
          0.01617033, 0.0145533, 0.01309797, 0.01178817, 0.01060935, 0.00954842,
          0.00859358])

    pmf_serv = np.asarray([0.02218152, 0.04637605, 0.07853824, 0.05725438, 0.00859358, 0.0969608,
            0.08726472, 0.01996337, 0.01617033, 0.02464613, 0.03042733, 0.10773422,
            0.02738459, 0.03380814, 0.06361598, 0.01178817, 0.0375646, 0.04173844,
            0.00954842, 0.01309797, 0.01060935, 0.05152894, 0.07068442, 0.0145533, 0.01796703])

    print("-----------------------------------")
    print("PMFs")
    print(pmf_cust,pmf_serv)
    print("-----------------------------------")

    M_res = run_max_weight( time_steps_M, N, pmf_cust, pmf_serv, W, alpha_values,0 )
    Q_paths_M = M_res["Q_paths"]
    C_paths_M = M_res["C_paths"]


    B_res = run_batching( time_steps_B, N, pmf_cust, pmf_serv, W, Tbatches,0 )
    Q_paths_B = B_res["Q_paths"]
    C_paths_B = B_res["C_paths"]
    (average_cost_B, average_queue_B) = utils.average_paths(C_paths_B,Q_paths_B)
    Q_paths_B = np.transpose( np.asarray( Q_paths_B)  )
    C_paths_B = np.transpose( np.asarray( C_paths_B ) )
    average_new = np.sum(C_paths_B,axis=0)/(Q_paths_B.shape[0] - Q_paths_B[-1,:])

    # (average_cost_M, average_queue_M) = utils.average_paths(C_paths_M,Q_paths_M)

    with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/lspu_' + str(N) + '_batch_rand.pkl', 'wb') as f:  
    # with open('./sim_paths_spatial_uniform_' + str(N) + '.pkl', 'wb') as f: 
        pickle.dump([C_paths_M, Q_paths_M, C_paths_B, Q_paths_B, average_new, N, W, pmf_cust, pmf_serv], f)
    print("Run completed successfully", N)


