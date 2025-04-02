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
N = 2
p = [0.3, 0.5, 0.8]
q = [0.2,0.6,0.9]
final = []
W = np.asarray([[1, 100],[1, 1]])
#for val in p:
  #for val2 in q:
    #final.append([val,val2])

final = [[0.8,0.2],[0.4,0.4],[0.7,0.7],[0.3,0.9]]


for [p,q] in final:

    is_spatial = 1
    
    time_steps_M = 10#0000
    time_steps_B = 200000

    alpha_values = np.asarray([0.84])
    #Tbatches = np.asarray([4, 8, 12, 15, 18, 20, 21, 24, 30, 40, 50, 60, 75, 90, 250, 200])
    #Tbatches = np.asarray([4, 5, 7, 8, 12, 15, 18, 20, 21, 24, 30, 35, 40, 45, 50, 55, 60, 65, 75])
    Tbatches = np.asarray([ 1, 4, 5, 7, 8, 12, 15, 18, 20, 21, 24, 30, 35, 38, 40, 44, 45, 50, 75])
    #Tbatches = np.asarray( [5] )

    pmf_cust = np.asarray([p,1-p])
    pmf_serv = np.asarray([q,1-q])

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

    # (average_cost_M, average_queue_M) = utils.average_paths(C_paths_M,Q_paths_M)

    with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/fc_2_' + str(int(p*10)) + str(int(q*10)) + '_batch.pkl', 'wb') as f:      
        # with open('./sim_paths_spatial_uniform_' + str(N) + '.pkl', 'wb') as f: 
        pickle.dump([C_paths_M, Q_paths_M, C_paths_B, Q_paths_B, N, W, pmf_cust, pmf_serv], f)
    print("Run completed successfully", N)

