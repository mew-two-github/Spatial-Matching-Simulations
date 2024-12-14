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
            res = utils.max_weight_G( time_steps, N, cust_process, serv_process, W, alpha )
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
    time_steps_M = 100000
    time_steps_B = 100000

    # Tbatches = np.asarray( [1,2, 3, 4, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60, 100  ] )
    # alpha_values = np.asarray( [0, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8,0.9,0.95] )
    #Tbatches = np.array([10, 50, 100, 1000, 1500, 2500, 4000])
    #Tbatches = np.append(Tbatches,np.arange(5000, 100001, 5000))
    #alpha_values = np.array([0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,0.46, 0.47, 0.48, 0.49, 0.5, 0.55, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.825, 0.85, 0.875,  0.9, 0.925, 0.95, 0.954, 0.958, 0.962, 0.966, 0.97, 0.974, 0.978, 0.982, 0.986, 0.99 ])
    alpha_values = np.asarray([0.84])
    Tbatches = np.arange(100,1001, 50)

    if is_spatial == 1:
        N = N*N

#    pmf_cust = np.random.rand(N)
#    pmf_cust = pmf_cust/(np.sum(pmf_cust))
    
    pmf_cust = [0.07975418, 0.01919971, 0.02258319, 0.063418  , 0.01690154, 0.00436386,
 0.0713188 , 0.03781694, 0.07588844, 0.02035429, 0.07178959, 0.01772757,
 0.037173  , 0.05643021, 0.03212186, 0.04991935, 0.04187258, 0.00426407,
 0.0004976 , 0.06667127, 0.05168924, 0.07233958, 0.01511579, 0.04976147,
 0.02102789]

    pmf_serv = [0.07027825, 0.01713261, 0.02124712, 0.01177862, 0.08442459, 0.04251858,
 0.01297518, 0.04118533, 0.05410482, 0.03633735, 0.07573665, 0.07874273,
 0.04071133, 0.06564617, 0.00056744, 0.03949442, 0.04429773, 0.07808677,
 0.00240852, 0.00708916, 0.06218571, 0.01051105, 0.04341731, 0.04109408,
 0.01802848]


    #pmf_serv = np.random.rand(N)
    #pmf_serv = pmf_serv/(np.sum(pmf_serv))

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

    with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/lspu_' + str(N) + '_largeq_random_batch_new.pkl', 'wb') as f:  
    # with open('./sim_paths_spatial_uniform_' + str(N) + '.pkl', 'wb') as f: 
        pickle.dump([C_paths_M, Q_paths_M, average_cost_B, average_queue_B, average_new, N, W, pmf_cust, pmf_serv], f)
    print("Run completed successfully", N)


