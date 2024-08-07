import numpy as np
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
import cvxpy as cp
import pickle
# from tempfile import TemporaryFile


def max_weight( time_steps, N, cust_process, serv_process, W, alpha ):

    cust_arrivals = cust_process.rvs( size = time_steps )
    serv_arrivals = serv_process.rvs( size = time_steps )

    q = np.zeros(N)
    q[cust_arrivals[0]] = 1
    qtilde = np.zeros(N)
    qtilde[serv_arrivals[0]] = 1

    cost_path = [0]
    total_queue = []
    total_queue.append(np.sum(q))

    for k in range(1,time_steps):
        # Maxweight Matching
        M = cp.Variable((N,N),integer=True)
        constraints = []
        for i in range(N):
            constraints.append( M[i,i] == 0)
            constraints.append( cp.sum(M[i,:]) <= q[i] )
            constraints.append( cp.sum(M[:,i]) <= qtilde[i] )
        constraints.append( M[:,:]>=0 )
        

        objective = cp.Maximize( cp.sum( (1-alpha)*( cp.multiply( q, cp.sum(M, axis=1) ) + cp.multiply( qtilde, cp.sum(M, axis=0) ) ) )
                                - cp.sum( alpha*cp.multiply( W,M  ) ) )
        problem  = cp.Problem(objective, constraints)         
        problem.solve(solver="GLPK_MI")
        
        

        # Check arrivals
        a = np.zeros(N)
        atilde = np.zeros(N)

        a[cust_arrivals[k]] = 1
        atilde[serv_arrivals[k]] = 1
        # State Update
        Mstar = M.value
        # print("M=", Mstar)
        matching_cost = np.sum(np.multiply(Mstar,W)) 
        q = q + a - np.sum(Mstar, 1)
        qtilde = qtilde + atilde - np.sum(Mstar,0)
        
        total_queue.append(np.sum(q))
        cost_path.append(matching_cost)
    
    return({"QP": total_queue, "CP": cost_path})

def run_max_weight( time_steps, N, pmf_cust, pmf_serv, W, alpha_values, gurobi_flag = 1 ):
    
    arrival_ids = np.arange(0,N,1)
    cust_process = rv_discrete(name='cust_arr', values=(arrival_ids, pmf_cust))
    serv_process = rv_discrete(name='serv_arr', values=(arrival_ids, pmf_serv))
    
    Q_paths = []
    C_paths = []
    for alpha in alpha_values:
        res = max_weight( time_steps, N, cust_process, serv_process, W, alpha )
        Q_paths.append(res["QP"])
        C_paths.append(res["CP"])
    return({"Q_paths": Q_paths,"C_paths":C_paths})


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
Nvalues = [2,3,4]
for N in Nvalues:
    is_spatial = 1
    W = generate_weights(N,is_spatial)
    time_steps_M = 100000

    Tbatches = np.asarray( [1,2, 3, 4, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60, 100  ] )
    # alpha_values = np.asarray( [0, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8,0.9,0.95] )
    alpha_values = np.asarray( [0, 0.2, 0.25, 0.35, 0.4, 0.45,0.46, 0.47, 0.48, 0.49, 0.5, 0.55, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.825, 0.85, 0.875,  0.9, 0.925, 0.95] )
    

    if is_spatial == 1:
        N = N*N

    pmf_cust = np.ones(N)*(1/N)
    pmf_serv = np.ones(N)*(1/N)

    print(pmf_cust,pmf_serv)

    M_res = run_max_weight( time_steps_M, N, pmf_cust, pmf_serv, W, alpha_values,0 )
    Q_paths_M = M_res["Q_paths"]
    C_paths_M = M_res["C_paths"]


    with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/max_weight_no_diag_' + str(N) + '.pkl', 'wb') as f:  
    # with open('./sim_paths_spatial_uniform_' + str(N) + '.pkl', 'wb') as f: 
        pickle.dump([M_res, N, W, pmf_cust, pmf_serv], f)

    print("Run completed successfully", N)


