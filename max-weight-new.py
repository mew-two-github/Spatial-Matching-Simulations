import numpy as np
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
import cvxpy as cp
import pickle
# import gurobipy

def mw_single_path(time_steps, N, cust_process, serv_process, W, alpha):
    cust_arrivals = cust_process.rvs( size = time_steps )
    serv_arrivals = serv_process.rvs( size = time_steps )
    
    q = np.zeros(N)
    q[cust_arrivals[0]] = 1
    qtilde = np.zeros(N)
    qtilde[serv_arrivals[0]] = 1

    cost_path = [0]
    total_queue = []
    total_queue.append(np.sum(q))

    Mstar = np.zeros(shape=(2,2))

    for k in range(1,time_steps):
        M11 = cp.Variable(integer=True)
        M12 = cp.Variable(integer=True)
        M22 = cp.Variable(integer=True)
        M21 = cp.Variable(integer=True)
        constraints = []

        constraints.append( M11 + M12 <= q[0] )
        constraints.append( M21 + M22 <= q[1] )
        constraints.append( M11 + M21 <= qtilde[0] )
        constraints.append( M12 + M22 <= qtilde[1] )

        constraints.append( M11>=0 )
        constraints.append( M12>=0 )
        constraints.append( M21>=0 )
        constraints.append( M22>=0 )

        objective = cp.Maximize( (1-alpha)*( q[0]*(M11+M12) + q[1]*(M21 + M22) + qtilde[0]*(M11+M21) + qtilde[1]*(M12 + M22)  ) 
                                    - alpha*(W[0][0]*M11 + W[1][0]*M21 + W[0][1]*M12 + W[1][1]*M22) )
        problem  = cp.Problem(objective, constraints)
        problem.solve()

        # Check arrivals
        a = np.zeros(N)
        atilde = np.zeros(N)

        a[cust_arrivals[k]] = 1
        atilde[serv_arrivals[k]] = 1
        # State Update
        Mstar[0][0] = M11.value
        Mstar[0][1] = M12.value
        Mstar[1][0] = M21.value
        Mstar[1][1] = M22.value
        matching_cost = (W[0][0]*Mstar[0][0] + W[1][0]*Mstar[1][0] + W[0][1]*Mstar[0][1] + W[1][1]*Mstar[1][1]) 
        q = q + a - np.sum(Mstar, 1)
        qtilde = qtilde + atilde - np.sum(Mstar,0)

        total_queue.append(np.sum(q))
        cost_path.append(matching_cost)
    return({"QP": total_queue, "CP": cost_path})

# Problem setup
N = 2
W = np.asarray([[2.57353889, 2.95659375],[2.42280636, 5.37940011]])
arrival_ids = np.arange(0,N,1)
pmf_cust = np.asarray([0.76737737, 0.23262263])
pmf_serv = np.asarray([0.83284138, 0.16715862])

cust_process = rv_discrete(name='cust_arr', values=(arrival_ids, pmf_cust))
serv_process = rv_discrete(name='serv_arr', values=(arrival_ids, pmf_serv))

time_steps = 100000
cust_arrivals = cust_process.rvs( size = time_steps )
serv_arrivals = serv_process.rvs( size = time_steps )

Qpaths = []
Cpaths = []

alphas = np.asarray( [0, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8,0.9,0.95] )
for alpha in alphas:
    return_val = mw_single_path(time_steps, N, cust_process, serv_process, W, alpha)
    Qpaths.append(return_val["QP"])
    Cpaths.append(return_val["CP"])

with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/max_weight_random_2.pkl', 'wb') as f:  
# with open('./sim_paths_mw.pkl', 'wb') as f:
    pickle.dump([Qpaths,Cpaths, N, W, pmf_cust, pmf_serv], f)
