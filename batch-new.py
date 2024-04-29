import numpy as np
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
import cvxpy as cp
import pickle

def batch_single_path(time_steps, N, cust_process, serv_process, W, Tbatch):
    cust_arrivals = cust_process.rvs( size = time_steps )
    serv_arrivals = serv_process.rvs( size = time_steps )
    q = np.zeros(N)
    q[cust_arrivals[0]] = 1
    qtilde = np.zeros(N)
    qtilde[serv_arrivals[0]] = 1

    cost_path = [0]
    total_queue = []
    total_queue.append(np.sum(q))

    Q = []
    Qtilde=  []

    Q.append(q)
    Qtilde.append(qtilde)

    count = 0
    for k in range(1,time_steps):
        Mstar = np.zeros(shape=(2,2))
        # Batching
        if k%Tbatch == 0:
            count += 1
            M11 = cp.Variable(integer=True)
            M12 = cp.Variable(integer=True)
            M22 = cp.Variable(integer=True)
            M21 = cp.Variable(integer=True)
            constraints = []

            constraints.append( M11 + M12 == q[0] )
            constraints.append( M21 + M22 == q[1] )
            constraints.append( M11 + M21 == qtilde[0] )
            constraints.append( M12 + M22 == qtilde[1] )

            constraints.append( M11>=0 )
            constraints.append( M12>=0 )
            constraints.append( M21>=0 )
            constraints.append( M22>=0 )

            objective = cp.Minimize( W[0][0]*M11 + W[1][0]*M21 + W[0][1]*M12 + W[1][1]*M22 )
            problem  = cp.Problem(objective, constraints)
            problem.solve()

            Mstar[0][0] = M11.value
            Mstar[0][1] = M12.value
            Mstar[1][0] = M21.value
            Mstar[1][1] = M22.value

        # Check arrivals
        a = np.zeros(N)
        atilde = np.zeros(N)

        a[cust_arrivals[k]] = 1
        atilde[serv_arrivals[k]] = 1

        matching_cost = (W[0][0]*Mstar[0][0] + W[1][0]*Mstar[1][0] + W[0][1]*Mstar[0][1] + W[1][1]*Mstar[1][1])
        q = q + a - np.sum(Mstar, 1)
        qtilde = qtilde + atilde - np.sum(Mstar,0)

        total_queue.append(np.sum(q))
        cost_path.append(matching_cost)
        Q.append(q)
        Qtilde.append(qtilde)
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

# Tbatches = np.asarray( [3, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60, 100  ] )
Tbatches = np.asarray( [1,2, 3, 4, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60, 100  ] )
for Tbatch in Tbatches:
    return_val = batch_single_path(time_steps, N, cust_process, serv_process, W, Tbatch)
    Qpaths.append(return_val["QP"])
    Cpaths.append(return_val["CP"])

with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/sim_paths_random_2_batch.pkl', 'wb') as f:  
# with open('./sim_paths_batch.pkl', 'wb') as f:
    pickle.dump([Qpaths,Cpaths, N, W, pmf_cust, pmf_serv], f)
