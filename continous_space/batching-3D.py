import numpy as np
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
import cvxpy as cp
import pickle
from scipy.optimize import linear_sum_assignment

def batch_single_path(time_steps, grid_size, Tbatch):
    cust_arrivals_x = np.random.uniform(low=0, high=grid_size, size=time_steps)
    cust_arrivals_y = np.random.uniform(low=0, high=grid_size, size=time_steps)
    cust_arrivals_z = np.random.uniform(low=0, high=grid_size, size=time_steps)
    # cust_arrivals = np.column_stack((x_coords, y_coords))
    serv_arrivals_x = np.random.uniform(low=0, high=grid_size, size=time_steps)
    serv_arrivals_y = np.random.uniform(low=0, high=grid_size, size=time_steps)
    serv_arrivals_z = np.random.uniform(low=0, high=grid_size, size=time_steps)

    customers = []
    servers = []

    customers.append([cust_arrivals_x[0],cust_arrivals_y[0],cust_arrivals_z[0]])
    servers.append([serv_arrivals_x[0],serv_arrivals_y[0],serv_arrivals_z[0]])

    cost_path = [0]
    total_queue = []
    total_queue.append(len(customers))

    for k in range(1,time_steps):

        queue_length = len(customers)
        W = np.zeros(shape=(queue_length,queue_length))
        matching_cost = 0

        # Batching
        if k%Tbatch == 0:
            coords1 = np.array(customers)
            coords2 = np.array(servers)
            diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
            W = np.sqrt(np.sum(diff**2, axis=-1))
            row_ind, col_ind = linear_sum_assignment(W)
            matching_cost = W[row_ind,col_ind].sum()
            customers = []
            servers = []

        customers.append([cust_arrivals_x[k],cust_arrivals_y[k],cust_arrivals_z[k]])
        servers.append([serv_arrivals_x[k],serv_arrivals_y[k],serv_arrivals_z[k]])

        total_queue.append(len(customers))
        cost_path.append(matching_cost)

    return({"QP": total_queue, "CP": cost_path})

# Problem setup
N = 2

time_steps = 10

Qpaths = []
Cpaths = []

# Tbatches = np.asarray( [3, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60, 100  ] )
Tbatches = np.asarray( [1,2, 3, 4, 5, 8, 10, 12, 15, 18, 20, 30, 40, 50, 60, 100  ] )
Tbatches = np.append(Tbatches, np.arange(200,3001,200) )
for Tbatch in Tbatches:
        
    return_val = batch_single_path(time_steps, N, Tbatch)
    Qpaths.append(return_val["QP"])
    Cpaths.append(return_val["CP"])
    print("T =", Tbatch)
    # with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/continous/batch-new.pkl', 'wb') as f:
        # pickle.dump([Qpaths,Cpaths], f)

# with open('/storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/continous/batch-new-3D.pkl', 'wb') as f:  
with open('./batch-new-3D.pkl', 'wb') as f:
    pickle.dump([Qpaths,Cpaths], f)
print("Full Run Completed successfully")
