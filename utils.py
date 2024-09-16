import numpy as np
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB

def fluid_solution(N, W, pmf_cust, pmf_serv, verbose = 1):
    x = cp.Variable((N,N)) # fraction of matchings from i to j
    print(W)
    constraints = []                          
    for i in range(N):
        for j in range(N):
            constraints.append(x[i,j] >= 0)
            constraints.append(x[i,j] <= 1)
    # inequality to avoid round off, and violating equality is anyway penalised by objective
    for i in range(N):
        constraints.append(cp.sum(x[i,:]) == pmf_cust[i] ) 
    for j in range(N):
        constraints.append(cp.sum(x[:,j]) == pmf_serv[j] )                      

    objective = cp.Minimize( cp.sum( cp.multiply( x , W ) ) )
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if verbose:  
        print(problem)
    return({"X": x.value, "Cost": problem.value })


def fluid_solution_gurobi(N, W, pmf_cust, pmf_serv):

    model = gp.Model("Fluid Model")

    x = model.addMVar((N,N),lb=0,ub=1.0,vtype=GRB.CONTINUOUS, name="x")
    
    # Set objective
    obj = gp.quicksum(x[i, j] * W[i,j]
                            for i in range(N)
                            for j in range(N))
    model.setObjective( obj, GRB.MINIMIZE )
    
    # for i in range(N):
    #     for j in range(N):
    #         model.addConstr( x[i,j] >= 0 )
    #         model.addConstr( x[i,j] <= 1 )
    
    for i in range(N):
        model.addConstr( gp.quicksum(x[i,:]) == pmf_cust[i] )
    for j in range(N):
        model.addConstr( gp.quicksum(x[:,j]) == pmf_serv[j] )
    # Optimize model
    model.optimize()
    # Print solution
    if model.status == GRB.OPTIMAL:
        print('Objective Value: %.2f' % model.objVal)
        print('x: ', x.x)
    else:
        print('Optimization failed. Status:', model.status)
    return({"Cost": model.objVal, "X_matrix": x.x })

def average_paths(Cost_paths, Queue_paths, fract = 0, plot = 0):
    Queue_paths = np.transpose( np.asarray(Queue_paths) )
    Cost_paths = np.transpose( np.asarray(Cost_paths) )
    time_steps = len(Queue_paths[0])
    average_queue_lengths = np.average(Queue_paths[int(time_steps*fract):,:], axis = 0)
    average_cost = np.average(Cost_paths[int(time_steps*fract):,:], axis=0)
    if plot:
        plt.plot(average_queue_lengths,average_cost,'x')
        plt.show()
    return(average_cost,average_queue_lengths)

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
            constraints.append( cp.sum(M[i,:]) <= q[i] )
            constraints.append( cp.sum(M[:,i]) <= qtilde[i] )
        constraints.append( M[:,:]>=0 )

        objective = cp.Maximize( cp.sum( (1-alpha)*( cp.multiply( q, cp.sum(M, axis=1) ) + cp.multiply( qtilde, cp.sum(M, axis=0) ) ) )
                                - cp.sum( alpha*cp.multiply( W,M  ) ) )
        problem  = cp.Problem(objective, constraints)         
        # problem.solve(solver="GLPK_MI")
        problem.solve(solver="GLPK_MI", canon_backend=cp.SCIPY_CANON_BACKEND )
        # print("q=",q)
        # print("qtilde=",qtilde)
        # print(problem)
        

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

def batching( time_steps, N, cust_process, serv_process, W, T ):

    # Generate arrivals
    cust_arrivals = cust_process.rvs( size = time_steps )
    serv_arrivals = serv_process.rvs( size = time_steps )

    q = np.zeros(N)
    q[cust_arrivals[0]] = 1
    qtilde = np.zeros(N)
    qtilde[serv_arrivals[0]] = 1
    
    running_cost = 0
    cost_path = [0]
    total_queue = [] 
    total_queue.append(np.sum(q))

    for k in range(1,time_steps):
        # Check arrivals
        a = np.zeros(N)
        atilde = np.zeros(N)

        a[cust_arrivals[k]] = 1
        atilde[serv_arrivals[k]] = 1
        # State Update
        Mstar = np.zeros(shape=(N,N))
        
        # Matching is done every T time steps 
        if(k%T==0):
            constraints = []
            M = cp.Variable((N,N),integer=True)
            constraints = []
            for i in range(N):
                constraints.append( cp.sum(M[i,:]) == q[i] )
                constraints.append( cp.sum(M[:,i]) == qtilde[i] )
            constraints.append( M[:,:]>=0 )

            objective = cp.Minimize( cp.sum( cp.multiply( W, M ) ) )
            problem  = cp.Problem(objective, constraints)         
            # problem.solve(solver="GLPK_MI")
            problem.solve(solver="GLPK_MI", canon_backend=cp.SCIPY_CANON_BACKEND)
            Mstar = M.value

        running_cost += np.sum(np.multiply(Mstar,W))
        q = q + a - np.sum(Mstar, 1)
        if q[0] < 0:
            print(q-a+np.sum(Mstar, 1))
            print(Mstar, k)
            print(constraints)
            break
        qtilde = qtilde + atilde - np.sum(Mstar,0)
        
        total_queue.append(np.sum(q))
        cost_path.append(np.sum(np.multiply(Mstar,W)))
    return({"QP": total_queue, "CP": cost_path})


def qc_plot( Ex, fluid_sol, avg_Q_B, avg_C_B, avg_Q_M, avg_C_M, save_file =0 ):
    
    plt.figure( figsize=(10,6) ) 
    
    # C = 0 Case
    
    # C = Inf/Fluid Case
    ylim = max( max( avg_Q_B ), max( avg_Q_M ) )
    plt.plot([fluid_sol['Cost'],fluid_sol['Cost']],[0,ylim],linewidth = 3)
    # Batching data
    plt.plot(avg_C_B,avg_Q_B,'x',markersize = 10, markeredgewidth=3, markeredgecolor="darkblue")
    # Max Weight data
    plt.plot(avg_C_M, avg_Q_M,'go',markersize = 8)
    plt.plot([Ex],[1],'r*',markersize = 12, markeredgewidth=3)
    plt.ylabel("Average queue length", weight='bold',fontsize=12)
    plt.xlabel("Average cost", weight='bold',fontsize=12)
    plt.title("c vs q", weight='bold',fontsize=15)
    plt.legend(['Fluid', 'Batching', 'Max-weight', 'c = 0 sol'])
    # plt.legend([ 'Batching', 'Max-weight', 'c = 0 sol'])
    plt.grid(True)
    if save_file != 0:
        plt.savefig(save_file,bbox_inches='tight')
    plt.show()

def batching_gurobi( time_steps, N, cust_process, serv_process, W, T ):

    # Generate arrivals
    cust_arrivals = cust_process.rvs( size = time_steps )
    serv_arrivals = serv_process.rvs( size = time_steps )

    q = np.zeros(N)
    q[cust_arrivals[0]] = 1
    qtilde = np.zeros(N)
    qtilde[serv_arrivals[0]] = 1
    
    running_cost = 0
    cost_path = [0]
    total_queue = [] 
    total_queue.append(np.sum(q))

    for k in range(1,time_steps):
        # Check arrivals
        a = np.zeros(N)
        atilde = np.zeros(N)
        
        a[cust_arrivals[k]] = 1
        atilde[serv_arrivals[k]] = 1
        # State Update
        Mstar = np.zeros(shape=(N,N))
        
        # Matching is done every T time steps 
        if(k%T==0):
            model = gp.Model()
            M = model.addMVar((N,N),lb=0, vtype=GRB.INTEGER)
            model.setParam('OutputFlag',0)
            # Set objective
            obj = gp.quicksum(M[i, j] * W[i,j] for i in range(N) for j in range(N))
            model.setObjective( obj, GRB.MINIMIZE )
            
            for i in range(N):
                model.addConstr( gp.quicksum(M[i,:]) >= q[i] )
            for j in range(N):
                model.addConstr( gp.quicksum(M[:,j]) >= qtilde[j] )
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                Mstar = M.x
            else:
                print('Optimization failed. Status:', model.status)
            

        running_cost += np.sum(np.multiply(Mstar,W))
        q = q + a - np.sum(Mstar, 1)
        qtilde = qtilde + atilde - np.sum(Mstar,0)
        
        total_queue.append(np.sum(q))
        cost_path.append(np.sum(np.multiply(Mstar,W)))
    return({"QP": total_queue, "CP": cost_path})

def max_weight_gurobi( time_steps, N, cust_process, serv_process, W, alpha ):

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
        model = gp.Model()
        M = model.addMVar((N,N),lb=0, vtype = GRB.INTEGER)
        obj = gp.quicksum( ((1-alpha)*(q[i] + qtilde[j]) - alpha*W[i,j])*M[i,j] for i in range(N) 
                                                                for j in range(N) )
        for i in range(N):
            model.addConstr( gp.quicksum( M[i,j] for j in range(N) )<= q[i] )
        for j in range(N):
            model.addConstr( gp.quicksum( M[i,j] for i in range(N) )<= qtilde[j] )
        model.setObjective(obj, GRB.MAXIMIZE)
        model.setParam('OutputFlag',0)
        model.setParam('Method',1)
        model.optimize()
        a = np.zeros(N)
        atilde = np.zeros(N)

        a[cust_arrivals[k]] = 1
        atilde[serv_arrivals[k]] = 1

        if model.status == GRB.OPTIMAL:
            Matches = M.x
            matching_cost = np.sum(np.multiply(Matches,W))
            q = q + a - np.sum(Matches, 1)
            qtilde = qtilde + atilde - np.sum(Matches,0)
            total_queue.append(np.sum(q))
            cost_path.append(matching_cost)
        else:
            print('Optimization failed. Status:', model.status, '\t iteration', k)
            break

    return({"QP": total_queue, "CP": cost_path})

def minmax(average_cost_M, average_queue_M, average_cost_B, average_queue_B):
    len_M = len(average_cost_M)
    len_B = len(average_cost_B)
    max_val = 0
    for i in range(len_M):
        shifted = np.abs(average_queue_B - average_queue_M[i])
        idx = shifted.argmin()
        if(max_val < average_cost_B[idx] - average_cost_M[i]):
            max_val = average_cost_B[idx] - average_cost_M[i]
            (cM,qM,cB,qB)= ( average_cost_M[i], average_queue_M[i], average_cost_B[idx], average_queue_B[idx] )
    return cM, qM, cB, qB
