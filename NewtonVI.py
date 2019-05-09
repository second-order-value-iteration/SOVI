import numpy as np
import mdptoolbox, mdptoolbox.example
import random
from numpy.random import seed
import time



#np.random.seed(0)
#P, R = mdptoolbox.example.forest()
s = 10
a= 5
discount = 0.9
episodes = 100
iterations = 50

vi_diff = np.zeros((episodes,iterations,1))
newton_diff= np.zeros((episodes,iterations,1))

vi_pol_diff = np.zeros((episodes,iterations,1))
newton_pol_diff = np.zeros((episodes,iterations,1))

vi_time = np.zeros((episodes,1))
newton_time = np.zeros((episodes,1))

for count in range(episodes):
    
    np.random.seed((count+1)*100)
    random.seed((count+1)*110)

    P, R = mdptoolbox.example.rand(s, a)
        #print(P)
        #print(np.min(P))

    vi = mdptoolbox.mdp.ValueIteration(P, R, discount,epsilon=0.0000001,max_iter = iterations)
    vi.run()
    #print(vi.V,vi.iter,vi.max_iter)

    # vi2 = mdptoolbox.mdp.QValueIteration(P,R,discount,max_iter = 1)
    # vi2.run()

    # print(np.max(vi2.Q,axis = 0),vi2.max_iter)
    # print(np.max(vi2.modQ,axis = 0))
    start = time.time()
    vi3 = mdptoolbox.mdp.QValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy)
    end = time.time()
    
    vi3.run()
    
    vi_time[count] = end-start

    #soft_error = 
    start = time.time()
    vi4 = mdptoolbox.mdp.NewtonQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy)
    end = time.time()
    
    vi4.run()
    #print(end-start)
    newton_time[count] = end-start
    
    vi_diff[count] = vi3.store_error
    newton_diff[count] = vi4.store_error
    
    vi_pol_diff[count] = vi3.policy_error
    newton_pol_diff[count] = vi4.policy_error


# print(np.max(vi3.Q,axis = 0),vi3.max_iter)
# print(np.max(vi3.modQ,axis = 0))
# print(np.max(vi3.nmodQ,axis = 0))

#print(np.linalg.norm(vi.V - np.max(vi3.Q,axis = 0),ord=np.inf)) #Q-bellman
#print(np.linalg.norm(vi.V - np.max(vi3.modQ,axis = 0), axis=0,ord=np.inf)) #Soft-max
#print(np.linalg.norm(vi.V - np.max(vi4.nmodQ,axis = 0),axis=0,ord=np.inf))
#print(vi3.store_error, vi4.store_error, vi3.policy,vi4.policy)
#print(vi3.policy_error,vi4.policy_error)
