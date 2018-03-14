# This code is to calculate a probability at each state for "4x3 world" problem in AI textbook.
# 
# the visualization of 4x3 world problem 
#
# 3 [__, __, __, +1]
# 2 [__, wa, __, -1]
# 1 [sp, __, __, __]
#     1   2   3   4
#
# a order to speak each position
# horizon first, vertical second
# +1 => (4, 3)
#
# sp : a start point (1, 1)
# wa : a wall which is blocked
# +1 : reward +1
# -1 : reward -1
# __ : a space which the agent can stay
#
# the agent can go,
# LEFT with probability 0.1,   P(L) = 0.1
# RIGHT with probability 0.1   P(R) = 0.1
# UP with probability 0.8      P(U) = 0.8
#
# Reward
# Except +1, -1, R(s) = -0.04
#
# An Optimal policy for the R(s) = -0.04 is
#
# 3 [->, ->, ->,  1]
# 2 [ ^,  x,  ^, -1]
# 1 [ ^, <-, <-, <-]
#
# discount factor "gamma", i.e. "df" is 1
#
# the utility equation is,
# U^(pi) (s) = E[ sigma(t=0 to inf) (df^(t) * R(S_t))
# 
# S_t is Random Variable for state
# 
# Simply speaking, U(s) = U^(pi) (s)
# U(3, 3) = E[R(S_0) + R(S_1) + ...]
# S_0 = (3, 3)
# S_1 = (4, 3) or (3, 3) or (2, 3)
# .....

# coordinate order: left, up
# state form [x, x] (list type)

import ipdb
import random
import os
import copy

def state_generator():
    """
    - return 
    states : All states(=s) in the set of state(=S) [type : list, element : str]
    """
    states = []
    for i1 in range(1, 5):
        for i2 in range(1, 4):
            if i1 == 2 and i2 == 2:
                continue
            str_cord = str(i1)+","+str(i2)
            states.append(str_cord)
    return states

def action_generator():
    A = ['Up', 'Right', 'Down', 'Left']
    return A

def transition_generator():
    tm = {}
    tm['Up'] = [(0.8, [0, 1]), (0.1, [-1, 0]), (0.1, [1, 0])] # up, left, right
    tm['Right'] = [(0.8, [1, 0]),(0.1, [0, 1]), (0.1, [0, -1])] # right, up, down
    tm['Down'] = [(0.8, [0, -1]), (0.1, [1, 0]), (0.1, [-1, 0])]
    tm['Left'] = [(0.8, [-1, 0]), (0.1, [0, -1]), (0.1, [0, 1])] # left, down, up
    return tm

def reward_generator(States):
    """
    States : All states(=s) in the set of state(=S) [type : list, element : str
    """
    rewards = dict()
    for s in States:
        if s == "4,3":
            rewards[s] = 1
        elif s == "4,2":
            rewards[s] = -1
        else:
            rewards[s] = -0.04
    return rewards

#def find_max_value(s, A, States, U):
#    """
#    - arguments
#    s : present state [str]
#    A : Actions with state [dict]
#    States : states(=s) in the set of state(=S)
#    U : dictionary of utility(key is States)
#    - return
#    maximum value of action results
#    """
#
#    action_results = []
#    for a, pc_list in A.items():
#        action_r = 0
#        for p, direct in pc_list:
#
#            new_s = [sum(x) for x in zip(s, direct)]
#            if new_s in States:
#                if new_s == [4,3]:
#                    continue
#                elif new_s == [4,2]:
#                    continue
##                new_str = state2str(new_s)
#                action_r += p * U[new_s]
#            else:
#                if s == [4,3] or  s == [4,2]:
#                    continue
##                str = state2str(s)
#                action_r += p * U[s]
#        action_results.append(action_r)
#
#    return max(action_results)    
        

def str2cord(s):
    return map(lambda x: int(x), s.split(','))

def cord2str(s):
    return str(s[0])+','+str(s[1])

def get_neighbor(s, States):
    move = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    neighbor = []
    cord = str2cord(s)
    for m in move:
        cord_n = [sum(x) for x in zip(cord, s)]
        str_n = cord2str(cord_n)
        if str_n in States:
            neighbor.append(str_n)
        else:
            neighbor.append(s)

    neighbor = list(set(neighbor))
    return neighbor

def is_neighbor(s, move, States):
    cord = str2cord(s)
    cord_n = [sum(x) for x in zip(cord, move)]
    str_n = cord2str(cord_n)
    if str_n in States:
        return str_n, True
    else:
        return s, False

def value_iteration(States, A, TM, R, df=1, eps=0.0001, pf=False, piter=100):
    """
    - arguments
    States : all states in the set of state ['1,1', '1,2', ... '4,3']
    A : actions e.g. 
    TM : Transition Model P(s`|s, a)
    R : Rewards R(s)
    df : discount factor
    eps : epsilon, the maximum error allowed in the utility of any state
    - local varibles
    U, U` : vectors of utilities for states in S, initially zero
    delta : the maximum change in the utility of any state in an iteration
    - return 
    U : the final vector of utility for states in S
    """ 
    U = dict()
    U_new = dict()
    for s in States:
        U[s] = 0
        U_new[s] = 0
    delta = 0

    iter = 0
    while 1:
        iter += 1
        U = copy.deepcopy(U_new)
        delta = 0

        for s in States:  
            action_rslts = []
            if s == '4,3':
                U_new[s] = R[s]
                continue
            elif s == '4,2':
                U_new[s] = R[s]
                continue
            for a in A:
                pu_sum = 0
                for pc, move in TM[a]: 
                    n, _ = is_neighbor(s, move, States)
                    pu_sum += pc * U[n]
                action_rslts.append(pu_sum)
            max_act = max(action_rslts)
            
            U_new[s] = R[s] + df * max_act 
            n_delta = abs(U_new[s] - U[s])
            if n_delta > delta:
                delta = n_delta

        if pf:
            if iter % piter == 0:
                print("delta : {} , threshold : {}\n".format(delta, eps * (1 - df)/ df))

        if delta < eps * (1 - df) / df:
            break

    print("Iteration to converge : {}\n".format(iter)) 
    return U

def print_U(U):
#    for key_s, u in U.items():       
#        print("state ({},{}) : utility : {}\n".format(key_s[0], key_s[2], u))
    for vert in [3, 2, 1]:
        string = '['
        for horz in [1, 2, 3, 4]:
            s = str(horz)+','+str(vert)
            if s == '2,2':
                string += "_____"
            else:
                string += str(round(U[s], 3))
            if horz != 4:
                string += "  "
        
        string += ']'
        print(string)

if __name__ == "__main__":
    
    states = state_generator()
    actions = action_generator()
    tm = transition_generator()
    rewards = reward_generator(states)
    U = value_iteration(states, actions, tm, rewards, df=0.9999, eps=0.001, pf=True, piter=100)
    print_U(U)
