from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        #print([(k,self.obs_dict[k]) for k in self.obs_dict])
        #print(np.vectorize(self.obs_dict.get)(Osequence))
        obs_index = np.vectorize(self.obs_dict.get)(Osequence)
        B_trans = np.transpose(self.B)
        new_obs_arr = B_trans[obs_index]
        new_obs_arr = np.transpose(new_obs_arr)
        if(self.pi.ndim == 1):
            self.pi = self.pi[np.newaxis,:]
        z1 = np.transpose(new_obs_arr)[0]
        z1 = z1[np.newaxis,:]
        alpha_base = np.multiply(self.pi,z1)
        S = len(self.pi[0])
        alpha_temp = np.zeros([L,S])
        alpha_temp[0] = alpha_base
        for t in range(1,L):
            transitions = np.transpose(self.A) * alpha_base
            transitions = np.sum(transitions,axis=1)
            #emission_t = B_trans[obs_index[t]]
            emission_t = np.transpose(new_obs_arr)[t]
            emission_t = emission_t[np.newaxis,:]
            alpha_base = np.multiply(emission_t,transitions)
            alpha_temp[t] = alpha_base
        alpha = alpha_temp.T
        #print(alpha)
        
        #alpha[:,0] = np.transpose(alpha_base)
        #obs_index = np.vectorize(self.obs_dict.get)(Osequence)
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi[0])
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        
        beta_base = np.ones((1,S))
        obs_index = np.vectorize(self.obs_dict.get)(Osequence)
        B_trans = np.transpose(self.B)
        new_obs_arr = B_trans[obs_index]
        new_obs_arr = np.transpose(new_obs_arr)
        
        beta_temp = np.zeros([L,S])
        beta_temp[L-1] = beta_base
        
        for t in reversed(range(L-1)):
            emission_t = np.transpose(new_obs_arr)[t+1]
            emission_t = emission_t[np.newaxis,:]
            transitions = emission_t * beta_base
            #emission_t = B_trans[obs_index[t]]
            prod = self.A*transitions
            beta_base = np.sum(prod,axis=1)
            beta_base = beta_base[np.newaxis,:]
            beta_temp[t] = beta_base
        beta = beta_temp.T
            
        
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        prob = np.sum(alpha[:,-1],axis=0)
        
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        prob = np.multiply(alpha,beta)/seq_prob
        
        
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi[0])
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        obs_index = np.vectorize(self.obs_dict.get)(Osequence)
        
        
        ksi = np.zeros([S,S,L-1])
        
        for t in range(L-1):
            for i in range(S):
                for j in range(S):
                    ksi[i,j,t] = (alpha[i,t]*self.A[i,j]*self.B[j,obs_index[t+1]]*beta[j,t+1])/seq_prob    
                    
        prob =ksi
        
        
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        
        S = len(self.pi[0])
        L = len(Osequence)
        
        obs_index = np.vectorize(self.obs_dict.get)(Osequence)
        B_trans = np.transpose(self.B)
        new_obs_arr = B_trans[obs_index]
        #new_obs_arr = np.transpose(new_obs_arr)
        delta = np.zeros([S,L])
        path_table = np.zeros([S,L],dtype = int)
        
        #prob = np.zeros([1,S])
        
        for i in range(S):
            delta[i,0] = self.pi[0][i] * new_obs_arr[0,i]
            path_table[i,0] = 0
        """for t in range(1,L):
            for i in range(S):
                for j in range(S):
                    prob[0,j] = new_obs_arr[t,i]*self.A[j,i]*delta[j,t-1]
                delta[i,t] = np.max(prob)
                path_table[i,j] = np.argmax(prob)
                print(prob)"""
        prob =[]
        for t in range(1, L):
            for i in range(S):
                for j in range(S):
                    temp = delta[j, t - 1] * self.A[j, i] * new_obs_arr[t,i]
                    prob.append(temp)
                delta[i, t] = np.max(prob)
                path_table[i, t] = np.argmax(prob)
                prob = []
                
        #print(path_table)    
        path_index = []
        path_index.append(np.argmax(delta[:,L-1]))
        for i in range(L-1,0,-1):
            temp = path_index[-1]
            path_index.append(path_table[temp,i])
        
        for i in range(len(path_index)):
            for key,value in self.state_dict.items():
                if value == path_index[i]:
                    path.append(key)
        
        #path = path_index
        
        path.reverse()
        
        ###################################################
        return path
