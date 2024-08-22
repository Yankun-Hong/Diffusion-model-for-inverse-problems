import numpy as np

def generate_latent_var(S, seed1=None, seed2=None):
    output = np.empty((S,9))
    if seed1:
        np.random.seed(seed1)
    output[:,0] = np.random.randint(low=0, high=4, size=S)
    if seed2:
        np.random.seed(seed2)
    output[:,1:] = np.random.random_sample((S,8))
    output[:,1:5] = output[:,1:5]*0.25+0.05
    return output

def generate_para_sample(N, lat_var):
    S = lat_var.shape[0]
    sam = np.zeros((S, N, N))
    for s in range(S):
        lvar = np.zeros(9)
        ms = 0.4 - lat_var[s,1]
        mg = lat_var[s,5] * ms
        lvar[1] = 0.05 + mg
        lvar[2] = lvar[1] + lat_var[s,1]
        ms = 0.4 - lat_var[s,2]
        mg = lat_var[s,6] * ms
        lvar[3] = 0.55 + mg
        lvar[4] = lvar[3] + lat_var[s,2] 
        ms = 0.4 - lat_var[s,3]
        mg = lat_var[s,7] * ms
        lvar[5] = 0.55 + mg
        lvar[6] = lvar[5] + lat_var[s,3] 
        ms = 0.4 - lat_var[s,4]
        mg = lat_var[s,8] * ms
        lvar[7] = 0.55 + mg
        lvar[8] = lvar[7] + lat_var[s,4] 
        if lat_var[s,0] == 0 or lat_var[s,0] == 1 or lat_var[s,0] == 2:
            for i in range(int(0.5*N), N):
                fi = i / N
                for j in range(int(0.5*N)):
                    fj = j / N
                    if lvar[1] < fj < lvar[2] and lvar[3] < fi < lvar[4]:
                        sam[s, i, j] = 1
        if lat_var[s,0] == 0 or lat_var[s,0] == 1 or lat_var[s,0] == 3:
            for i in range(int(0.5*N), N):
                fi = i / N
                for j in range(int(0.5*N), N):
                    fj = j / N
                    if lvar[5] < fj < lvar[6] and lvar[7] < fi < lvar[8]:
                        sam[s, i, j] = 1
    return sam.reshape((sam.shape[0], 1, sam.shape[1], sam.shape[2]))

