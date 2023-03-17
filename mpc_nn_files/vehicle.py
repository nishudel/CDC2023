import numpy as np
from scipy.integrate import odeint

class Vehicle:
    
    def __init__(self, p0, v0, bounds, T, Ts):
        self.p = p0; self.v = v0
        self.state = np.array([p0, v0])
        self.u = 0.0
        self.T = T; self.Ts = Ts
        self.X_hist = np.empty([2,0], dtype=np.ndarray)
        self.U_hist = np.empty([0], dtype=float)
        self.T_hist = np.empty([0], dtype=float)
        self.X_hist = np.hstack([self.X_hist, self.state.reshape([-1,1])])
        self.v_min = bounds['v_min'] + 1e-2; self.v_max = bounds['v_max'] - 1e-2
        self.u_min = bounds['u_min']; self.u_max = bounds['u_max']
        self.CFM_params = np.array([0.8, 1.2, 2.0, 20.0, 5.0])
        
    def double_integrator(self, z, t, u):
        v = z[1]
        dz = np.zeros(2)
        dz[0] = v; dz[1] = u
        return dz
    
    def run(self, u, time_stamp):    
        u = max(self.u_min, min(self.u_max, u)) # Control input should be saturated
        u = max(u, (self.v_min - self.v)/self.Ts)
        # u = min(u, (self.v_max - self.v)/self.Ts)
            
        tspan = np.linspace(0, self.Ts, 3)
        y = odeint(self.double_integrator, self.state, tspan, args=(u,)) 
        self.state = y[-1,]
        self.p, self.v = self.state
        self.u = u

    def save_data(self, time_stamp):
        self.X_hist = np.hstack([self.X_hist, self.state.reshape([-1,1])])
        self.U_hist = np.hstack([self.U_hist, self.u])  
        self.T_hist = np.hstack([self.T_hist, time_stamp])  

    def print(self, i):
        print("Some information for vehicle", i, ":", self.p, self.v)
    
    def const_U(self, U, t):
        time = np.arange(t, t+self.T, self.Ts)
        for k in time:
            self.run(U, k)  
        self.save_data(t+self.T)
        
    def const_V(self, V, t):
        time = np.arange(t, t+self.T, self.Ts)
        for k in time:
            U = (V-self.v)/self.Ts
            self.run(U, k)  
        self.save_data(t+self.T)   


class CAV(Vehicle):
    
    def __init__(self, p0, v0, bounds, T, Ts):
        super().__init__(p0, v0, bounds, T, Ts)
        self.type = "CAV" # of course


class HDV(Vehicle):
    
    def __init__(self, p0, v0, bounds, T, Ts):
        super().__init__(p0, v0, bounds, T, Ts)
        self.type = "HDV" # of course

    def IRL(self, CAV_state, Q):
        p = self.p; v = self.v

        u = IRL_CFM(p, v, CAV_state[0], CAV_state[1], 0.0, Q, self.T, self.v_max, 0.0)
            
        u = max(self.u_min, min(self.u_max, u))
        if v + self.T*u < self.v_min:
            u = (self.v_min-v)/self.T
#         elif v + self.T*u > self.v_max:
#             u = (self.v_max-v)/self.T
        return u
    
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

## This class is to compute the action of the human driver by IRL model
def IRL_CFM(p2_0, v2_0, p1_0, v1_0, u1_0, W, T = .2, v_max = 15., r = 0.):
        # Note that p2_0 and p1_0 should be the distance to the conflict point
        C = np.zeros(7)
        C[0] = W[0] + W[1]*T**2
        C[1] = 2*W[1]*(v2_0-v_max)*T
        C[2] = W[1]*(v2_0-v_max)**2
        C[3] = W[2]
        C[4] = (0.5*T**2)**2
        C[5] = 2*0.5*T**2*(p2_0+T*v2_0)
        C[6] = ((p1_0+v1_0*T)**2+(p2_0+v2_0*T)**2-r**2)  
        
        coeff = [(2*C[0]*C[4]), (C[1]*C[4]+2*C[0]*C[4]), (C[1]*C[5]+2*C[0]*C[6]-2*C[3]*C[4]), \
                  (C[1]*C[6]-C[3]*C[5])]
        root = np.roots(coeff)
        sol = np.real(root[np.isreal(root)])[0]
        return sol    