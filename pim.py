import matplotlib.pyplot as plt
import numpy as np
import os
from poisson1d import solve_poisson as sp

class onedsim(object):
    def __init__(self,
                 ## particle parameters ##
                 q1 = -1,
                 q2 = -1,
                 m1 = 1,
                 m2 = 1,
                 ## stream parameters ##
                 v1 = 4,
                 v2 = -4,
                 Np = 256,
                 ## simulation parameters ##
                 L = 2 * np.pi,
                 k = 1,
                 Ng = 64,
                 perturb = True):
    
        self.q1 = q1
        self.q2 = q2
        self.m1 = m1
        self.m2 = m2
        
        self.init_v1 = v1
        self.init_v2 = v2
        self.Np = Np
    
        self.L = L
        self.k = k
        self.Ng = Ng
        self.dt = L / Ng / v1
        self.dx = L / Ng
        self.clock = 0
        
        self.grid_edges = np.zeros(self.Ng)
        self.grid_centers = get_centers(self.grid_edges)
        self.phi = np.zeros_like(self.grid_centers) # an array to fill with the potential
        self.field = np.zeros_like(self.grid_centers) # an array to fill with minus grad(phi)
        self.perturb = perturb
        
        self.average_rho = self.get_average_rho()
        
        self.x = np.empty((2, Np))
        self.x_d = np.empty((2, Np))
        
        self.initialize_particles()
        self.initialize_grid()
        
        self.rho = 999
        self.rho1 = 999
        self.rho2 = 999
    
    def initialize_grid(self):
        for i in range(self.Ng):
            self.grid_edges[i] = i * self.dx
        return(0)
    
    def initialize_particles(self):
        A = 0.01 * self.L
        stream1_x = np.linspace(0, self.L, self.Np)
        stream2_x = np.linspace(0, self.L, self.Np)
        
        if self.perturb:
            pert = A * np.sin(stream1_x * self.k / self.L)
            stream1_x += pert
            stream2_x += pert
        
        stream1_xd = np.zeros(self.Np) + self.init_v1
        stream2_xd = np.zeros(self.Np) + self.init_v2
        
        self.x[0] = stream1_x
        self.x[1] = stream2_x
        
        self.x_d[0] = stream1_xd
        self.x_d[1] = stream2_xd
        return(0)
    
    """
    Leapfrog the particles to get the next timestep.
    
    Returns
    -------------------
    x_new, x_d_new: The timestepped state vectors
    """
    def leapfrog(self, phi):
        x_half = self.x + ((1/2) * self.x_d * self.dt)
        
        x_dd = self.get_acc(phi)
        
        x_d_new = self.x_d + (x_dd * self.dt)
        x_new = (x_half + ((1/2) * x_d_new * self.dt))%self.L
        
        #print(self.x_d, x_d_new)
        
        return(x_new, x_d_new)
    
    def get_acc(self, phi):
        
        field = - self.CFD(phi)

        x_dd = field[np.digitize(self.x, self.grid_edges)%(self.Ng)]
        
        x_dd[0] /= self.m1
        x_dd[1] /= self.m2
        return(x_dd)
    
    def get_average_rho(self):
        rho1 = self.q1 * self.Np / self.L
        rho2 = self.q2 * self.Np / self.L
        return(rho1 + rho2)
    
    """
    A second-order centered finite difference to calculate the field
    """
    def CFD(self, potential):
        phi_plus = np.zeros_like(potential)
        phi_minus = np.zeros_like(potential)
        
        phi_plus[:self.Ng - 1] = potential[1:]
        phi_plus[-1] = potential[0]

        phi_minus[1:] = potential[:self.Ng - 1]
        phi_minus[0] = potential[-1]

        phi_p = (phi_plus - phi_minus) / (2 * self.dx)
        return(phi_p)
     
     
    def scott_CIC(self, pos, m):
        rho = np.zeros_like(self.grid_edges)
        for i in range(self.Ng):
            for j in range(self.Np):
                x = self.grid_edges[i] - pos[j]
                if (-self.dx) <= x and (x < 0):
                    rho[i] += self.q1 * (1 + (x / self.dx))
                    if(i==self.Ng-1):
                        xb = self.L - pos[j]
                        rho[0] += self.q1 * (1 - (xb / self.dx))
                elif (x < self.dx) and (x >= 0):
                    rho[i] += self.q1 * (1 - (x/self.dx))
        
        rho /= self.dx
        rho -= self.average_rho
        rho[0] = rho[-1]
        return(rho)
        
    """
    A cloud-in-cell method to get rho from the particles
    """   
    def CIC(self, pos, m):
        rho = np.zeros_like(self.grid_edges)
        for i in range(self.Ng):
            #print(f"for grid cell {i}:")
            sum = 0
            for j in range(self.Np):
                jj = self.W_CIC(self.grid_edges[i] - pos[j])
                #print(f"\t particle {j} has a contribution of {jj}")
                sum += jj
                
            rho[i] = (m / self.dx) * sum
        rho[0] = rho[-1]
        #rho[1] = rho[-1]
        self.rho -= self.average_rho
        return(rho) 
    
    """
    the weighting function for the CIC method
    """
    def W_CIC(self, sep):
        
        retval = 0
    
        if (-self.dx) < sep and np.abs(sep) < self.dx:
            retval = 1 + (sep / self.dx)
            
        elif sep < self.dx and np.abs(sep) < self.dx:
            retval = 1 - (sep / self.dx)

        return(retval)
    
    def get_potential(self, phi, rho):
        U = rho * phi
        return(U)
    
    def get_kinetic(self, vel, m):
        K = (1/2) * m * (vel**2)
        return(K)
    
    def timestep(self, nsteps):
        history = np.zeros((nsteps, 4, self.Np))
        iinit = 1
        for step in range(0, nsteps):            
            history[step][0] = self.x[0]
            history[step][1] = self.x[1]
            history[step][2] = self.x_d[0]
            history[step][3] = self.x_d[1]

            rho1 = self.CIC(self.x[0], self.m1)
            rho2 = self.CIC(self.x[1], self.m2)
            self.rho1 = rho1
            self.rho2 = rho2
            self.rho = rho1 + rho2
            phi = sp(self.rho, self.dx, iinit)
            self.phi = phi
            iinit = 0
        
            self.x, self.x_d = self.leapfrog(phi)
        
            self.clock += self.dt

        
        return(history)
    
    
   
        
"""
arr should always be the output of np.linspace
"""
def get_centers(arr):
    diff = np.diff(arr)
    
    if np.abs(np.max(diff) - np.min(diff)) > 1e-10:
        raise ValueError('bad input to get_centers')
        
    return((arr + (diff[0]/2))[:-1])


def make_movie(data, dir, ylim = (-20,20)):
    for i in range(len(data)):
        plt.cla()
        x0 = data[i][0]
        x1 = data[i][1]
        v0 = data[i][2]
        v1 = data[i][3]
        plt.scatter(x0, v0)
        plt.scatter(x1, v1)
        plt.xlim(0,2*np.pi)
        ymin, ymax = ylim
        plt.ylim(ymin, ymax)
        plt.savefig(f'{dir}/frame_{i}.png')
    os.system(f'ffmpeg -framerate 60 -i {dir}/frame_%d.png {dir}/out.mp4')
        

if __name__ == "__main__":
    sim1 = onedsim()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    