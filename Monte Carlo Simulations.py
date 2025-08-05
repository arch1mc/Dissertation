# -*- coding: utf-8 -*-
"""Monte-carlo Simulation."""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from dataclasses import dataclass
from skopt import gp_minimize, forest_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import pandas as pd

@dataclass
class heston:
    
    v0 : float #intial vol
    x0 : float #intial price
    r : float # risk free rate
    rho : float # correlation between Wiener processes
    sigma: float # vol of the vol
    kappa: float #mean reversion
    theta: float # long run vol
    iterations: int
    dt:float #time step
    
        
    @staticmethod
    def v_next(v_previous: float, kappa: float, dt: float, theta: float, sigma: float) -> float:
        d = 4 * kappa * theta / (sigma ** 2)
        c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
        lambda_ = (4 * kappa * np.exp(-kappa * dt) * v_previous) / (sigma ** 2 * (1 - np.exp(-kappa * dt)))
        return min(c * ss.ncx2.rvs(d, lambda_),1)
    
    @staticmethod       
    def gen_Ic(v1:float,v2:float,
               dt:float) -> float: #only valid for dt << 1
                
        return (v1+v2)/2 * dt
    
    @staticmethod     
    def x_next(x_previous:float,r:float,
                       dt:float,Ic:float,rho:float,
                       sigma:float,v2:float,v1:float,
                       theta:float,kappa:float) -> float:
                
        loc = r*dt - 0.5*Ic + (rho/sigma) * (
                    
            v2-v1 - kappa*theta*dt + sigma*Ic ) 
                
        scale = (1-rho**2) * Ic            
            
        log_return = ss.norm.rvs(loc = loc , scale = scale)
        
        return x_previous * np.exp(log_return)
    
    def simulate_paths(self) -> list:
        
        X = [self.x0]
        V = [self.v0]
        
        nu = 4*self.kappa*self.theta / (self.sigma**2) #degrees of freedom
        
        i = 1 
        
        while i < self.iterations:
            
            V.append( 
                
                self.v_next(V[-1],self.kappa,self.dt,nu,self.sigma)
            
            )
            
            Ic = self.gen_Ic(V[-2],V[-1],self.dt)
            
            X.append(
                
                self.x_next(X[-1], self.r, self.dt,
                            Ic, self.rho, self.sigma, V[-1], V[-2], 
                            self.theta, self.kappa)
                
                )
            
            i += 1
            
        return X,V
    
    @staticmethod
    def log_return(data:list) -> list:
        
        a = []
        
        for i in range(len(data)-1):
            
            a.append(
                
                np.log(
                    
                    data[i+1]/data[i]
                    
                    )
                
                )
        
        return np.array(a)

    def calibration(self,data:list) -> list:
        
        data = np.array(data)
        
        space = [
            Real(0.01, 0.2, name="v0"),
            Real(-1,1, name = "rho"),
            Real(0.01,1, name = "sigma"),
            Real(1,10, name = "kappa"),
            Real(0.05,0.3, name = "theta")
            ]
        
        self.dt = 1/len(data)
        
        self.iterations = len(data)
        
        def objective(params:list)->float:
            
            self.x0 = data[0]
            
            self.v0 , self.rho , self.sigma , self.kappa , self.theta = params
            
            error = 0
            
            for i in range(100):
            
                cost = np.array(data - np.array(self.simulate_paths()[0]))
                
                #if np.isinf(cost) or np.isnan(cost):
                
                 #   cost = 1e6 
                    
                error += np.mean(cost**2)
            
            return error/100
        
        result = forest_minimize(
                    func=objective,
                    dimensions=space,
                    n_calls=50,
                    n_random_starts=15,
                    random_state=0,
                    verbose=True
                )
        
        plot_convergence(result)
        plt.ylim(25,250) 
        plt.show()
        
        return result.x, result.fun
    
    def calibration_2(self,data:list) -> list:
        
        data = np.array(data)
        
        space = [
            Real(0.1, 0.2, name="v0"),
            Real(-0.8,-0.4, name = "rho"),
            Real(0.3,0.9, name = "sigma"),
            Real(0.5,1.5, name = "kappa"),
            Real(0.09,0.16, name = "theta")
            ]
        
        self.dt = 1/len(data)
        
        self.iterations = len(data)   
        
        def objective(params:list)-> float:
            
            self.x0 = data[0]
            
            self.v0 , self.rho , self.sigma , self.kappa , self.theta = params
            
            mean = 0
            
            for i in range(5):
            
                a = np.array(self.simulate_paths()[0])
                
                arb = self.log_return(a) - self.log_return(data)
            
                mean += sum(abs(arb))
            
            return mean/10
        
        result = forest_minimize(
                    func=objective,
                    dimensions=space,
                    n_calls=100,
                    n_random_starts=35,
                    random_state=0,
                    verbose=True)
        
        #plot_convergence(result)
        #plt.ylim(0,12.5) 
        #plt.show()
        
        return result.x, result.fun
        
    
if __name__ == "":
    
    df = pd.read_csv("stress_test_data0.csv")
    
    data = df['close_APA'][:252]
    
    a = heston(0,0, 0.05, 0, 0, 0, 0, 0, 0) # random 
    
    params = a.calibration_2(data)[0]
    
    data_new = df['close_APA'].iloc[-252:].tolist()
    
    fit = heston(params[0]
                 ,data_new[0],
                 0.05,
                 params[1],
                 params[2],
                 params[3],
                 params[4],
                 len(data),
                 1/len(data))
    
    plt.title("New data vs Heston fit simulation")
    plt.plot(data_new, color = "red", label = "True data")
    plt.grid()
    for i in range(10):
        
        plt.plot(fit.simulate_paths()[0], color = "grey", alpha = 0.3)
        
    plt.show()
    
    plt.title("Volatility simulation")
    plt.grid()
    for i in range(10):
        
        plt.plot(fit.simulate_paths()[1])
        
    plt.show()

class MonteCarlo_OptionPricing:
    
    def __init__(self,data:list):
        

        self.data  = np.array(data)
        
        self.params = heston(0,0, 0.05, 0, 0, 0, 0, 0, 0).calibration_2(data)[0]
        
        self.fit = heston(params[0]
                 ,self.data[-1], # this start as real data ends 
                 0.05,
                 params[1],
                 params[2],
                 params[3],
                 params[4],
                 len(data),
                 1/252)
        
    def vanilla(self,strike:float,
                iterations:int,call:bool
                ) -> float:
        
        def call(X,K):
            
            return max(X-K,0)
        
        def put(X,K):
            
            return max(K-X,0)
        
        if call == True:
            
            i = 1 
            
            payoff = []
            
            while i < iterations:
                
                payoff.append( call(self.fit.simulate_paths()[0][-1], strike))
                
                i += 1
                
        else:
            
            i = 1 
            
            payoff = []
            
            while i < iterations:
                
                payoff.append( put(self.fit.simulate_paths()[0][-1], strike))
                
                i += 1
         
        payoff = np.array(payoff)
        
        V = np.exp(-0.05) * np.mean(payoff)
         
        plt.title(f"Distribution of payoff with strike = {strike} ")
        plt.hist(payoff,density = True,colour = "red")
        plt.axvline(V,0,1)
        plt.show()
                
            
        
        
        
        
    
    
    
    