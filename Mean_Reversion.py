# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:20:01 2025. Mean Reversion Stratergy.

@author: Archie McNamee
"""
from typing import Union,Generator
import numpy as np
import pandas as pd
from scipy import optimize as opt
from dataclasses import dataclass
import time
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

class Momentum:
    """Simple Mean Reversion Trading Algorithm.
    
    Both anti momentum and momentum stratergies.
    """
    
    def __init__(self,
                 initial_capital,
                 asset_ts_data):
        
        self.asset_price : Union[pd.Series,np.ndarray] = np.array(asset_ts_data)
        self.shape : tuple = (np.shape(self.asset_price))
        self.portfolio_value = np.zeros(self.shape)
        self.portfolio_value[0] = initial_capital
        
    @staticmethod
    def MA(data:np.ndarray
           ,window:int
           ) -> Generator: # this need to be changed as massively hinders time-complexity
        """
        MA of  a defined window.

        Parameters
        ----------
        data : np.ndarray
            Underlying Price Data.
        window : int
            Unit length of window e.g hours,days etc.

        Yields
        ------
        Generator
            Moving average value at time .

        """
        for m in range(len(data)-(window-1)):
            
            yield sum(np.ones(window) * data[m:m+window]) / window
     
    @staticmethod
    def daily_log_return(data):
        
        log_return = []
        for i in range(len(data)):
            
            log_return.append(
                
                np.log(data[i]/data[i-1])
                
                )
        return log_return
            
    def trading_anti(
            self,window:int,
            sensitivity:float,
            transaction_cost_constant:float,
            integrated_transaction: bool
            )-> list:
        """
                Mean Reversion trading stratergy.
        
        Using a given price sensitivity, buy or sell a given underlying
        asset with respect to a defined moving average price.
        
        If price falls below moving average, algorithm will allocate all cash 
        available to asset. If price moves above moving average algorithm will
        sell all shares in underlying asset.

        Parameters
        ----------
        window : int
            DESCRIPTION.
        sensitivity : float
            DESCRIPTION.
        transaction_cost_constant : float
            DESCRIPTION.
        integrated_transaction : bool
            if true transaction costs are removed from portfolio value as
            trades occur, if false are accumlated in a seperate account.

        Returns
        -------
        list
            Portfolio Value at the end of data set.
        list
            sum of transaction costs

        """
        phi =  np.zeros(shape=self.shape) # number of shares held
        omega = np.zeros(shape=self.shape) # proportion of portfolio value in cash 
        omega[0] = 1
        theta = np.zeros(shape=self.shape) # proportion of portfolio value in asset
        c = transaction_cost_constant
        transaction_cost = np.zeros(shape=self.shape)
        net_transaction_cost = 0
        
        i = 1
        
        while i < len(self.asset_price):
            
            self.portfolio_value[i] = (
                
                (phi[i-1]*self.asset_price[i]) + (omega[i-1] * self.portfolio_value[i-1] * (1+(np.log(1.04)/252)) )
                
                )
                
            if i < window:
                
                phi[i] = phi[i-1]
                    
                omega[i] = omega[i-1]
                    
                theta[i] = theta[i-1]
                
                i += 1
                
            else:
                
                if i <= 252:
                    
                    a = 0
                    
                if i > 252:
                    
                    a = i -252 
                
                ma = (list(self.MA(self.asset_price[a:i], window)))
                
                if (self.asset_price[i] > (1+sensitivity) * ma[-1]) and (phi[i-1] > 0):
                    
                    phi[i] = 0
                    
                    theta[i] = 0
                    
                    omega[i] = 1
                    
                    
                elif (self.asset_price[i] < (1-sensitivity)* ma[-1] ) and (omega[i-1] > 0):
                    
                    phi[i] = self.portfolio_value[i]/self.asset_price[i]
                    
                    theta[i] = 1
                    
                    omega[i] = 0
                    
                else:
                    
                    phi[i] = phi[i-1]
                    
                    omega[i] = omega[i-1]
                    
                    theta[i] = theta[i-1]
                    
                cost = min(abs(phi[i]-phi[i-1]) * self.asset_price[i] * c,1)
           
                transaction_cost[i] = cost 
            
                net_transaction_cost += cost
                
                if integrated_transaction == True and i > 1:
                
                    self.portfolio_value[i] = (self.portfolio_value[i]-transaction_cost[i])
                i += 1
                    
        self.phi_final = phi
        self.theta_final = theta
        self.omega_final = omega
        self.net_transaction_cost = net_transaction_cost
        
        
            
        return self.portfolio_value , self.daily_log_return(self.portfolio_value)
    
    
    def trading(
            self,window:int,
            sensitivity:float,
            transaction_cost_constant:float,
            integrated_transaction: bool
            )-> list:
        """
                Mean Reversion trading stratergy.
        
        Using a given price sensitivity, buy or sell a given underlying
        asset with respect to a defined moving average price.
        
        If price falls below moving average, algorithm will allocate all cash 
        available to asset. If price moves above moving average algorithm will
        sell all shares in underlying asset.

        Parameters
        ----------
        window : int
            DESCRIPTION.
        sensitivity : float
            DESCRIPTION.
        transaction_cost_constant : float
            DESCRIPTION.
        integrated_transaction : bool
            if true transaction costs are removed from portfolio value as
            trades occur, if false are accumlated in a seperate account.

        Returns
        -------
        list
            Portfolio Value at the end of data set.
        list
            sum of transaction costs

        """
        phi =  np.zeros(shape=self.shape) # number of shares held
        omega = np.zeros(shape=self.shape) # proportion of portfolio value in cash 
        omega[0] = 1
        theta = np.zeros(shape=self.shape) # proportion of portfolio value in asset
        c = transaction_cost_constant
        transaction_cost = np.zeros(shape=self.shape)
        net_transaction_cost = 0
        
        i = 1
        
        while i < len(self.asset_price):
            
            self.portfolio_value[i] = (
                
                (phi[i-1]*self.asset_price[i]) + (omega[i-1] * self.portfolio_value[i-1]  )
                
                )
                
            if i < window:
                
                phi[i] = phi[i-1]
                    
                omega[i] = omega[i-1]
                    
                theta[i] = theta[i-1]
                
                i += 1
                
            else:
                
                if i <= 252:
                    
                    a = 0
                    
                if i > 252:
                    
                    a = i -252 
                
                ma = (list(self.MA(self.asset_price[a:i], window)))
                
                if (self.asset_price[i] < (1+sensitivity) * ma[-1]) and (phi[i-1] > 0):
                    
                    phi[i] = 0
                    
                    theta[i] = 0
                    
                    omega[i] = 1
                    
                    
                elif (self.asset_price[i] >  (1-sensitivity)* ma[-1] ) and (omega[i-1] > 0):
                    
                    phi[i] = self.portfolio_value[i]/self.asset_price[i]
                    
                    theta[i] = 1
                    
                    omega[i] = 0
                    
                else:
                    
                    phi[i] = phi[i-1]
                    
                    omega[i] = omega[i-1]
                    
                    theta[i] = theta[i-1]
                    
                cost = min(abs(phi[i]-phi[i-1]) * self.asset_price[i] * c,1)
           
                transaction_cost[i] = cost 
            
                net_transaction_cost += cost
                
                if integrated_transaction == True and i > 1:
                
                    self.portfolio_value[i] = (self.portfolio_value[i]-transaction_cost[i])
                i += 1
                    
        self.phi_final = phi
        self.theta_final = theta
        self.omega_final = omega
        self.net_transaction_cost = net_transaction_cost
        
        
            
        return self.portfolio_value , self.daily_log_return(self.portfolio_value)   
             
    def optimise(self,pos:bool) -> float:
        """
        Optimiser of Trading algorithm above.
        
        Using different moving average windows and price sensitivities to find
        optimum portfolio value.

        Returns
        -------
        float
            Maximum possible return from asset price using Stratergy.

        """
        bounds = [(0.05,0.2),(3,25)]
        
        if pos == True:
            
            func = self.trading()
            
        else:
            
            func = self.trading_anti()
        
        def objective(params
                ) -> float:
            
            sensitivity , window = params
            
            window = int(round(window))
            
            final_value = (func(window=window,
                                       sensitivity = sensitivity,
                                       transaction_cost_constant = 0.0025,
                                       integrated_transaction=True)[0])[-1]
            return  -final_value
        
        result = opt.shgo(objective, bounds)
        
        print("Best price sensivitiy and MA window respectively:",result.x)
    
        return result.fun
    
class MovingAverageIntersection:
    """Execute trades on given short term and long term averages.
    
    If short-term MA rises or falls below long-term MA by a chosen sensitivity 
    we sell or buy respectively.
    """
    
    def __init__(self,
                 initial_capital,
                 asset_ts_data):
        import numpy as np
        self.asset_price : Union[pd.Series,np.ndarray] = np.array(asset_ts_data)
        self.shape : tuple = (np.shape(self.asset_price))
        self.portfolio_value = np.zeros(self.shape)
        self.portfolio_value[0] = initial_capital
        
    @staticmethod
    def MA(data:np.ndarray
           ,window:int
           ) -> Generator: # this need to be changed as it massively hinders time-complexity
        """
        MA of  a defined window.

        Parameters
        ----------
        data : np.ndarray
            Underlying Price Data.
        window : int
            Unit length of window e.g hours,days etc.

        Yields
        ------
        Generator
            Moving average value at time .

        """
        for m in range(len(data)-(window-1)):
            
            yield sum(np.ones(window) * data[m:m+window]) / window #can be replaced by np.convole
     
    @staticmethod
    def daily_log_return(data: np.ndarray) -> list: 
        """
        Calculate daily log return of data set.

        Parameters
        ----------
        data : np.ndarray
            Price data.

        Returns
        -------
        list
            Daily log return.

        """
        log_return = []
        
        for i in range(len(data)):
            
            log_return.append(
                
                    np.log(data[i]/data[i-1])
                
                    )
            
        return log_return 
    
    def trading(
            self,short_term_window:int,
            long_term_window:int,
            max_hold_cash:int,
            sensitiviy:float,
            transaction_cost_constant:float,
            integrated_transaction:bool
            )-> np.ndarray:
        """
        Excutes trades.

        Parameters
        ----------
        short_term_window : int
            length of short term MA.
        long_term_window : int
            length of long term MA.
        max_hold_cash : int
            maximum amount of time to hold cash position.
        sensitiviy : float
            how much price needs to move to intiate trade.
        transaction_cost_constant : float
            proportion of trade volume.
        integrated_transaction : bool
            subtract fee from value.

        Returns
        -------
        np.ndarray
            portfolio value time series.

        """
        phi =  np.zeros(shape=self.shape) # number of shares held
        omega = np.zeros(shape=self.shape) # proportion of portfolio value in cash 
        omega[0] = 1
        theta = np.zeros(shape=self.shape) # proportion of portfolio value in asset
        c = transaction_cost_constant
        transaction_cost = np.zeros(shape=self.shape)
        net_transaction_cost = 0
        
        i = 1
        
        while i < len(self.asset_price):
            
            self.portfolio_value[i] = (
                
                phi[i-1]*self.asset_price[i] + omega[i-1]*self.portfolio_value[i-1]
        
                )
            
            if i < long_term_window + 1:
                
                phi[i] = phi[i-1]
                
                omega[i] = omega[i-1]
                
                theta[i] = theta[i-1]
                
                i +=1
                
            else:
                
                if i <= 252:
                    
                    a = 0
                    
                elif i > 252:
                    
                    a = i -252 
                    
                short_ma = (list(self.MA(self.asset_price[a:i], short_term_window)))
                 
                long_ma  = (list(self.MA(self.asset_price[a:i], long_term_window)))
                
                if (((short_ma[-1]*(1+sensitiviy) > long_ma[-1])) and (omega[i-1]>0)) and (short_ma[-2] < long_ma[-2]):
                    
                    phi[i] = (
                        
                        self.portfolio_value[i]/self.asset_price[i]
                
                        )
                    
                    theta[i] = 1
                    
                    omega[i] = 0
                    
                elif (((short_ma[-1]*(1-sensitiviy) < long_ma[-1])) and (theta[i-1]>0)) and (short_ma[-2] > long_ma[-2]):
                    
                    phi[i] = 0
                    
                    theta[i] = 0
                    
                    omega[i] = 1
                    
                elif sum(omega[i-max_hold_cash:i]) == len(omega[i-max_hold_cash:i]):
                    
                    phi[i] = (
                        
                        self.portfolio_value[i]/self.asset_price[i]
                        
                        )
                    
                    theta[i] = 1
                    
                    omega[i] = 0
                    
                else:
                    
                    phi[i] = phi[i-1]
                    
                    theta[i] = theta[i-1]
                    
                    omega[i] = omega[i-1]
                    
                cost = min(
                    
                    abs(phi[i]-phi[i-1]) * self.asset_price[i] * c, 1
                    
                    )
           
                transaction_cost[i] = cost 
            
                net_transaction_cost += cost
                
                if integrated_transaction == True and i > 1:
                
                    self.portfolio_value[i] = (
                        
                        self.portfolio_value[i]-transaction_cost[i]
                        
                        )
                i += 1
        
        self.phi_final = phi
        self.theta_final = theta
        self.omega_final = omega
        self.net_transaction_cost = net_transaction_cost
        
        return self.portfolio_value , self.daily_log_return(self.portfolio_value)

    def optimise(self):
        
        space = [
            Integer(5, 22, name="short_window"),
            Integer(66, 245, name="long_window"),
            Integer(22,66, name="max_hold"),
            Real(0.0001,0.15,name="sens")]
        
        def objective(params) -> float:
            
            short_window, long_window, max_hold ,sens = params

       
            result = self.trading(
            short_term_window=short_window,
            long_term_window=long_window,
            max_hold_cash=max_hold,
            sensitiviy=sens,  
            transaction_cost_constant=0.0025,
            integrated_transaction=True)[0]

            best = result[-1]

            return -best
        
        result =  gp_minimize(func=objective,
                              dimensions=space,
                              acq_func="EI",       
                              n_calls=25,
                              n_random_starts=5,
                              random_state=0
                              )
         
        plot_convergence(result)
        plt.show()
        
        return result.x, -result.fun 
                
class Utilisation:
    """Optimise."""
    
    def __init__(self,data:pd.Series,range_in:list,range_out:list):
        
        self.blackbox = MovingAverageIntersection2(1, data[min(range_in):max(range_in)])
        self.out = MovingAverageIntersection2(1, data[min(range_out):max(range_out)])
        self.data = data
        self.range_in = range_in
        self.range_out = range_out
        
    def run(self):
        """
        Fit's parameters on sample data then applies to out of sample data.

        Returns
        -------
        plots
            A converge plot,plot showing trading times,plot showing trading strat 
            vs BAH stratergy.

        """  
        range_in = self.range_in
        range_out = self.range_out
        data = self.data
        short_window, long_window, max_hold ,sens = self.blackbox.optimise()[0] 
        print(short_window, long_window, max_hold ,sens)
           
        implement = (self.out).trading(
            short_term_window=short_window,
            long_term_window=long_window,
            max_hold_cash=max_hold,
            sensitivity=sens,
            transaction_cost_constant=0.0025,
            integrated_transaction=True)[0]
           
        plt.figure()
        plt.title("Out of sample test omega vs theta")
        plt.plot(
            
            data.index[min(range_out):max(range_out)],
            self.out.omega_final,
            label = "Phi"
            )
        plt.plot(
            
            data.index[min(range_out):max(range_out)],
            self.out.theta_final,
            label = "theta"
            )
        plt.show()
        
        plt.figure()
        plt.title("Out of sample test")
        plt.plot(
               
               data.index[min(range_out):max(range_out)],
               data[min(range_out):max(range_out)]/data[min(range_out)],
               label = "BAH"
               
               )
           
        plt.plot(
               data.index[min(range_out):max(range_out)],
               implement,
               label = ("Stratergy applied out of sample")
               
               )
        
        
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(
               data.index[min(range_out):max(range_out)],
               implement-data[min(range_out):max(range_out)]/data[min(range_out)],
               label = ("Stratergy applied out of sample")
               
               )
        plt.show()

        return 0
    
class Utilisation2:
    """Optimise."""
    
    def __init__(self,data:pd.Series,range_in:list,range_out:list):
        
        self.blackbox = MovingAverageIntersection2(1, data[min(range_in):max(range_in)])
        self.out = MovingAverageIntersection2(1, data[min(range_out):max(range_out)])
        self.data = data
        self.range_in = range_in
        self.range_out = range_out
        
    def run(self):
        """
        Fit's parameters on sample data then applies to out of sample data.

        Returns
        -------
        plots
            A converge plot,plot showing trading times,plot showing trading strat 
            vs BAH stratergy.

        """            
        range_in = self.range_in
        range_out = self.range_out
        data = self.data
        short_window, long_window, max_hold ,sens, sens_down = self.blackbox.optimise()[0] 
        print(short_window, long_window, max_hold ,sens, sens_down)
           
        implement = (self.out).trading(
            short_term_window=short_window,
            long_term_window=long_window,
            max_hold_cash=max_hold,
            sensitivity=sens,
            sensitivity_down = sens_down,# still keeping typo if needed
            transaction_cost_constant=0.0025,
            integrated_transaction=True)[0]
           
        plt.figure()
        plt.title("Out of sample test omega vs theta")
        plt.plot(
            
            data.index[min(range_out):max(range_out)],
            self.out.omega_final,
            label = "Phi"
            )
        plt.plot(
            
            data.index[min(range_out):max(range_out)],
            self.out.theta_final,
            label = "theta"
            )
        plt.show()
        
        plt.figure()
        plt.title("Out of sample test")
        plt.plot(
               
               data.index[min(range_out):max(range_out)],
               data[min(range_out):max(range_out)]/data[min(range_out)],
               label = "BAH"
               
               )
           
        plt.plot(
               data.index[min(range_out):max(range_out)],
               implement,
               label = ("Stratergy applied out of sample")
               
               )
        
        
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(
               data.index[min(range_out):max(range_out)],
               implement-data[min(range_out):max(range_out)]/data[min(range_out)],
               label = ("Stratergy applied out of sample")
               
               )
        plt.show()

        return 0
        
        
                
class MovingAverageIntersection2:
    """Execute trades on given short term and long term averages.
    
    If short-term MA rises or falls below long-term MA by a chosen sensitivity 
    we sell or buy respectively.
    """
    
    def __init__(self,
                 initial_capital,
                 asset_ts_data):
        import numpy as np
        self.asset_price : Union[pd.Series,np.ndarray] = np.array(asset_ts_data)
        self.shape : tuple = (np.shape(self.asset_price))
        self.portfolio_value = np.zeros(self.shape)
        self.portfolio_value[0] = initial_capital
        
    @staticmethod
    def MA(data:np.ndarray
           ,window:int
           ) -> Generator: # this need to be changed as it massively hinders time-complexity
        """
        MA of  a defined window.

        Parameters
        ----------
        data : np.ndarray
            Underlying Price Data.
        window : int
            Unit length of window e.g hours,days etc.

        Yields
        ------
        Generator
            Moving average value at time .

        """
        for m in range(len(data)-(window-1)):
            
            yield sum(np.ones(window) * data[m:m+window]) / window #can be replaced by np.convole
     
    @staticmethod
    def daily_log_return(data: np.ndarray) -> list: 
        """
        Calculate daily log return of data set.

        Parameters
        ----------
        data : np.ndarray
            Price data.

        Returns
        -------
        list
            Daily log return.

        """
        log_return = []
        
        for i in range(len(data)):
            
            log_return.append(
                
                    np.log(data[i]/data[i-1])
                
                    )
            
        return log_return 
    
    def trading(
            self,short_term_window:int,
            long_term_window:int,
            max_hold_cash:int,
            sensitivity:float,
            sensitivity_down:float,
            transaction_cost_constant:float,
            integrated_transaction:bool
            )-> np.ndarray:
        
        phi =  np.zeros(shape=self.shape) # number of shares held
        omega = np.zeros(shape=self.shape) # proportion of portfolio value in cash 
        omega[0] = 1
        theta = np.zeros(shape=self.shape) # proportion of portfolio value in asset
        c = transaction_cost_constant
        transaction_cost = np.zeros(shape=self.shape)
        net_transaction_cost = 0
        """
        Excutes trades.

        Parameters
        ----------
        short_term_window : int
            length of short term MA.
        long_term_window : int
            length of long term MA.
        max_hold_cash : int
            maximum amount of time to hold cash position.
        sensitiviy : float
            how much price needs to move to intiate purchase.
        senstivity_down : float
            how much price needs to move to intiate sale.
        transaction_cost_constant : float
            proportion of trade volume.
        integrated_transaction : bool
            subtract fee from value.

        Returns
        -------
        np.ndarray
            portfolio value time series.

        """       
        i = 1
        
        while i < len(self.asset_price):
            
            self.portfolio_value[i] = (
                
                phi[i-1]*self.asset_price[i] + omega[i-1]*self.portfolio_value[i-1]
        
                )
            
            if i < long_term_window + 1:
                
                phi[i] = phi[i-1]
                
                omega[i] = omega[i-1]
                
                theta[i] = theta[i-1]
                
                i +=1
                
            else:
                
                if i <= 252:
                    
                    a = 0
                    
                elif i > 252:
                    
                    a = i -252 
                    
                short_ma = (list(self.MA(self.asset_price[a:i], short_term_window)))
                 
                long_ma  = (list(self.MA(self.asset_price[a:i], long_term_window)))
                
                if (((short_ma[-1]*(1+sensitivity) > long_ma[-1])) and (omega[i-1]>0)) and (short_ma[-2] < long_ma[-2]):
                    
                    phi[i] = (
                        
                        self.portfolio_value[i]/self.asset_price[i]
                
                        )
                    
                    theta[i] = 1
                    
                    omega[i] = 0
                    
                elif (((short_ma[-1]*(1-sensitivity_down) < long_ma[-1])) and (theta[i-1]>0)) and (short_ma[-2] > long_ma[-2]):
                    
                    phi[i] = 0
                    
                    theta[i] = 0
                    
                    omega[i] = 1
                    
                elif sum(omega[i-max_hold_cash:i]) == len(omega[i-max_hold_cash:i]):
                    
                    phi[i] = (
                        
                        self.portfolio_value[i]/self.asset_price[i]
                        
                        )
                    
                    theta[i] = 1
                    
                    omega[i] = 0
                    
                else:
                    
                    phi[i] = phi[i-1]
                    
                    theta[i] = theta[i-1]
                    
                    omega[i] = omega[i-1]
                    
                cost = min(
                    
                    abs(phi[i]-phi[i-1]) * self.asset_price[i] * c, 1
                    
                    )
           
                transaction_cost[i] = cost 
            
                net_transaction_cost += cost
                
                if integrated_transaction == True and i > 1:
                
                    self.portfolio_value[i] = (
                        
                        self.portfolio_value[i]-transaction_cost[i]
                        
                        )
                i += 1
        
        self.phi_final = phi
        self.theta_final = theta
        self.omega_final = omega
        self.net_transaction_cost = net_transaction_cost
        
        return self.portfolio_value , self.daily_log_return(self.portfolio_value)

    def optimise(self):
        """
        Optimiser of Trading algorithm above.
        
        Using different moving average windows and price sensitivities to find
        optimum portfolio value.

        Returns
        -------
        float
            Maximum possible return from asset price using Stratergy.

        """          
        space = [
            Integer(5, 22, name="short_window"),
            Integer(66, 245, name="long_window"),
            Integer(22,66, name="max_hold"),
            Real(0.0001,0.15,name="sens"),
            Real(0.0001,0.15,name="sens_down")
            ]
        
        def objective(params) -> float:
            
            short_window, long_window, max_hold ,sens, sens_down = params

       
            result = self.trading(
            short_term_window=short_window,
            long_term_window=long_window,
            max_hold_cash=max_hold,
            sensitivity=sens,
            sensitivity_down = sens_down,# still keeping typo if needed
            transaction_cost_constant=0.0025,
            integrated_transaction=True)[0]

            best = result[-1]

            return -best
        
        result =  gp_minimize(func=objective,
                              dimensions=space,
                              acq_func="EI",        # Expected Improvement
                              n_calls=25,
                              n_random_starts=5,
                              random_state=0
                              )
         
        plot_convergence(result)
        plt.show()
        
        return result.x, -result.fun             