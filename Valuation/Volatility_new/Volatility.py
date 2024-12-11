import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model

class GARCH:

    def __init__(self, Ticker:str = 'KRW=X', start_date:str = '2022-08-30', end_date:str = '2024-08-30'):
        self.garch_model = None
        self.start_date = start_date
        self.end_date = end_date
        self.Ticker = Ticker
        self.df = self.get_raw_data()
        self.log_df = self.get_log_return()

    def get_raw_data(self):
        return yf.download(self.Ticker, start = self.start_date, end = self.end_date)
    
    def get_log_return(self):
        df = pd.DataFrame(columns = ['Date', 'Log_return'])
        df['Log_return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        df = df[1:]
        df['Date'] = self.df.index[1:]
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df.reset_index(drop = True, inplace = True)
        return df
    
    def plot_log_df(self):
        plt.figure(figsize = (100, 10))
        plt.plot(self.log_df.index, self.log_df['Log_return'], 'g--', label =f'Log_return of {self.Ticker}')
        plt.grid(True)
        plt.ylabel('Log_return')
        plt.xlabel('Index(one day)')
        plt.legend(loc = 'best')
        plt.show()
        return True

    def check_acf(self):
        sm.graphics.tsa.plot_acf(self.log_df)
        plt.show()
        return True

    def check_pacf(self):
        sm.graphics.tsa.plot_pacf(self.log_df)
        plt.show()
        return True

    #Fitting GARCH Model
    #Future Revision: Find which p,q is optimal for the model.
    def fit_garch_model(self, p = 1, q = 1):
        self.garch_model = arch_model(self.log_df['Log_return'], p = p, q = q)
        self.garch_model = self.garch_model.fit(update_freq = 10)
        return self.garch_model
    
    def get_alpha1(self):
        alpha1 = self.garch_model.params['alpha[1]']
        return alpha1
    
    def get_beta1(self):
        beta1 = self.garch_model.params['beta[1]']
        return beta1
    
    def get_omega(self):
        omega = self.garch_model.params['omega']
        if omega == None:
            return 1e-15
        else:
            return omega
        

