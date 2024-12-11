import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from arch import arch_model
import datetime
import sys
#r_d, r_f를 risk_free 파일에서 가져가다 r_d - r_f를 하고 로그 수익률로 변환해서 GARCH모형 fit.
sys.path.append("/Users/choejihun/Desktop/FIRM_2024/Risk_free")
import risk_free

class GARCH:

    def __init__(self, Ticker = 'KRW=X'):
        self.garch_model = None
        self.df = yf.download(Ticker, start = '2022-08-30', end = '2024-08-30')
        self.log_df = None
        self.domestic_rf, self.foreign_rf = risk_free.get_risk_free_df()
        self.rate_diff = self.domestic_rf.sub(self.foreign_rf)
        self.rate_diff['Date'] = self.domestic_rf['Date']

    #Unused
    def show_close(self, df):
        plt.figure(figsize = (100, 10))
        plt.plot(df.index, df['Close'], 'g--', label ='Price')
        plt.grid(True)
        plt.ylabel('Change')
        plt.legend(loc = 'best')
        plt.show()
        return True


    # Unused
    def get_log_return(self):
        self.log_df = pd.DataFrame(columns = ['Date', 'Log_return'])
        self.log_df['Log_return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.log_df = self.log_df[1:]
        self.log_df['Date'] = self.df.index[1:]
        self.log_df['Date'] = pd.to_datetime(self.log_df['Date']).dt.strftime('%Y-%m-%d')
        self.log_df.reset_index(drop = True, inplace = True)
        return self.log_df

    def check_acf(self):
        sm.graphics.tsa.plot_acf(self.rate_diff)
        plt.show()
        return True

    def check_pacf(self):
        sm.graphics.tsa.plot_pacf(self.rate_diff)
        plt.show()
        return True


    # GARCH model fitting 하는 함수 
    def get_garch_model(self, column_name = 'Log_return', p = 1, q = 1):
        self.get_log_return()
        self.garch_model = arch_model(self.log_df[column_name], p = p, q = q)
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

if __name__ == "__main__":
    garch_model = GARCH()
    fit_garch = garch_model.get_garch_model()
    alpha = garch_model.get_alpha1()
    beta = garch_model.get_beta1()
    omega = garch_model.get_omega()