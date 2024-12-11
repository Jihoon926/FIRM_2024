import numpy as np
import pandas as pd
from arch import arch_model
import GARCH
import matplotlib.pyplot as plt
from tqdm import trange


# Risk free, lambda 계산해야됨...

class Risk_Neutral_GARCH(GARCH.GARCH):
    
    def __init__(self, Ticker = 'KRW=X'):
        self.risk_free = None
        np.random.seed(42)
        super().__init__(Ticker)
        self.stock_process = None
        self.sim_num = 100
        self.variance = None
        self.payoff = None
        self.N_t = None
        self.K = None
        self.lambda_risk = None
        self.S0 = self.df['Close']['KRW=X'][0]
        self.historical_vol = None
        self.expected_return = None
        self.lambda_risk = None

    def cal_call_payoff(self):
        payoff_array = np.zeros(self.sim_num)
        for i in range(self.sim_num):
            payoff_array[i] = np.exp(-self.risk_free * self.N_t / 252) * np.max(self.stock_process[i][self.N_t-1] - self.K, 0)
        self.payoff = np.mean(payoff_array)
        return self.payoff
    

    def cal_DLS_payoff(self):
        payoff_array = np.zeros(self.sim_num)
        for i in range(self.sim_num):
            if self.stock_process[i][self.N_t-1] >= 1200:
                payoff_array[i] = 1000 * np.exp(-self.risk_free * self.N_t / 252) * 1.0376
            else:
                payoff_array[i] = 1000 * np.exp(-self.risk_free * self.N_t / 252) * 0.99
        self.payoff = np.mean(payoff_array)
        return self.payoff



    def print_simulation(self):
        for i in range(self.sim_num):
            plt.plot(self.stock_process[i])
        plt.title("MC simulation")
        plt.show()


    def get_historical_vol(self):
        super().get_log_return()
        self.historical_vol =  0
        for i in range(len(self.log_df)):
            self.historical_vol = self.historical_vol + self.log_df['Log_return'].iloc[i] ** 2
        self.historical_vol = self.historical_vol / len(self.log_df)
        #print("Historical vol is: ", self.historical_vol)
        return self.historical_vol


    def get_risk_premium(self, beta_1, lambda_risk):
        if lambda_risk != None:
            self.lambda_risk = lambda_risk
            return self.lambda_risk
        
        if self.historical_vol == None:
            self.get_historical_vol()
        if self.risk_free == None:
            self.risk_free = np.mean(self.rate_diff['Price']) / 100
        super().get_log_return()
        self.expected_return = np.mean(self.log_df['Log_return']) / 100
        self.lambda_risk = (self.expected_return / 100 - self.risk_free / 100) / beta_1
        #print(self.lambda_risk)
        return self.lambda_risk

    def get_RN_process(self, sim_num:int = 100000, N_t:int = 378, lambda_risk = None, K:int = 1200):
        self.risk_free = np.mean(self.rate_diff['Price']) / 100
        #self.risk_free = rf
        self.N_t = N_t
        self.K = K
        self.sim_num = sim_num
        super().get_garch_model()
        alpha_1 = super().get_alpha1()
        beta_1 = super().get_beta1()
        alpha_0 = super().get_omega()
        self.get_risk_premium(beta_1, lambda_risk)
        #print(self.lambda_risk)
        #print(self.garch_model.summary())
        self.stock_process = np.zeros((sim_num, N_t))
        self.variance = np.zeros(N_t)
        print(self.garch_model.summary())

        self.variance[0] = alpha_0 / (1 - (1 + self.lambda_risk ** 2)* alpha_1 - beta_1)

        for i in trange(0, self.sim_num, desc = 'Simulation'):
            xi = np.random.normal(0, 1, N_t + 1)
            self.stock_process[i][0] = self.S0
            for j in range(1, N_t):
                self.variance[j] = alpha_0 + alpha_1 * ((xi[j-1] - self.lambda_risk) ** 2) * self.variance[j-1] + beta_1 * self.variance[j-1]
                #variance[j] = alpha_0 + beta_1 * variance[j-1]
                self.stock_process[i][j] = self.stock_process[i][j-1] * np.exp(self.risk_free * 1 / 252 - 1/2 * self.variance[j] + xi[j] * np.sqrt(self.variance[j]))

        return self.variance, self.stock_process

    def print_risk_free(self):
        print("risk free rate is:", self.risk_free)
        return self.risk_free
    

if __name__ == '__main__':
    rn_garch = Risk_Neutral_GARCH()
    variance, stock_process = rn_garch.get_RN_process(sim_num = 10000)
    payoff = rn_garch.cal_call_payoff()
    payoff_dls = rn_garch.cal_DLS_payoff()
    rn_garch.print_simulation()
    rn_garch.print_risk_free()
    print("Payoff of DLS is: ", payoff_dls)
    print(payoff)


    #K_array = np.arange(71, 101, 1)
    #payoff_array = np.zeros(30)
    #for i in K_array:
        #rn_garch.get_RN_process(sim_num = 10000, K = i)
        #payoff_array[i-71] = rn_garch.cal_call_payoff()
    #plt.plot(K_array, payoff_array, 'r-')
    #plt.title('Call option price on different strike prices')
    #print(payoff_array)
    #plt.show()


    #S_array = np.arange(71, 101, 1)
    #payoff_array = np.zeros(30)
    #for i in S_array:
        #rn_garch.get_RN_process(sim_num = 10000, K = 85)
        #payoff_array[i-71] = rn_garch.cal_call_payoff()
    #plt.plot(S_array, payoff_array, 'r-')
    #plt.title('Call option price on different strike prices')
    #print(payoff_array)
    #plt.show()