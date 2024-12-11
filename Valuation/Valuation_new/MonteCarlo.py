from Valuation.Volatility_new import RN_GARCH
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

class MonteCarlo():
    def __init__(self, N_t:int = 378, sim_num:int = 100000, Ticker:str = 'KRW=X', start_date:str = '2022-08-30', end_date:str = '2024-08-30'):
        self.rn_garch = RN_GARCH(Ticker=Ticker, start_date=start_date, end_date=end_date)
        self.N_t = N_t
        self.risk_free = self.rn_garch.get_risk_free()
        self.S0 = self.rn_garch.get_S0()
        self.sim_num = sim_num
        self.stock_process = np.zeros((sim_num, N_t))
        self.payoff = None


    def simulation(self):
        for i in trange(0, self.sim_num, desc = 'Simulation'):
            self.stock_process[i][0] = self.S0
            variance_process, xi = self.rn_garch.get_RN_process(self.N_t)
            for j in range(1, self.N_t):
                self.stock_process[i][j] = self.stock_process[i][j-1] * np.exp(self.risk_free * 1 / 252 - 1/2 * variance_process[j] + xi[j] * np.sqrt(variance_process[j]))
        self.print_simulation()

    def print_simulation(self):
        for i in range(self.sim_num):
            plt.plot(self.stock_process[i])
        plt.title("MC simulation")
        plt.show()

    def cal_call_payoff(self, K:int = 1200):
        payoff_array = np.zeros(self.sim_num)
        for i in range(self.sim_num):
            payoff_array[i] = np.exp(-self.risk_free * self.N_t / 252) * np.max(self.stock_process[i][self.N_t-1] - K, 0)
        self.payoff = np.mean(payoff_array)
        return self.payoff
    

    #1000원 넣었다고 생각했을 때 가치
    def cal_DLS_payoff(self, K:int = 1200):
        payoff_array = np.zeros(self.sim_num)
        for i in range(self.sim_num):
            if self.stock_process[i][self.N_t-1] >= K:
                payoff_array[i] = 1000 * np.exp(-self.risk_free * self.N_t / 252) * 1.0376
            else:
                payoff_array[i] = 1000 * np.exp(-self.risk_free * self.N_t / 252) * 0.99
        self.payoff = np.mean(payoff_array)
        return self.payoff
    
    def get_stock_process(self):
        return self.stock_process