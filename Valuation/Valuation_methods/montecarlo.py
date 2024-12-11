#from Volatility.GARCH import *

import sys
import numpy as np
sys.path.append("/Users/choejihun/Desktop/FIRM_2024/Risk_free")
import risk_free

def get_historical_vol():
    domestic_rf, foreign_rf = risk_free.get_risk_free_df()
    rate_diff = domestic_rf.sub(foreign_rf)
    rate_diff['Date'] = domestic_rf['Date']
    historical_vol =  0
    for i in range(len(rate_diff)):
        historical_vol = historical_vol + rate_diff['Price'].iloc[i] ** 2
    historical_vol = historical_vol / len(rate_diff)
    return historical_vol

def Montecarlo(sim_num, S_0, r_d, r_f, dt, N_t, volatility_model):
    sigma_process = volatility_model.forecast(horizon = 10, method = 'bootstrap', reindex = False).variance.iloc[0].values
    #이거 왜 horizon 더 키우면 에러 나는지 파악할 것
    sigma_process = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    stock_process = np.zeros((sim_num, N_t))
    for i in range(sim_num):
        stock_process[i][0] = S_0
        for j in range(1, N_t):
            Z_t = np.random.normal(0, 1)
            next_S = stock_process[i][j-1] * np.exp((r_d - r_f - 1/2 * sigma_process[j % 10] ** 2) * dt + sigma_process[j % 10] * np.sqrt(dt) * Z_t)
            stock_process[i][j] = next_S
    return stock_process

def Get_graph(MC_simulation, index):
    for i in range(len(MC_simulation)):
        plt.plot(MC_simulation[i])
    plt.title(index)
    plt.show()

if __name__ == "__main__":
    vol = get_historical_vol()
    print(vol)
    r_d = 0.05
    r_f = 0.03
    sigma = 0.2
    dt = 1/252
    Z_t = 0.3
    N_t = 200
    present_S = 0
    ticker = "005930.KS"
    startdate = '2023-10-10'
    enddate = '2024-10-10'
    S_0  = 1200
    sim_num = 100
    #garch_model = get_volatility_model(ticker, startdate, enddate)
    #MC_simulation = Montecarlo(sim_num, S_0, r_d, r_f, dt, N_t, garch_model)
    #Get_graph(MC_simulation, "SAMSUNG")
