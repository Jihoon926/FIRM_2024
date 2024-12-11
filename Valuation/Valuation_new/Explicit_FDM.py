import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Valuation.Volatility_new import RN_GARCH



class Explicit_FDM():
    def __init__(self, K:int =1200, N_t:int = 378, N_s:int=100):
        self.rn_garch = RN_GARCH()
        self.K = K
        self.N_t = N_t
        self.N_s = N_s
        self.dt = 1/252
        self.risk_free = self.rn_garch.get_risk_free()
        self.S0 = self.rn_garch.get_S0()
        self.historical_vol = self.rn_garch.get_historical_vol()
        self.dS = self.historical_vol * np.sqrt(3 * self.dt)
        self.p_u, self.p_m, self.p_d = self.cal_prob()
        self.lnS, self.x_grid, self.y_grid = self.make_grid()
        self.V_fdm = self.FDM()

    def cal_prob(self):
        # 확률 계산
        p_u = (1 / 2) * (1 / (1 + self.risk_free * self.dt)) * self.dt * ((self.historical_vol / self.dS) ** 2 + (self.risk_free - (1/2) * (self.historical_vol ** 2)) / self.dS)
        p_m = (1 / (1 + self.risk_free * self.dt)) * (1 - self.dt * (self.historical_vol / self.dS) ** 2)
        p_d = (1 / 2) * (1 / (1 + self.risk_free * self.dt)) * self.dt * ((self.historical_vol / self.dS) ** 2 - (self.risk_free - (1/2) * (self.historical_vol ** 2)) / self.dS)
        return p_u, p_m, p_d

    def make_grid(self):
        # ln(S) 격자 생성
        lnS_max = np.log(self.S0) + self.N_s * self.dS / 2
        lnS_min = np.log(self.S0) - self.N_s * self.dS / 2
        lnS = np.linspace(lnS_min, lnS_max, self.N_s)
        # t 격자 생성
        t_max = self.dt * self.N_t
        t_min = 0
        t = np.linspace(t_min, t_max, self.N_t)
        # 둘을 합해 2차원 격자 생성
        x, y = np.meshgrid(lnS, t)
        return lnS, x, y

    def FDM(self):
        # 격자 초기화 
        # 1. 만기 시점의 콜옵션 페이오프
        V = np.zeros((self.N_t, self.N_s), dtype = float)
        for j in range(self.N_s):
            if self.x_grid[-1][j] >= self.K:
                V[-1][j] = 1.0376
            else:
                V[-1][j] = 0.99

        # 2. 가치가 낮을 때의 페이오프
        for i in range(self.N_t):
            V[i][0] = 0.99

        # 3. 가치가 높을 때의 페이오프
        for i in range(self.N_t):
            V[i][-1] = 1.0376

        # 4. Explicit FDM
        for i in reversed(range(self.N_t)):
            if i == self.N_t -1:
                continue
            for j in reversed(range(self.N_s)):
                if j == 0 or j == self.N_s -1:
                    continue
                V[i][j] = self.p_u * V[i+1][j+1] + self.p_m * V[i+1][j] + self.p_d * V[i+1][j-1]
        return V

    def plot_FDM(self):
        S = np.exp(self.lnS)
        plt.plot(S, self.V_fdm[0], color='limegreen')
        plt.plot(S, self.V_fdm[round(self.N_t * 0.25)], 'g-')
        plt.plot(S, self.V_fdm[round(self.N_t * 0.5)], 'r-')
        plt.plot(S, self.V_fdm[round(self.N_t * 0.75)], 'b-')
        plt.plot(S, self.V_fdm[self.N_t -1], 'm-')
        plt.show()
    
    def save_FDM(self):
        df = pd.DataFrame(self.V_fdm)
        df.to_csv('sample.csv', index=False)