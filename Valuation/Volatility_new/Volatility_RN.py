from Valuation.Volatility_new.Volatility import GARCH
from Valuation.Risk_free import get_diff
import numpy as np
# Risk free, lambda 계산해야됨...

class RN_GARCH(GARCH):
    
    def __init__(self, Ticker:str = 'KRW=X', start_date:str = '2022-08-30', end_date:str = '2024-08-30', N_t:int = 378):
        super().__init__(Ticker, start_date, end_date)
        super().fit_garch_model()
        np.random.seed(42)
        self.N_t = N_t
        self.historical_vol = self._get_historical_vol()
        self.risk_free = self._get_risk_free()
        self.lambda_risk = self._get_risk_premium()
        self.stock_process = None
        self.variance_process = None
        self.xi = None


    def _get_historical_vol(self):
        vol =  0
        for i in range(len(self.log_df)):
            vol = vol + self.log_df['Log_return'].iloc[i] ** 2
        vol = vol / len(self.log_df)
        return vol

    def _get_risk_free(self):
        rate_diff = get_diff()
        rf = np.mean(rate_diff['Price'] / 100)
        return rf

    def _get_risk_premium(self):
        beta_1 = super().get_beta1()
        expected_return = np.mean(self.log_df['Log_return']) / 100
        lambda_risk = (expected_return - self.risk_free) / beta_1 
        return lambda_risk
    #모델의 안정성 검증 필요

    def get_RN_process(self, N_t:int = 378):
        self.N_t = N_t
        alpha_1 = super().get_alpha1()
        beta_1 = super().get_beta1()
        alpha_0 = super().get_omega()
        self.variance_process = np.zeros(N_t)
        self.variance_process[0] = alpha_0 / (1 - (1 + self.lambda_risk ** 2)* alpha_1 - beta_1)
        self.xi = np.random.normal(0, 1, N_t + 1)
        for i in range(1, N_t):
            self.variance_process[i] = alpha_0 + alpha_1 * ((self.xi[i-1] - self.lambda_risk) ** 2) * self.variance_process[i-1] + beta_1 * self.variance_process[i-1]

        return self.variance_process, self.xi
    
    def get_risk_free(self):
        return self.risk_free
    
    def get_S0(self):
        return self.df['Close']['KRW=X'][0]
    
    def get_historical_vol(self):
        return self.historical_vol