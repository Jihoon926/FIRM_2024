import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import GARCH

sys.path.append("/Users/choejihun/Desktop/FIRM_2024/Risk_free")
import risk_free

def get_historical_vol():
    garch_model = GARCH.GARCH()
    fit_garch = garch_model.get_garch_model()
    log_df = garch_model.get_log_return()
    historical_vol =  0
    for i in range(len(log_df)):
        historical_vol = historical_vol + log_df['Log_return'].iloc[i] ** 2
    historical_vol = historical_vol / len(log_df)
    return historical_vol


def black_scholes_call(S, K, T, r, sigma):
    """
    블랙-숄즈 방정식을 이용해 유럽형 콜옵션 가격을 계산하는 함수.
    
    S: 기초자산의 현재 가격
    K: 옵션 행사가격
    T: 만기까지 남은 시간 (연 단위)
    r: 무위험 이자율 (연 단위)
    sigma: 기초자산의 변동성 (연 단위)
    """
    # d1과 d2 계산
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # 콜옵션 가격 계산
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

# 예제 입력값
S = 100  # 기초자산 가격
sigma = get_historical_vol() # 변동성
K = 100              # 행사가격
r = -0.017674634146341465   # 무위험 이자율 (연간)
T = 378 / 252          # 옵션 만기 (30일을 연 단위로 환산)
N_t = 378             # 시간 격자의 칸 수 (1일 단위)
N_S = 100           # 자산 가격 격자의 칸 수
dt = T / N_t          # 시간 간격 (하루)
dS = sigma * np.sqrt(3 * dt)  # 기초 자산 가격의 로그 간격
lnS_max = np.log(S) + N_S * dS / 2
lnS_min = np.log(S) - N_S * dS / 2
lnS = np.linspace(lnS_min, lnS_max, N_S)



"""
K_array = np.arange(71, 101, 1)
call_price_array = np.zeros(30)
# 콜옵션 가격 계산
for i in K_array:
    call_price_array[i-71]= black_scholes_call(S, i, T, r, sigma)

#Simulation num = 1000
garch_price_array = np.array([27.20412764, 26.31918587, 24.83289656, 24.11145912, 22.93158124, 21.69085262,
 21.05332739, 20.26398709, 18.86796154, 17.81744289, 16.92223794, 15.38233778,
 15.08676145, 13.82292792, 12.29904384, 11.96005321, 10.00856108,  9.56887845,
  8.30198971,  7.41562003,  6.66881491,  5.07470806,  4.86512649,  4.18932182,
  2.39007754,  1.37684442,  0.31144469, -0.15392065, -1.56351353, -2.39623962,])


#Simulation num = 10000
garch_price_array = np.array([27.13018153, 25.93953915, 25.1807631,  24.12645737, 22.89166185, 21.8289626,
 20.97994802, 19.81085797, 18.81975791, 17.86866896, 16.79158376, 15.75162227,
 14.50352897, 13.41759201, 12.75035062, 11.51707367, 10.49743805,  9.67523649,
  8.58886303,  7.55990299,  6.48087338,  5.63987349,  4.64660187,  3.68393379,
  2.49881405,  1.42578172,  0.34415058, -0.63526189, -1.66741039, -2.66463963])

plt.plot(K_array[:-3], call_price_array[:-3], 'r-')
plt.xlabel('Strike Price')
plt.ylabel('Call option price at t = 0')
plt.plot(K_array[:-3], garch_price_array[:-3], 'g-')
plt.title('Call option price on different strike prices(S_0 = 100)')
plt.show()

plt.plot(K_array[:-3], np.subtract(garch_price_array[:-3], call_price_array[:-3]), 'r-')
plt.xlabel('Strike price')
plt.ylabel('Difference between two call option prices(S_0 = 100)')
plt.title('Garch_price - BS_price')
plt.show()
"""



S_array = np.arange(71, 101, 1)
call_price_array = np.zeros(30)
# 콜옵션 가격 계산(S0를 바꾸면서)
for i in S_array:
    call_price_array[i-71]= black_scholes_call(i, 85, T, r, sigma)


garch_price_array = np.array([-16.34682844, -15.25650698, -14.37568631, -13.17856008, -12.1977193,
 -11.35459954, -10.38312424,  -9.24569154,  -8.35756143,  -7.3298838,
  -6.26908184,  -5.31008125,  -4.32127236,  -3.50755608,  -2.56043146,
  -1.25441005,  -0.43364707,   0.57099206,   1.75149212,   2.69833079,
   3.69622551,   4.64801437,   5.82012759,   6.85282274,   7.91526313,
   8.76543391,   9.72116441,  10.66754475,  11.7140263,   12.70872088])

plt.plot(S_array[17:], call_price_array[17:], 'r-')
plt.xlabel('S_0')
plt.ylabel('Call option price at t = 0')
plt.plot(S_array[17:], garch_price_array[17:], 'g-')
plt.title('Call option price on different initial underlying price(K = 100)')
plt.show()

plt.plot(S_array[17:], np.subtract(garch_price_array[17:], call_price_array[17:]), 'r-')
plt.xlabel('S_0')
plt.ylabel('Difference between two call option prices(S_0 = 100)')
plt.title('Garch_price - BS_price')
plt.show()