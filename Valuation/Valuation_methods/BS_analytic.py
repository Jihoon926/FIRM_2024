import numpy as np
from scipy.stats import norm

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
sigma = 0.2 # 변동성
K = 100              # 행사가격
r = 0.05              # 무위험 이자율 (연간)
T = 378 / 252          # 옵션 만기 (30일을 연 단위로 환산)
N_t = 378             # 시간 격자의 칸 수 (1일 단위)
N_S = 100           # 자산 가격 격자의 칸 수
dt = T / N_t          # 시간 간격 (하루)
dS = sigma * np.sqrt(3 * dt)  # 기초 자산 가격의 로그 간격
lnS_max = np.log(S) + N_S * dS / 2
lnS_min = np.log(S) - N_S * dS / 2
lnS = np.linspace(lnS_min, lnS_max, N_S)

# 콜옵션 가격 계산
call_price = black_scholes_call(S, K, T, r, sigma)
print(f"유럽형 콜옵션 가격: {call_price:.2f}")