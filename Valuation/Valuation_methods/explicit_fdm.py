import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# risk_free가 음수라 값이 터지는 문제 발생

def cal_prob(dt, sigma, dS, r):
    # 확률 계산
    p_u = (1 / 2) * (1 / (1 + r * dt)) * dt * ((sigma / dS) ** 2 + (r - (1/2) * (sigma ** 2)) / dS)
    #p_u = 0.16891002037726907
    p_m = (1 / (1 + r * dt)) * (1 - dt * (sigma / dS) ** 2)
    #p_m = 0.6665753549742043
    p_d = (1 / 2) * (1 / (1 + r * dt)) * dt * ((sigma / dS) ** 2 - (r - (1/2) * (sigma ** 2)) / dS)
    #p_d = 0.1643776571098333
    return p_u, p_m, p_d

def make_grid(S0, N_S, N_t, dS, T):
    # ln(S) 격자 생성
    lnS_max = np.log(S0) + N_S * dS / 2
    lnS_min = np.log(S0) - N_S * dS / 2
    lnS = np.linspace(lnS_min, lnS_max, N_S)
    # t 격자 생성
    t_max = T
    t_min = 0
    t = np.linspace(t_min, t_max, N_t)
    # 둘을 합해 2차원 격자 생성
    x, y = np.meshgrid(lnS, t)
    return lnS, x, y

def FDM(N_t, N_S, x, y, p_u, p_d, p_m):
    # 격자 초기화 
    # 1. 만기 시점의 콜옵션 페이오프
    V = np.zeros((N_t, N_S), dtype = float)
    for j in range(N_S):
        V[-1][j] = np.maximum(np.exp(x[-1][j]) - K, 0)

    # 2. 가치가 낮을 때의 페이오프
    for i in range(N_t):
        V[i][0] = 0

    # 3. 가치가 높을 때의 페이오프
    for i in range(N_t):
        V[i][-1] = np.maximum(np.exp(x[i][-1]) - K * np.exp(-r * (T - y[i][-1])), 0)

    # 4. Explicit FDM
    for i in reversed(range(N_t)):
        if i == N_t -1:
            continue
        for j in reversed(range(N_S)):
            if j == 0 or j == N_S -1:
                continue
            V[i][j] = p_u * V[i+1][j+1] + p_m * V[i+1][j] + p_d * V[i+1][j-1]
    return V

def BS_analytic(K, T, r, sigma, x, y, N_t, N_S):
    V = np.zeros((N_t, N_S), dtype = float)
    for i in range(N_t):
        for j in range(N_S):
            # zero division 때문에 1e-15를 더함
            d1 = (np.log(np.exp(x[i][j]) / K) + (r + 0.5 * sigma**2) * (T - y[i][j])) / (sigma * np.sqrt(T - y[i][j] + 1e-15))
            d2 = (np.log(np.exp(x[i][j]) / K) + (r - 0.5 * sigma**2) * (T - y[i][j])) / (sigma * np.sqrt(T - y[i][j] + 1e-15))
            #d1 = (np.log(np.exp(x[i][j]) / K) + (r + 0.5 * sigma**2) * (T )) / (sigma * np.sqrt(T ))
            #d2 = (np.log(np.exp(x[i][j]) / K) + (r - 0.5 * sigma**2) * (T)) / (sigma * np.sqrt(T ))
    
    # 콜옵션 가격 계산
            V[i][j] = np.exp(x[i][j]) * norm.cdf(d1) - K * np.exp(-r * (T - y[i][j])) * norm.cdf(d2)
    
    return V


if __name__ == "__main__":

    # 파라미터 설정
    K = 100              # 행사가격
    S0 = 100              # 기준이 되는 가격
    r = 0.05              # 무위험 이자율
    sigma = 0.2           # 기초 자산의 변동성
    T = 378 / 252          # 옵션 만기 
    N_t = 378             # 시간 격자의 칸 수 (1일 단위)
    N_S = 100           # 자산 가격 격자의 칸 수

    # 시간과 가격 격자 설정
    dt = T / N_t          # 시간 간격 (하루)
    dS = sigma * np.sqrt(3 * dt)  # 기초 자산 가격의 로그 간격

    p_u, p_m, p_d = cal_prob(dt, sigma, dS, r)
    lnS, x, y = make_grid(S0, N_S, N_t, dS, T)
    V_fdm = FDM(N_t, N_S, x, y, p_u, p_d, p_m)

    df = pd.DataFrame(V_fdm)
    S = np.exp(lnS)
    plt.plot(S, V_fdm[0], 'g-')
    plt.plot(S, V_fdm[100], 'r-')
    plt.plot(S, V_fdm[230], 'b-')
    plt.plot(S, V_fdm[370], 'm-')
    plt.show()
    df.to_csv('sample.csv', index=False)

    #BS 방정식에서 구한 analytic solution
    V_bs = BS_analytic(K, T, r, sigma, x, y, N_t, N_S)
    plt.plot(S, V_bs[0], 'g-')
    plt.plot(S, V_bs[100], 'r-')
    plt.plot(S, V_bs[230], 'b-')
    plt.plot(S, V_bs[370], 'm-')
    plt.show()
    SE = np.zeros(N_t)
    for i in range(N_t):
        error = 0
        for j in range(N_S):
            error = error + (V_fdm[i][j] - V_bs[i][j]) ** 2
        SE[i] = error
    t = np.linspace(0, T, N_t)
    print(SE)
    plt.plot(np.multiply(t, 252), SE)
    plt.show()

    # time 0에서 fdm이 계산한 option value와 bs로 계산한 option value의 차이
    plt.plot(S, V_fdm[0], 'g-')
    plt.plot(S, V_bs[0], 'r-')
    plt.show()