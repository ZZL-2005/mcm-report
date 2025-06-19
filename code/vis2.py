import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==== 数据生成（1D GMM） ====
np.random.seed(0)
X = np.hstack([
    np.random.normal(-2, 1, size=50),
    np.random.normal( 2, 1, size=50)
])
pi = 0.5
sigma = 1.0

# ==== EM 只更新 mu1, mu2 ====
def run_em(X, mu1_0, mu2_0, iters=9):
    mu1s, mu2s = [mu1_0], [mu2_0]
    mu1, mu2 = mu1_0, mu2_0
    for _ in range(iters):
        r1 = pi * norm.pdf(X, mu1, sigma)
        r2 = (1-pi) * norm.pdf(X, mu2, sigma)
        gamma = r1 / (r1 + r2)
        mu1 = np.sum(gamma * X) / np.sum(gamma)
        mu2 = np.sum((1-gamma) * X) / np.sum(1-gamma)
        mu1s.append(mu1)
        mu2s.append(mu2)
    return mu1s, mu2s

mu1s, mu2s = run_em(X, mu1_0=0.0, mu2_0=0.5, iters=9)

# ==== 对数似然和下界定义 ====
def log_lik(mu1, mu2, X):
    return np.sum(np.log(pi*norm.pdf(X,mu1,sigma)
                         + (1-pi)*norm.pdf(X,mu2,sigma)))

def lower_bound(mu1, mu2, mu1_old, mu2_old, X):
    r1 = pi * norm.pdf(X, mu1_old, sigma)
    r2 = (1-pi) * norm.pdf(X, mu2_old, sigma)
    gamma = r1 / (r1 + r2)
    t1 = gamma*(np.log(pi*norm.pdf(X,mu1,sigma)) - np.log(gamma))
    t2 = (1-gamma)*(np.log((1-pi)*norm.pdf(X,mu2,sigma)) - np.log(1-gamma))
    return np.sum(t1 + t2)

# ==== 绘图：沿轨迹截面 ====
taus = np.linspace(0,1,200)
plt.figure(figsize=(8,6))

for t in range(len(mu1s)-1):
    p0 = np.array([mu1s[t],   mu2s[t]])
    p1 = np.array([mu1s[t+1], mu2s[t+1]])
    # 生成这一段上的所有点
    pis = p0[None,:] + taus[:,None]*(p1-p0)[None,:]
    # 计算两条曲线
    ll = [log_lik(mu1, mu2, X) 
          for mu1,mu2 in pis]
    bd = [lower_bound(mu1, mu2, p0[0],p0[1], X)
          for mu1,mu2 in pis]
    # 画实线和虚线
    plt.plot(taus, ll, '-',  alpha=0.7, label=f'LL iter {t+1}' if t<1 else "")
    plt.plot(taus, bd, '--', alpha=0.7, label=f'Bound iter {t+1}' if t<1 else "")
    # 标记切点 tau=1
    plt.scatter(1, ll[-1], color='k', s=20)

plt.xlabel(r'$\tau$ along EM update direction')
plt.ylabel('Objective')
plt.title('LL and Bound slices along EM trajectory')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
