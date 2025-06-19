import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# True data generation parameters
t_true = (0.1, 0.9, 0.3)
true_pA, true_pB, true_pC = t_true
N = 1000

# Generate observations (latent a discarded)
data = []
for _ in range(N):
    a = 1 if random.random() < true_pA else 0
    o = (1 if random.random() < true_pB else 0) if a else (1 if random.random() < true_pC else 0)
    data.append((None, o))

# EM solver capturing trajectories
def em_solver(observations, init, max_iter=20):
    pA, pB, pC = init
    traj = {'pA': [pA], 'pB': [pB], 'pC': [pC]}
    for _ in range(max_iter):
        q1 = []
        for _, o in observations:
            like1 = pA * (pB**o) * ((1-pB)**(1-o))
            like0 = (1-pA) * (pC**o) * ((1-pC)**(1-o))
            q1.append(like1 / (like1 + like0))
        pA = sum(q1)/len(observations)
        sum_q1 = sum(q1)
        pB = sum(q * o for q, (_, o) in zip(q1, observations)) / sum_q1
        pC = sum((1-q) * o for q, (_, o) in zip(q1, observations)) / (len(observations) - sum_q1)
        traj['pA'].append(pA)
        traj['pB'].append(pB)
        traj['pC'].append(pC)
    return traj

# Initialization sets (labels are the tuples themselves)
initials = {
    '(0.15, 0.85, 0.25)': (0.15, 0.85, 0.25),
    '(0.12, 0.92, 0.28)': (0.12, 0.92, 0.28),
    '(0.08, 0.88, 0.32)': (0.08, 0.88, 0.32),
    '(0.90, 0.10, 0.50)': (0.90, 0.10, 0.50),
    '(0.80, 0.20, 0.10)': (0.80, 0.20, 0.10),
    '(0.20, 0.80, 0.60)': (0.20, 0.80, 0.60),
    '(0.60, 0.40, 0.90)': (0.60, 0.40, 0.90),
    '(0.40, 0.70, 0.20)': (0.40, 0.70, 0.20),
    '(0.30, 0.30, 0.80)': (0.30, 0.30, 0.80)
}
colors = plt.cm.tab10(range(len(initials)))
labels = list(initials.keys())

# Collect trajectories
histories = {lbl: em_solver(data, initials[lbl]) for lbl in labels}

# Plot 1: 3D trajectories with start/end markers and true generator
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for idx, lbl in enumerate(labels):
    hist = histories[lbl]
    xs, ys, zs = hist['pA'], hist['pB'], hist['pC']
    ax.plot(xs, ys, zs, color=colors[idx], linewidth=1.5)
    ax.scatter(xs[0], ys[0], zs[0], color=colors[idx], marker='o', s=50)
    ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[idx], marker='X', s=80)
# highlight true generator
ax.scatter(true_pA, true_pB, true_pC, color='black', marker='*', s=120, label='true')
# set axes labels and fixed limits
ax.set_xlabel('pA')
ax.set_ylabel('pB')
ax.set_zlabel('pC')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
# create legend for initializations only
for idx, lbl in enumerate(labels):
    ax.plot([], [], [], color=colors[idx], label=lbl)
ax.legend(loc='upper left', fontsize=8)
plt.title('EM Parameter Trajectories (Start: o, End: X)')
plt.tight_layout()
plt.show()

# Plot 2: average log-likelihood convergence
plt.figure()
average_ll = lambda obs, pA, pB, pC: sum(math.log(pA*(pB**o)*((1-pB)**(1-o)) + (1-pA)*(pC**o)*((1-pC)**(1-o))) for _, o in obs)/len(obs)
for idx, lbl in enumerate(labels):
    hist = histories[lbl]
    ll = [average_ll(data, hist['pA'][i], hist['pB'][i], hist['pC'][i]) for i in range(len(hist['pA']))]
    plt.plot(range(len(ll)), ll, color=colors[idx], linewidth=1.5)
    plt.scatter(0, ll[0], color=colors[idx], marker='o', s=50)
    plt.scatter(len(ll)-1, ll[-1], color=colors[idx], marker='X', s=80)
plt.xlabel('Iteration')
plt.ylabel('Avg Log-Likelihood')
plt.title('Log-Likelihood Convergence by Init')
# legend
for idx, lbl in enumerate(labels):
    plt.plot([], [], color=colors[idx], label=lbl)
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
