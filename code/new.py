import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------------------
# Data Generation
# -------------------------------
true_pA, true_pB, true_pC = 0.1, 0.9, 0.3
N = 1000
data = []
for _ in range(N):
    a = 1 if random.random() < true_pA else 0
    o = (1 if random.random() < true_pB else 0) if a else (1 if random.random() < true_pC else 0)
    data.append((None, o))
n1 = sum(o for _, o in data)
n0 = N - n1

# -------------------------------
# EM Solver (2 iterations)
# -------------------------------
def em_two_steps(init):
    pA, pB, pC = init
    for _ in range(2):
        q1 = [
            (pA * (pB**o) * ((1-pB)**(1-o))) /
            (pA * (pB**o) * ((1-pB)**(1-o)) + (1-pA) * (pC**o) * ((1-pC)**(1-o)))
            for _, o in data
        ]
        pA = sum(q1) / len(data)
        sum_q1 = sum(q1)
        pB = sum(q * o for q, (_, o) in zip(q1, data)) / sum_q1
        pC = sum((1-q) * o for q, (_, o) in zip(q1, data)) / (len(data) - sum_q1)
    return pA, pB, pC

# -------------------------------
# Collect Final Points & Fit Surface
# -------------------------------
num_points = 1000
final_pts = [em_two_steps((random.uniform(0.05,0.95),
                           random.uniform(0.05,0.95),
                           random.uniform(0.05,0.95)))
             for _ in range(num_points)]
fpA, fpB, fpC = zip(*final_pts)

# RBF fit for pC = f(pA,pB)
from scipy.interpolate import RBFInterpolator
eps = 0.05
grid_lin = np.linspace(eps, 1-eps, 60)
gx, gy = np.meshgrid(grid_lin, grid_lin)
coords = np.column_stack([gx.ravel(), gy.ravel()])
rbf = RBFInterpolator(np.column_stack([fpA, fpB]), fpC, neighbors=20, smoothing=0.01)
gz = rbf(coords).reshape(gx.shape)
# Clip surface to [0,1]
gz = np.clip(gz, 0, 1)

# -------------------------------
# Compute True Log-Likelihood Level
# -------------------------------
def avg_ll(pA, pB, pC):
    P1 = pA*pB + (1-pA)*pC
    return (n1 * math.log(P1) + n0 * math.log(1-P1)) / N

LL_true = avg_ll(true_pA, true_pB, true_pC)

# Build LL grid for isosurface (clipped domain)
res = 30
axis = np.linspace(eps, 1-eps, res)
Gx, Gy, Gz = np.meshgrid(axis, axis, axis, indexing='ij')
P1 = Gx*Gy + (1-Gx)*Gz
LL_grid = (n1 * np.log(P1) + n0 * np.log(1-P1)) / N

# Extract single isosurface at a level slightly below LL_true for better visualization
level_offset = -0.00000056 # 进一步降低浮动值，使等值面更加紧密
iso_level = LL_true + level_offset
spacing = (axis[1]-axis[0],)*3
verts, faces, normals, values = marching_cubes(LL_grid, level=iso_level, spacing=spacing)

# 计算每个面片的高度值（pC坐标）用于配色
face_heights = []
for face in faces:
    face_verts = verts[face]
    # 使用面片中心的pC坐标（z坐标）
    center_z = np.mean(face_verts[:, 2])
    face_heights.append(center_z)

face_heights = np.array(face_heights)
# 归一化到[0,1]范围用于配色
normalized_heights = (face_heights - face_heights.min()) / (face_heights.max() - face_heights.min() + 1e-8)

# 使用与左图相同的viridis配色方案
from matplotlib import cm
face_colors = cm.viridis(normalized_heights)

mesh = Poly3DCollection(verts[faces], alpha=0.6)
mesh.set_facecolors(face_colors)

# -------------------------------
# Plot 1x2: Left=Surface+Points, Right=LL Isosurface
# -------------------------------
fig = plt.figure(figsize=(12, 5))

# Left subplot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(fpA, fpB, fpC, alpha=0.3, c='royalblue', s=8, label='Final Points')
ax1.plot_surface(gx, gy, gz, alpha=0.6, cmap='viridis', edgecolor='none')
ax1.scatter(true_pA, true_pB, true_pC, color='red', marker='*', s=100, label='True Params')
ax1.set_xlabel('$p_A$'); ax1.set_ylabel('$p_B$'); ax1.set_zlabel('$p_C$')
ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.set_zlim(0,1)
ax1.set_title('EM Final Points + Fitted Surface')
ax1.legend()

# Right subplot
ax2 = fig.add_subplot(122, projection='3d')
ax2.add_collection3d(mesh)
ax2.scatter(true_pA, true_pB, true_pC, color='red', marker='*', s=100, label='True Params')
ax2.set_xlabel('$p_A$'); ax2.set_ylabel('$p_B$'); ax2.set_zlabel('$p_C$')
ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.set_zlim(0,1)
ax2.set_title('Log-Likelihood Isosurface (Colored by $p_C$)')
ax2.legend()

plt.tight_layout()
plt.show()

