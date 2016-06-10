import numpy as np
import random
import math

# Parameters
k = int(10)  # Interpretation space dimension
t = int(10)  # ALS number of iterations
landa = float(3)
lr_set_ratio = 0.9
test_set_ratio = 0.1
init_coef = 1
mean_rep = 5
f = open('log.txt', 'w')
f.write("t " + str(t) + "\tlanda " + str(landa))
f.write('\n')
f.write('k  \t Err \n')

print("t " + str(t))
print("landa " + str(landa))
# for k in range(1, 50, 4):
print("Dimension " + str(k))

# Importing data
# FILE 100K
# users = int(943)
# movies = int(1682)
# data_set_size = 100000
# R = np.empty((users,movies))
# Coor = np.empty((data_set_size,2))
# R[:] = np.NAN
#
# coord_index = 0
# with open("u.data") as FILE:
#     for line in FILE:
#         info = line.split("\t", 3)
#         user = int(info[0]) - 1
#         item = int(info[1]) - 1
#         rating = int(info[2])
#         R[user,item] = rating
#         Coor[coord_index,0] = user
#         Coor[coord_index,1] = item
#         coord_index += 1
# FILE.close()

# #  FILE 1M
users = int(6040)
movies = int(3952)
R = np.empty((users, movies))
R[:] = np.NAN
data_set_size = int(1000209)
Coor = np.empty((data_set_size, 2))

coord_index = 0
with open("ratings1M.dat") as FILE:
    for line in FILE:
        info = line.split("::", 3)
        user = int(info[0]) - 1
        item = int(info[1]) - 1
        rating = int(info[2])
        R[user, item] = rating
        Coor[coord_index, 0] = user
        Coor[coord_index, 1] = item
        coord_index += 1
FILE.close()

# #  FILE 10M
# users = int(71567)
# movies = int(10681)
# R = np.empty((users, movies))
# R[:] = np.NAN
# data_set_size = int(10000054)
# Coor = np.empty((data_set_size, 2))
#
# coord_index = 0
# with open("ratings10M.dat") as FILE:
#     for line in FILE:
#         info = line.split("::", 3)
#         user = int(info[0]) - 1
#         item = int(info[1]) - 1
#         rating = float(info[2])
#         R[user, item] = rating
#         Coor[coord_index, 0] = user
#         Coor[coord_index, 1] = item
#         coord_index += 1
# FILE.close()

# Sampling data
print("Sampling ...")
test_set_size = int(data_set_size * test_set_ratio)
Test_set = np.zeros((int(data_set_size * test_set_ratio), 3))

sample = random.sample(range(0, data_set_size), test_set_size)
data_count = 0
sample_index = 0

for w in range(0, test_set_size):
    i = int(Coor[sample[w], 0])
    j = int(Coor[sample[w], 1])
    Test_set[w, 0] = i
    Test_set[w, 1] = j
    Test_set[w, 2] = R[i, j]
    R[i, j] = float('nan')
    sample_index += 1
print("Sampling done")

# Learning
print("Learning ...")
U = np.random.rand(k, users)
V = np.random.rand(movies, k)
U *= 0.1
V *= 0.1
for i in range(0, users):
    U[0, i] = 1
for i in range(0, movies):
    V[i, 0] = 1
I = np.identity(k)

iteration = 0
while iteration < t:
    # Estimation of U
    for i in range(0, users):
        idx_IN_lines = np.nonzero(np.isnan(R[i, :]) == False)
        idx_IN_lines = np.asarray(idx_IN_lines)
        idx_IN_lines = idx_IN_lines.flatten()
        U[:, i] = np.linalg.solve(np.dot(np.transpose(V[idx_IN_lines, :]), V[idx_IN_lines, :]) + landa * I,
                                  np.dot(np.transpose(V[idx_IN_lines, :]), np.transpose(R[i, idx_IN_lines])))
    # Estimation of V
    for i in range(0, movies):
        idx_IN_lines = np.nonzero(np.isnan(R[:, i]) == False)
        idx_IN_lines = np.asarray(idx_IN_lines)
        idx_IN_lines = idx_IN_lines.flatten()
        S = np.linalg.solve(np.dot(U[:, idx_IN_lines], np.transpose(U[:, idx_IN_lines])) + landa * I,
                            np.dot(U[:, idx_IN_lines], R[idx_IN_lines, i]))
        V[i, :] = np.transpose(S)
    iteration += 1
print("Learning done")

# RMSE error computing
Err = 0
Err_mean = 0

for s in range(0, test_set_size):
    i = int(Test_set[s, 0])
    j = int(Test_set[s, 1])
    RT = Test_set[s, 2]
    Rij = np.dot(U[:, i], V[j, :])
    Err += math.pow((Rij - RT), 2)
Err /= test_set_size
Err = math.sqrt(Err)
print("RMSE")
print Err
f.write(str(k) + '\t')
f.write(str(Err) + '\n')
f.close()