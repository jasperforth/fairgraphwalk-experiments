# import numpy as np
# import scipy as sp
#
# p_dummy = np.array([[1/5,4/5,0,0],
#                   [4/5,0,2/3,4/3],
#                   [0,1/2,1/5,2/3],
#                   [0,1/2,5/3,2/3]])
#
#
# def get_graph_info(transitionmatrix: np.ndarray) -> dict:
#     graph_info: dict[str, any] = {}
#     n: int = transitionmatrix.shape[0]
#     d_max: int = 0
#     degrees: list[tuple[int, int]] = []
#     zeroes: list[list[int]] = []
#     edgecount: float = np.count_nonzero(transitionmatrix) / 2  # rename to edgecount
#
#     for i in range(n):
#         d: int = 0
#         zeroes.append([])
#
#         for j in range(n):
#             if i == j:
#                 continue
#             if transitionmatrix[i][j] != 0:
#                 d += 1
#             else:
#                 zeroes[i].append(j)
#
#         degrees.append((i, d))
#         d_max = max(d_max, d)
#
#     graph_info["n"] = n
#     graph_info["edgecount"] = edgecount
#     graph_info["d_max"] = d_max
#     graph_info["degrees"] = degrees
#     graph_info["zeroes"] = zeroes
#
#     # if edgecount > 20000:
#     #     #add stuff for subgradient
#
#     return graph_info
# def get_maximum_degree_p(graph_info: dict) -> np.ndarray:
#     n: int = graph_info["n"]
#     d_max: int = graph_info["d_max"]
#     degrees: list[tuple[int, int]] = graph_info["degrees"]
#     zeroes: list[list[int]] = graph_info["zeroes"]
#
#     p_md: np.ndarray = np.zeros((n, n))
#
#     for i in range(n):
#         d: int = degrees[i][1]
#
#         for j in range(n):
#             if i == j:
#                 p_md[i, j] = 1 - d / d_max
#             elif j in zeroes[i]:
#                 p_md[i, j] = 0
#             else:
#                 p_md[i][j] = 1 / d_max
#
#     return p_md
#
# p_md = get_maximum_degree_p(get_graph_info(p_dummy))
#
# eigen_data = {}
#
# eigen_data[1] = sp.sparse.linalg.eigsh(p_md, 2, which='LM')
#
# print(eigen_data[2-1][0][0])
# print(eigen_data[2-1][1])
#
# eigenvectors = eigen_data[1][1]
#
# u = np.array([item[0] for item in eigenvectors])
#
# for key, eigen in eigen_data.items():
#     print(key)
#     print(eigen)
#
# print(u)

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

P = cp.Variable((5, 5), symmetric=True)
OneVector = np.array([[1], [1], [1], [1], [1]])
I = np.identity(5)
s = cp.Variable()
n = P.size

constraints = [-s * I << P - (1 / n) * OneVector @ np.transpose(OneVector)]
constraints += [P - (1 / n) * OneVector @ np.transpose(OneVector) << s * I]
constraints += [P @ OneVector == OneVector]
constraints += [P == P.T]
constraints += [P[0][3] == 0]
constraints += [P[1][2] == 0]
constraints += [P[2][1] == 0]
constraints += [P[3][0] == 0]
constraints += [P[4][4] == 0]
constraints += [P >= 0]

prob = cp.Problem(cp.Minimize(s), constraints)
prob.solve(verbose=True, solver=cp.MOSEK)

print("The optimal value is", prob.value)
print("A solution P is")
print(P.value)

print(sp.linalg.eigsh(P.value, k=2, which='LM')[0][0])