# coding=utf-8
# 单纯形法的实现，只支持最简单的实现方法
# 且我们假设约束矩阵A的最后m列是可逆的
# 这样就必须满足A是行满秩的（m*n的矩阵）

import numpy as np


class Simplex(object):
    def __init__(self, c, A, b):
        # 形式 minf(x)=c.Tx
        # s.t. Ax=b
        self.c = c
        self.A = A
        self.b = b

    def run(self):
        c_shape = self.c.shape
        A_shape = self.A.shape
        b_shape = self.b.shape
        assert c_shape[0] == A_shape[1], "Not Aligned A with C shape"
        assert b_shape[0] == A_shape[0], "Not Aligned A with b shape"

        # 找到初始的B，N等值
        end_index = A_shape[1] - A_shape[0]
        N = self.A[:, 0:end_index]
        N_columns = np.arange(0, end_index)
        c_N = self.c[N_columns, :]
        # 第一个B必须是可逆的矩阵，其实这里应该用算法寻找，但此处省略
        B = self.A[:, end_index:]
        B_columns = np.arange(end_index, A_shape[1])
        c_B = self.c[B_columns, :]

        steps = 0
        while True:
            steps += 1
            print("Steps is {}".format(steps))
            is_optim, B_columns, N_columns = self.main_simplex(B, N, c_B, c_N, self.b, B_columns, N_columns)
            if is_optim:
                break
            else:
                B = self.A[:, B_columns]
                N = self.A[:, N_columns]
                c_B = self.c[B_columns, :]
                c_N = self.c[N_columns, :]

    def main_simplex(self, B, N, c_B, c_N, b, B_columns, N_columns):
        B_inverse = np.linalg.inv(B)
        P = (c_N.T - np.matmul(np.matmul(c_B.T, B_inverse), N)).flatten()
        if P.min() >= 0:
            is_optim = True
            print("Reach Optimization.")
            print("B_columns is {}".format(B_columns))
            print("N_columns is {}".format(sorted(N_columns)))
            best_solution_point = np.matmul(B_inverse, b)
            print("Best Solution Point is {}".format(best_solution_point.flatten()))
            print("Best Value is {}".format(np.matmul(c_B.T, best_solution_point).flatten()[0]))
            print("\n")
            return is_optim, B_columns, N_columns
        else:
            # 入基
            N_i_in = np.argmin(P)
            N_i = N[:, N_i_in].reshape(-1, 1)
            # By=Ni， 求出基
            y = np.matmul(B_inverse, N_i)
            x_B = np.matmul(B_inverse, b)
            N_i_out = self.find_out_base(y, x_B)
            tmp = N_columns[N_i_in]
            N_columns[N_i_in] = B_columns[N_i_out]
            B_columns[N_i_out] = tmp
            is_optim = False

            print("Not Reach Optimization")
            print("In Base is {}".format(tmp))
            print("Out Base is {}".format(N_columns[N_i_in]))   # 此时已经被换过去了
            print("B_columns is {}".format(sorted(B_columns)))
            print("N_columns is {}".format(sorted(N_columns)))
            print("\n")
            return is_optim, B_columns, N_columns

    def find_out_base(self, y, x_B):
        # 找到x_B/y最小且y>0的位置
        index = []
        min_value = []
        for i, value in enumerate(y):
            if value <= 0:
                continue
            else:
                index.append(i)
                min_value.append(x_B[i]/float(value))

        actual_index = index[np.argmin(min_value)]
        return actual_index


if __name__ == "__main__":
    '''
    c = np.array([-20, -30, 0, 0]).reshape(-1, 1)
    A = np.array([[1, 1, 1, 0], [0.1, 0.2, 0, 1]])
    b = np.array([100, 14]).reshape(-1, 1)
    '''
    c = np.array([-4, -1, 0, 0, 0]).reshape(-1, 1)
    A = np.array([[-1, 2, 1, 0, 0], [2, 3, 0, 1, 0], [1, -1, 0, 0, 1]])
    b = np.array([4, 12, 3]).reshape(-1, 1)
    simplex = Simplex(c, A, b)
    simplex.run()
    
##https://blog.csdn.net/cpluss/article/details/102596890
