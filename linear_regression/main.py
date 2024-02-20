from function import LinearRegressionBruteForce, LinearRegressionMatrix
import time

id = 0

start = time.time()
if id == 0:
    lr_brute_force = LinearRegressionBruteForce()

    lr_brute_force.solution_space()
    end = time.time()

    lr_brute_force.plot_mesh()
elif id == 1:
    lr_matrix = LinearRegressionMatrix()

    lr_matrix.linear_regression_solver()
    lr_matrix.y_regression()
    end = time.time()

    lr_matrix.plot_graph(lr_matrix.y_reg)

print(end - start)