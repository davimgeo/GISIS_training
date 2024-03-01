from function import LinearRegressionBruteForce, LinearRegressionMatrix
import time

id = 1

def linear_regression():
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

    return print(f"{end - start:5f}")

if __name__ == "__main__":
    linear_regression()