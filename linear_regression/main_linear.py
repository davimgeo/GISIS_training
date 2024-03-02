from function import LinearRegressionBruteForce, LinearRegressionMatrix, cmp_gather
import time

id = 2

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
    elif id == 2:
        cmp_gather_model = cmp_gather()

        cmp_gather_model.solution_space()
        cmp_gather_model.plot_graph()
        cmp_gather_model.plot_mesh()

    return print(f"{end - start:5f}")

if __name__ == "__main__":
    linear_regression()

"""
id = 0 -> linear brute force
id = 1 -> linear matrix
id = 2 -> cmp_gather 
"""