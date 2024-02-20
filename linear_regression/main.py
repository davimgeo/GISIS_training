from function import linear_regression_brute_force, linear_regression_gradient_descent, linear_regression_matrix
import time

id = 0

regression_model = [linear_regression_brute_force(),
                    linear_regression_gradient_descent(),
                    linear_regression_matrix()]

start = time.time()

regression_model[id].noise()
regression_model[id].solution_space()
regression_model[id].plot_mesh()

end = time.time()

print(end - start)