from cmp_gather_functions import cmp_gather

def cmp_gather_model():
    cmp_gather_object = cmp_gather()

    cmp_gather_object.solution_space()
    cmp_gather_object.plot_mesh()

if __name__ == "__main__":
    cmp_gather_model()