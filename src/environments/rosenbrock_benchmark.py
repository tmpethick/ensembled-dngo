

class Rosenbrock(object):
    def __init__(self, d):
        self.d = d
        super(Rosenbrock, self).__init__()

    def __call__(self, x):
        y = 0
        for i in range(0, self.d - 1, 2):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2
        return y

    def get_meta_information(self):
        return {'name': 'Rosenbrock',
                'num_function_evals': 200,
                'bounds': [[-5, 10]] * self.d,
                'f_opt': 0.0}
