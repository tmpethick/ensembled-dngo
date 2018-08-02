
from src.tests import *

f, bounds, f_opt = prepare_benchmark(Branin())
bo = test_gp(f, bounds, 100, do_plot=False)
ir = acc_ir(bo.model.Y, f_opt)
