

# callback function for the optimizer
# to record the objective values

import gurobipy as gp

def callback(model: gp.Model, where: int=gp.GRB.Callback.MIPSOL):
    """
    callback function to record the objective values and time
    """
    if where == gp.GRB.Callback.MIPSOL:
        time = model.cbGet(gp.GRB.Callback.RUNTIME)
        obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)

        model._obj_vals.append((time, obj))