import numpy as np


numpy_functions = {
    "neg": "np.negative",
    "pos": "np.positive",
    "add": "np.add",
    "sub": "np.subtract",
    "exp": "np.exp",
    "natlog": "np.log"
}


def compute_locally(*args, **kwargs):
    operator = kwargs.get("operator", None)
    op_type = kwargs.get("op_type", None)

    if op_type == "unary":
        value1 = args[0]

        return eval("{}({})".format(numpy_functions[operator], value1))

    elif op_type == "binary":
        value1 = args[0]
        value2 = args[1]

        return eval("{}({}, {})".format(numpy_functions[operator], value1, value2))
