import json
from functools import wraps

import numpy as np

from ..globals import globals as g
from ..strings import Operators, OpTypes, NodeTypes, functions, OpStatus
from ..utils import make_request, convert_to_ndarray


def t(value, dtype="ndarray", **kwargs):
    """
    To create scalars, tensors and other data types
    """
    if dtype == "ndarray":
        if isinstance(value, int):
            return Scalar(value, **kwargs)
        elif isinstance(value, float):
            return Scalar(value, **kwargs)
        else:
            return Tensor(value, **kwargs)
    elif dtype == "file":
        return File(value=value, dtype=dtype, **kwargs)


def create_op(operator=None, *args, **params):
    return __create_math_op(operator=operator, *args, **params)


def epsilon():
    return Scalar(1e-07)


def one():
    return Scalar(1)


def minus_one():
    return Scalar(-1)


def inf():
    return Scalar(np.inf)


def pi():
    return Scalar(np.pi)


def __create_math_op(*args, **kwargs):
    params = dict()
    for key, value in kwargs.items():
        if key in ["node_type", "op_type", "status", "name", "operator"]:
            continue
        if (
                isinstance(value, Op)
                or isinstance(value, Data)
                or isinstance(value, Scalar)
                or isinstance(value, Tensor)
        ):
            params[key] = value.id
        elif type(value).__name__ in ["int", "float"]:
            params[key] = Scalar(value).id
        elif isinstance(value, list) or isinstance(value, tuple):
            params[key] = Tensor(value).id
        elif type(value).__name__ == "str":
            params[key] = value
        elif isinstance(value, bool):
            params[key] = value

    if len(args) == 0:
        op_ids = None
        op_type = None
        node_type = None
    else:
        op_ids = []
        for op in args:
            op_ids.append(op.id)

        if len(op_ids) == 1:
            op_type = OpTypes.UNARY
        elif len(op_ids) == 2:
            op_type = OpTypes.BINARY
        else:
            op_type = None

        node_type = NodeTypes.MIDDLE

    op_ids = json.dumps(op_ids)
    node_type = kwargs.get("node_type", node_type)
    op_type = kwargs.get("op_type", op_type)
    status = kwargs.get("status", OpStatus.PENDING)
    operator = kwargs.get("operator", None)

    op = make_request("op/create/", "post", {
        "name": kwargs.get("name", None),
        "graph_id": g.graph_id,
        "node_type": node_type,
        "inputs": op_ids,
        "outputs": None,
        "op_type": op_type,
        "operator": operator,
        "status": status,
        "params": json.dumps(params),
    })

    # op = ravdb.create_op(
    #     name=kwargs.get("name", None),
    #     graph_id=g.graph_id,
    #     node_type=node_type,
    #     inputs=op_ids,
    #     outputs=None,
    #     op_type=op_type,
    #     operator=operator,
    #     status=status,
    #     params=json.dumps(params),
    # )
    op = op.json()
    print(type(op), op)
    return Op(id=op["op_id"])


class ParentClass(object):
    def __init__(self, id=None, **kwargs):
        self._error = None
        self._status_code = None

        if id is not None:
            self.__get(endpoint=self.get_endpoint)
        else:
            self.__create(self.create_endpoint, **kwargs)

    def __get(self, endpoint):
        print(endpoint)
        res = make_request(endpoint, "get")
        print("Response:GET:", res.json())
        status_code = res.status_code
        res = res.json()
        if status_code == 200:
            for k, v in res.items():
                self.__dict__[k] = v
            self._status_code = 200
        else:
            self._error = res['message']
            self._status_code = status_code

    def __create(self, endpoint, **kwargs):
        res = make_request(endpoint, "post", payload={**kwargs})
        print("Response:POST:", res.json(), kwargs)
        status_code = res.status_code
        res = res.json()
        if status_code == 200:
            # Set class attributes
            for k, v in res.items():
                self.__dict__[k] = v
            self._status_code = 200
        else:
            self._error = res['message']
            self._status_code = status_code

    @property
    def error(self):
        if hasattr(self, "_error"):
            return self._error
        return None

    @property
    def status_code(self):
        if hasattr(self, "_status_code"):
            return self._status_code
        return None

    def valid(self):
        if self.status_code == 200:
            return True
        else:
            return False


class Op(ParentClass):
    def __init__(self, id=None, **kwargs):
        self.get_endpoint = f"op/get/?id={id}"
        self.create_endpoint = f"op/create/"

        if id is not None:
            super().__init__(id=id)
        else:

            inputs = kwargs.get("inputs", None)
            outputs = kwargs.get("outputs", None)
            operator = kwargs.get("operator", None)

            if (inputs is not None or outputs is not None) and operator is not None:
                info = self.extract_info(**kwargs)
                info['graph_id'] = g.graph_id
                info['params'] = json.dumps(kwargs)
                info["name"] = kwargs.get("name", None)

                super().__init__(id, **info)

    def extract_info(self, **kwargs):
        inputs = kwargs.get("inputs", None)
        outputs = kwargs.get("outputs", None)
        operator = kwargs.get("operator", None)

        # Figure out node type
        if inputs is None and outputs is not None:
            node_type = NodeTypes.INPUT
        elif inputs is not None and outputs is None:
            node_type = NodeTypes.MIDDLE
        else:
            raise Exception("Invalid node type")

        if inputs is not None:
            if len(inputs) == 1:
                op_type = OpTypes.UNARY
            elif len(inputs) == 2:
                op_type = OpTypes.BINARY
            else:
                raise Exception("Invalid number of inputs")
        else:
            op_type = OpTypes.OTHER

        if outputs is None:
            status = OpStatus.PENDING
        else:
            status = OpStatus.COMPUTED

        inputs = json.dumps(inputs)
        outputs = json.dumps(outputs)

        return {
            "node_type": node_type,
            "op_type": op_type,
            "status": status,
            "inputs": inputs,
            "outputs": outputs,
            "operator": operator
        }

    # def __create(self, operator, inputs=None, outputs=None, **kwargs):
    #     if (inputs is not None or outputs is not None) and operator is not None:
    #         # Figure out node type
    #         if inputs is None and outputs is not None:
    #             node_type = NodeTypes.INPUT
    #         elif inputs is not None and outputs is None:
    #             node_type = NodeTypes.MIDDLE
    #         else:
    #             raise Exception("Invalid node type")
    #
    #         if inputs is not None:
    #             if len(inputs) == 1:
    #                 op_type = OpTypes.UNARY
    #             elif len(inputs) == 2:
    #                 op_type = OpTypes.BINARY
    #             else:
    #                 raise Exception("Invalid number of inputs")
    #         else:
    #             op_type = OpTypes.OTHER
    #
    #         if outputs is None:
    #             status = OpStatus.PENDING
    #         else:
    #             status = OpStatus.COMPUTED
    #
    #         inputs = json.dumps(inputs)
    #         outputs = json.dumps(outputs)
    #
    #         # op = ravdb.create_op(
    #         #     name=kwargs.get("name", None),
    #         #     graph_id=g.graph_id,
    #         #     node_type=node_type,
    #         #     inputs=inputs,
    #         #     outputs=outputs,
    #         #     op_type=op_type,
    #         #     operator=operator,
    #         #     status=status,
    #         #     params=json.dumps(kwargs),
    #         # )
    #         payload = {
    #             "name": kwargs.get("name", None),
    #             "graph_id": g.graph_id,
    #             "node_type": node_type,
    #             "inputs": inputs,
    #             "outputs": outputs,
    #             "op_type": op_type,
    #             "operator": operator,
    #             "status": status,
    #             "params": json.dumps(kwargs),
    #         }
    #         op = make_request("op/create/", "post", payload)
    #         op = op.json()
    #         print(type(op), op)
    #         return op['op_id']
    #     else:
    #         raise Exception("Invalid parameters")

    # def to_scalar(self):
    #     self._op_db = ravdb.refresh(self._op_db)
    #     if self._op_db.outputs is None or self._op_db.outputs == "null":
    #         return None
    #
    #     data_id = json.loads(self._op_db.outputs)[0]
    #     data = Data(id=data_id)
    #     return Scalar(data.value)

    def get_output(self):
        return self.get_data().get_value()

    def get_dtype(self):
        return self.get_data().get_dtype()

    def get_shape(self):
        return self.get_data().get_shape()

    def get_data(self):
        if self.outputs is None or self.outputs == "null":
            return None

        data_id = json.loads(self.outputs)[0]
        data = Data(id=data_id)
        return data

    def __str__(self):
        return (
            "Op:\nId:{}\nName:{}\nType:{}\nOperator:{}\n\nStatus:{}\n".format(
                self.id,
                self.name,
                self.op_type,
                self.operator,
                self.status,
            )
        )

    def __call__(self, *args, **kwargs):
        return self.get_output()

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __pos__(self):
        return pos(self)

    def __neg__(self):
        return neg(self)

    def __abs__(self):
        return abs(self)

    def __invert__(self):
        return inv(self)

    def __lt__(self, other):
        return less(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __eq__(self, other):
        return equal(self, other)

    def __ne__(self, other):
        return not_equal(self, other)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __getitem__(self, item):
        if type(item).__name__ == "slice":
            return self.slice(begin=item.start, size=item.stop - item.start)
        elif type(item).__name__ == "int":
            return self.gather(Scalar(item))
        elif type(item).__name__ == "tuple":
            var = self
            for i in item:
                var = var.gather(Scalar(i))
            return var


def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still be used normally

    return decorator


for key, value in functions.items():
    exec(
        """@add_method(Op)\ndef {}(*args, **kwargs):\n    return __create_math_op(*args, operator="{}", **kwargs)""".format(
            key, key
        )
    )


class Scalar(Op):
    def __init__(self, value=None, id=None, data=None, **kwargs):
        if id is not None:
            # Get
            super().__init__(id=id)

        elif data is not None:
            if data.valid():

                # Create scalar
                super().__init__(
                    operator=Operators.LINEAR, inputs=None, outputs=[data.id], **kwargs
                )
            else:
                self.__dict__['_status_code'] = 400
                self.__dict__['_error'] = "Invalid data"

        elif value is not None:
            # Create data and then op

            data = Data(value=value)
            if data.valid():
                super().__init__(
                    operator=Operators.LINEAR, inputs=None, outputs=[data.id], **kwargs
                )
            else:
                self.__dict__['_status_code'] = 400
                self.__dict__['_error'] = "Invalid data"

    def __str__(self):
        return "Scalar Op:\nId:{}\nOutput:{}\nStatus:{}\nDtype:{}\n".format(
            self.id, self.get_output(), self.status, self.get_dtype
        )

    def __float__(self):
        return float(self.get_output())


class Tensor(Op):
    """
    It supports:
    1. list
    2. ndarray
    3. string(list)
    """

    def __init__(self, value=None, id=None, data=None, **kwargs):
        if id is not None:
            # Get
            super().__init__(id=id)

        elif data is not None:
            if data.valid():
                # Create scalar
                super().__init__(
                    operator=Operators.LINEAR, inputs=None, outputs=[data.id], **kwargs
                )
            else:
                self.__dict__['_status_code'] = 400
                self.__dict__['_error'] = "Invalid data"

        elif value is not None:
            # Create data and then op
            data = Data(value=value)
            if data.valid():
                super().__init__(
                    operator=Operators.LINEAR, inputs=None, outputs=[data.id], **kwargs
                )
            else:
                self.__dict__['_status_code'] = 400
                self.__dict__['_error'] = "Invalid data"

    def __str__(self):
        return "Tensor Op:\nId:{}\nOutput:{}\nStatus:{}\nDtype:{}".format(
            self.id, self.get_output(), self.status, self.get_dtype
        )


class File(Op):
    def __init__(self, value, **kwargs):
        data = Data(value=value, dtype="file")
        super().__init__(
            operator=Operators.LINEAR, inputs=None, outputs=[data.id], **kwargs
        )

    @property
    def dtype(self):
        return "file"

    @property
    def shape(self):
        return None

    def __str__(self):
        return "File Op:\nId:{}\nOutput:{}\nStatus:{}\nDtype:{}\n".format(
            self.id, self.get_output(), self.status, self.dtype
        )

    def __call__(self, *args, **kwargs):
        return self.get_output()


class Data(ParentClass):
    def __init__(self, value=None, id=None, **kwargs):
        self.get_endpoint = f"data/get/?id={id}"
        self.create_endpoint = f"data/create/"

        # value = kwargs.get("value", None)
        if value is not None and isinstance(value, np.ndarray):
            value = value.tolist()
            kwargs['value'] = value
        else:
            kwargs['value'] = value

        super().__init__(id, **kwargs)

    def get_value(self):
        if hasattr(self, 'value'):
            return convert_to_ndarray(self.value)
        else:
            return None

    def get_dtype(self):
        if hasattr(self, "dtype"):
            return self.dtype
        else:
            return None

    def get_shape(self):
        if hasattr(self, 'value'):
            if self.value is not None:
                return self.value.shape
        return None


# class Data(object):
#     def __init__(self, id=None, **kwargs):
#         print(kwargs)
#
#         for k, v in kwargs.items():
#             self.__dict__["_{}".format(k)] = v
#
#         print(self.dtype)
#
#         if id is not None:
#             res = make_request(f"data/get?id={id}", "get")
#             if res.status_code == 200:
#                 data_dict = res.json()
#             # self._data_db = ravdb.get_data(data_id=id)
#             else:
#                 raise Exception("Invalid data id")
#
#         elif self.value is not None and self.dtype is not None:
#             if self.dtype == "ndarray":
#                 if type(self.value).__name__ == "list":
#                     value = np.array(self.value)
#                 elif type(self.value).__name__ == "str":
#                     x = json.loads(self.value)
#                     if type(x).__name__ == "list":
#                         value = np.array(self.value)
#             elif self.dtype == "file":
#                 pass
#
#             data_dict = self.__create(value=self.value, dtype=self.dtype)
#
#             if not data_dict:
#                 raise Exception("Unable to create data")
#             # if self._data_db is None:
#             #     raise Exception("Unable to create data")
#         print(self.__dict__)
#
#     def __create(self, value, dtype):
#         print("DTYPEEEE", dtype, "VALUEEEE", value)
#         res = make_request("data/create/", "post", {"value": value, "dtype": dtype})
#         print(res)
#         res = res.json()
#         for k, v in res.items():
#             self.__dict__["_{}".format(k)] = v
#         print(res, "DICT", res["id"])
#         # data = ravdb.create_data(type=dtype)
#
#         # if dtype == "ndarray":
#         #     file_path = dump_data(data.id, value)
#         #     # Update file path
#         #     ravdb.update_data(data, file_path=file_path)
#         # elif dtype in ["int", "float"]:
#         #     ravdb.update_data(data, value=value)
#
#         # elif dtype == "file":
#         #     filepath = os.path.join(
#         #         DATA_FILES_PATH, "data_{}_{}".format(data.id, value)
#         #     )
#         #     copy_data(source=value, destination=filepath)
#         #     ravdb.update_data(data, file_path=filepath)
#
#         return res
#
#     @property
#     def value(self):
#         if self.dtype == "ndarray":
#             file_path = self.file_path
#             value = np.load(file_path, allow_pickle=True)
#             return value
#         elif self.dtype in ["int", "float"]:
#             if self.dtype == "int":
#                 return int(self._value)
#             elif self.dtype == "float":
#                 return float(self._value)
#         elif self.dtype == "file":
#             return self.file_path
#
#     @property
#     def id(self):
#         return self._id
#
#     @property
#     def dtype(self):
#         return self._dtype
#
#     @property
#     def file_path(self):
#         return self._file_path
#
#     def __str__(self):
#         return "Data:\nId:{}\nDtype:{}\n".format(self.id, self.dtype)
#
#     def __call__(self, *args, **kwargs):
#         return self.value
#

class Graph(ParentClass):
    """A class to represent a graph object"""

    def __init__(self, id=None, **kwargs):
        self.get_endpoint = f"graph/get/?id={id}"
        self.create_endpoint = f"graph/create/"

        if id is None:
            id = g.graph_id

        if id is not None:
            super().__init__(id=id)
        else:
            super().__init__(**kwargs)

    # def add(self, op):
    #     """Add an op to the graph"""
    #     op.add_to_graph(self.id)
    #
    # @property
    # def progress(self):
    #     """Get the progress"""
    #     stats = self.get_op_stats()
    #     if stats["total_ops"] == 0:
    #         return 0
    #     progress = (
    #                        (stats["computed_ops"] + stats["computing_ops"] + stats["failed_ops"])
    #                        / stats["total_ops"]
    #                ) * 100
    #     return progress
    #
    # def get_op_stats(self):
    #     """Get stats of all ops"""
    #     ops = ravdb.get_graph_ops(graph_id=self.id)
    #
    #     pending_ops = 0
    #     computed_ops = 0
    #     computing_ops = 0
    #     failed_ops = 0
    #
    #     for op in ops:
    #         if op.status == "pending":
    #             pending_ops += 1
    #         elif op.status == "computed":
    #             computed_ops += 1
    #         elif op.status == "computing":
    #             computing_ops += 1
    #         elif op.status == "failed":
    #             failed_ops += 1
    #
    #     total_ops = len(ops)
    #     return {
    #         "total_ops": total_ops,
    #         "pending_ops": pending_ops,
    #         "computing_ops": computing_ops,
    #         "computed_ops": computed_ops,
    #         "failed_ops": failed_ops,
    #     }
    #
    # def clean(self):
    #     ravdb.delete_graph_ops(self._graph_db.id)
    #
    # @property
    # def ops(self):
    #     ops = ravdb.get_graph_ops(self.id)
    #     return [Op(id=op.id) for op in ops]
    #
    # def print_ops(self):
    #     """Print ops"""
    #     for op in self.ops:
    #         print(op)
    #
    # def get_ops_by_name(self, op_name, graph_id=None):
    #     ops = ravdb.get_ops_by_name(op_name=op_name, graph_id=graph_id)
    #     return [Op(id=op.id) for op in ops]

    def __str__(self):
        return "Graph:\nId:{}\nStatus:{}\n".format(self.id, self.status)
