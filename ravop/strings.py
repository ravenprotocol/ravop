class OpTypes(object):
    UNARY = "unary"
    BINARY = "binary"
    OTHER = "other"


class NodeTypes(object):
    INPUT = "input"
    MIDDLE = "middle"
    OUTPUT = "output"


class Operators(object):
    # Arithmetic
    LINEAR = "linear"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    MULTIPLICATION = "multiplication"
    DIVISION = "division"
    POSITIVE = "positive"
    NEGATION = "negation"
    EXPONENTIAL = "exponential"
    NATURAL_LOG = "natural_log"
    POWER = "power"
    SQUARE = "square"
    CUBE = "cube"
    SQUARE_ROOT = "square_root"
    CUBE_ROOT = "cube_root"
    ABSOLUTE = "absolute"

    # Matrix
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    MULTIPLY = "multiply"  # Elementwise multiplication
    DOT = "dot"
    TRANSPOSE = "transpose"
    MATRIX_SUM = "matrix_sum"
    SORT = "sort"
    SPLIT = "split"
    RESHAPE = "reshape"
    CONCATENATE = "concatenate"
    MIN = "min"
    MAX = "max"
    UNIQUE = "unique"
    ARGMAX = "argmax"
    ARGMIN = "argmin"
    EXPAND_DIMS = "expand_dims"
    INVERSE = "inv"
    GATHER = "gather"
    REVERSE = "reverse"
    STACK = "stack"
    TILE = "tile"
    SLICE = "slice"
    FIND_INDICES = "find_indices"
    SHAPE = "shape"

    # Comparison Operators
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"

    # Logical
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    LOGICAL_NOT = "logical_not"
    LOGICAL_XOR = "logical_xor"

    # Statistical
    MEAN = "mean"
    AVERAGE = "average"
    MODE = "mode"
    VARIANCE = "variance"
    MEDIAN = "median"
    STANDARD_DEVIATION = "standard_deviation"
    PERCENTILE = "percentile"
    RANDOM = "random"

    BINCOUNT = "bincount"
    WHERE = "where"
    SIGN = "sign"
    FOREACH = "foreach"

    # Data Preprocessing
    ONE_HOT_ENCODING = "one_hot_encoding"

    SET_VALUE = "set_value"

    FEDERATED_MEAN = "federated_mean"
    FEDERATED_VARIANCE = "federated_variance"
    FEDERATED_STANDARD_DEVIATION = "federated_standard_deviation"


class TFJSOperators(object):
    SIGMOID = "sigmoid"
    SIN = "sin"
    SINH = "sinh"
    SOFTPLUS = "softplus"


functions = {'lin': Operators.LINEAR,
             'add': Operators.ADDITION,
             'sub': Operators.SUBTRACTION,
             'mul': Operators.MULTIPLICATION,
             'div': Operators.DIVISION,
             'pos': Operators.POSITIVE,
             'neg': Operators.NEGATION,
             'exp': Operators.EXPONENTIAL,
             'natlog': Operators.NATURAL_LOG,
             'pow': Operators.POWER,
             'square': Operators.SQUARE,
             'cube': Operators.CUBE,
             'square_root': Operators.SQUARE_ROOT,
             'cube_root': Operators.CUBE_ROOT,
             'abs': Operators.ABSOLUTE,
             'matmul': Operators.MATRIX_MULTIPLICATION,
             'multiply': Operators.MULTIPLY,
             'dot': Operators.DOT,
             'transpose': Operators.TRANSPOSE,
             'sum': Operators.MATRIX_SUM,
             'sort': Operators.SORT,
             'split': Operators.SPLIT,
             'reshape': Operators.RESHAPE,
             'concat': Operators.CONCATENATE,
             'min': Operators.MIN,
             'max': Operators.MAX,
             'unique': Operators.UNIQUE,
             'argmax': Operators.ARGMAX,
             'argmin': Operators.ARGMIN,
             'expand_dims': Operators.EXPAND_DIMS,
             'inv': Operators.INVERSE,
             'gather': Operators.GATHER,
             'reverse': Operators.REVERSE,
             'stack': Operators.STACK,
             'tile': Operators.TILE,
             'slice': Operators.SLICE,
             'find_indices': Operators.FIND_INDICES,
             'shape': Operators.SHAPE,
             'greater': Operators.GREATER,
             'greater_equal': Operators.GREATER_EQUAL,
             'less': Operators.LESS,
             'less_equal': Operators.LESS_EQUAL,
             'equal': Operators.EQUAL,
             'not_equal': Operators.NOT_EQUAL,
             'logical_and': Operators.LOGICAL_AND,
             'logical_or': Operators.LOGICAL_OR,
             'logical_not': Operators.LOGICAL_NOT,
             'logical_xor': Operators.LOGICAL_XOR,
             'mean': Operators.MEAN,
             'average': Operators.AVERAGE,
             'mode': Operators.MODE,
             'variance': Operators.VARIANCE,
             'median': Operators.MEDIAN,
             'std': Operators.STANDARD_DEVIATION,
             'percentile': Operators.PERCENTILE,
             'random': Operators.RANDOM,
             'bincount': Operators.BINCOUNT,
             'where': Operators.WHERE,
             'sign': Operators.SIGN,
             'foreach': Operators.FOREACH,
             'one_hot_encoding': Operators.ONE_HOT_ENCODING,
             'set_value': Operators.SET_VALUE,

             # Federated functions

             'federated_mean': Operators.FEDERATED_MEAN,
             'federated_variance': Operators.FEDERATED_VARIANCE,
             'federated_standard_deviation': Operators.FEDERATED_STANDARD_DEVIATION,
             }


class Status(object):
    PENDING = "pending"
    COMPUTED = "computed"
    FAILED = "failed"
    COMPUTING = "computing"


class OpStatus(Status):
    pass


class GraphStatus(Status):
    pass


class MappingStatus(Status):
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    NOT_ACKNOWLEDGED = "not_acknowledged"
    NOT_COMPUTED = "not_computed"
    REJECTED = "rejected"
