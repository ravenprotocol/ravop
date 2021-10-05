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


class TFJSOperators(object):
    SIGMOID = "sigmoid"
    SIN = "sin"
    SINH = "sinh"
    SOFTPLUS = "softplus"


functions = {'lin': 'LINEAR', 'add': 'ADDITION', 'sub': 'SUBTRACTION',
             'mul': 'MULTIPLICATION', 'div': 'DIVISION', 'pos': 'POSITIVE', 'neg': 'NEGATION',
             'exp': 'EXPONENTIAL', 'natlog': 'NATURAL_LOG', 'pow': 'POWER', 'square': 'SQUARE',
             'cube': 'CUBE', 'square_root': 'SQUARE_ROOT', 'cube_root': 'CUBE_ROOT', 'abs': 'ABSOLUTE',
             'matmul': 'MATRIX_MULTIPLICATION', 'multiply': 'MULTIPLY', 'dot': 'DOT',
             'transpose': 'TRANSPOSE', 'sum': 'MATRIX_SUM', 'sort': 'SORT', 'split': 'SPLIT',
             'reshape': 'RESHAPE', 'concat': 'CONCATENATE', 'min': 'MIN', 'max': 'MAX', 'unique': 'UNIQUE',
             'argmax': 'ARGMAX', 'argmin': 'ARGMIN', 'expand_dims': 'EXPAND_DIMS', 'inv': 'INVERSE', 'gather': 'GATHER',
             'reverse': 'REVERSE', 'stack': 'STACK', 'tile': 'TILE', 'slice': 'SLICE', 'find_indices': 'FIND_INDICES',
             'shape': 'SHAPE', 'greater': 'GREATER', 'greater_equal': 'GREATER_EQUAL', 'less': 'LESS',
             'less_equal': 'LESS_EQUAL', 'equal': 'EQUAL', 'not_equal': 'NOT_EQUAL', 'logical_and': 'LOGICAL_AND',
             'logical_or': 'LOGICAL_OR', 'logical_not': 'LOGICAL_NOT', 'logical_xor': 'LOGICAL_XOR', 'mean': 'MEAN',
             'average': 'AVERAGE', 'mode': 'MODE', 'variance': 'VARIANCE', 'median': 'MEDIAN',
             'std': 'STANDARD_DEVIATION', 'percentile': 'PERCENTILE', 'random': 'RANDOM',
             'bincount': 'BINCOUNT', 'where': 'WHERE', 'sign': 'SIGN', 'foreach': 'FOREACH',
             'one_hot_encoding': 'ONE_HOT_ENCODING', 'federated_training': "FEDERATED_TRAINING"}


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
