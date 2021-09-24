from enum import Enum


class OpStatus(Enum):
    PENDING = "pending"
    COMPUTING = "computing"
    COMPUTED = "computed"
    FAILED = "failed"


class GraphStatus(Enum):
    PENDING = "pending"
    COMPUTING = "computing"
    COMPUTED = "computed"
    FAILED = "failed"


class ClientOpMappingStatus(Enum):
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    NOT_ACKNOWLEDGED = "not_acknowledged"
    COMPUTING = "computing"
    COMPUTED = "computed"
    NOT_COMPUTED = "not_computed"
    FAILED = "failed"
    REJECTED = "rejected"