import datetime

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Graph(Base):
    __tablename__ = 'graph'
    id = Column(Integer, primary_key=True)
    ops = relationship("Op", backref="graph")

    # Status of this graph 1. pending 2. computing 3. computed 4. failed
    status = Column(String(10), default="pending")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Data(Base):
    __tablename__ = 'data'
    id = Column(Integer, primary_key=True)
    type = Column(String(20), nullable=False)
    file_path = Column(String(200), nullable=True)
    value = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Client(Base):
    __tablename__ = 'client'
    id = Column(Integer, primary_key=True)
    cid = Column(String(100), nullable=False)
    sid = Column(String(100), nullable=True)
    client_ip = Column(String(20), nullable=True)
    status = Column(String(20), nullable=False, default="disconnected")

    # 1. ravop 2. ravjs
    type = Column(String(10), nullable=True)
    client_ops = relationship("ClientOpMapping", backref="client", lazy="dynamic")

    reporting = Column(String(20), nullable=False, default="pending")
    ftp_credentials = Column(String(100), nullable=True, default=None)
    context = Column(Text, nullable=True, default=None)

    connected_at = Column(DateTime, default=datetime.datetime.utcnow)
    disconnected_at = Column(DateTime, default=datetime.datetime.utcnow)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Op(Base):
    __tablename__ = 'op'
    id = Column(Integer, primary_key=True)

    # Op name
    name = Column(String(20), nullable=True)

    # Graph id
    graph_id = Column(Integer, ForeignKey('graph.id'))

    # Store list of op ids
    inputs = Column(Text, nullable=True)

    # Store list of data ids
    outputs = Column(Text, nullable=True)

    # 1. input 2. output 3. middle
    node_type = Column(String(10), nullable=True)
    op_type = Column(String(50), nullable=True)
    operator = Column(String(50), nullable=True)
    message = Column(Text, nullable=True)

    # 1. pending 2. computing 3. computed 4. failed
    status = Column(String(10), default="pending")

    # Dict of params
    params = Column(Text, nullable=True)

    op_mappings = relationship("ClientOpMapping", backref="op", lazy="dynamic")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ClientOpMapping(Base):
    __tablename__ = "client_op_mapping"
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('client.id'))
    op_id = Column(Integer, ForeignKey('op.id'))
    sent_time = Column(DateTime, default=None)
    response_time = Column(DateTime, default=None)

    # 1. computing 2. computed 3. failed
    status = Column(String(10), default="computing")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class GraphClientMapping(Base):
    __tablename__ = "graph_client_mapping"
    id = Column(Integer, primary_key=True)
    graph_id = Column(Integer, ForeignKey('graph.id'))
    client_id = Column(Integer, ForeignKey('client.id'))
    sent_time = Column(DateTime, default=None)
    response_time = Column(DateTime, default=None)

    # 1. computing 2. computed 3. failed
    status = Column(String(10), default="computing")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


"""
Federated and analytics
"""


class Objective(Base):
    __tablename__ = "objective"
    id = Column(Integer, primary_key=True)
    graph_id = Column(Integer, ForeignKey('graph.id'))
    name = Column(String(50), nullable=True, default=None)
    operator = Column(String(50), nullable=True, default=None)
    rules = Column(Text, nullable=True, default=None)
    result = Column(Text, nullable=True, default=None)
    # 1. pending 2. active 3. completed 4. failed
    status = Column(String(10), default="pending")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ObjectiveClientMapping(Base):
    __tablename__ = "objective_client_mapping"
    id = Column(Integer, primary_key=True)
    objective_id = Column(Integer, ForeignKey('objective.id'))
    client_id = Column(Integer, ForeignKey('client.id'))
    sent_time = Column(DateTime, default=None)
    response_time = Column(DateTime, default=None)

    result = Column(Text, default=None)

    # 1. computing 2. computed 3. failed
    status = Column(String(10), default="computing")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
