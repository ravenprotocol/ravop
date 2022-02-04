import json

import ravop as R

g = R.Graph(name="Mean of age data", approach="federated",
            rules=json.dumps({"rules": [{
                    "min": 10, "max": 60
                }, {
                    "min": 10, "max": 60
                }],
                "max_clients": 2}))

b = R.federated_mean()
b = R.federated_variance()
b = R.federated_standard_deviation()


# g2 = R.Graph(name="Mean of age data", approach="distributed")
# a = R.t([4, 5, 6])
# c = R.mean(a)
# g2.compile()

"""
1. Graph created
2. ops created
3. sub graphs not created 
4. subgraph client mappings not created

Scheduler
1. federated
2. distributed

subgraph - 1. Graph is too big to handle for a client - client(memory intesive)
           2. Parallelize - graph (sequential) or parallel

"""

