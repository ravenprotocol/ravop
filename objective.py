"""
1. Can contain multiple ops
2. They can be independent of each other
3.
"""
from ravop import Graph, t, create_op
from ravop import *


def execute_rule(rule):
    rule = "x > 0 && x < 100) || dtype(x) == float"
    x = 10
    if 0 < x < 100 or type(x) == float:
        pass


class Objective(Graph):
    def __init__(self, id=None, **kwargs):
        if id is None:
            g.graph_id = None
            super(Objective, self).__init__()
            self.__objective = ravdb.create_objective(id=self.id, graph_id=self.id, **kwargs)
        else:
            super(Objective, self).__init__(id=id)
            self.__objective = ravdb.get_objective(self.id)

        print(self.id, self.__objective)

        # op = create_op(status="computed", name="analytics", rules=json.dumps(rules))
        #
        # add(t(10), t(20))
        #

        """
        Objective: mean of age numbers
        
        1. age number cannot be in negative
        2. it should be less than hundred
    
        0 < x < 100 - true
        type 
        minimum
        maximum
        
        1. minimum
        2. maximum
        3. consistent data type
        4. 1d vector
        
        """
    def __str__(self):
        return "Objective-Id:{}\nStatus:{}\n".format(self.id, self.status)


if __name__ == '__main__':
    rules = {
        "lower-limit": 0,
        "upper-limit": 100,
        "type": "tensor",
        "rank": "0",
        "dtype": "int64",
        "participants": 3
    }

    Objective(name="Mean of ages", operator="mean", rules=json.dumps(rules))
    Objective(name="Variance of ages", operator="variance", rules=json.dumps(rules))
    Objective(name="Std of ages", operator="standard_deviation", rules=json.dumps(rules))

