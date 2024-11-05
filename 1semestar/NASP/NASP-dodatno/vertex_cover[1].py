from pulp import *

vertices=['a','b','c','d', 'e']
costs = {'a':2,'b':1,'c':1,'d':1,'e':1}

x = LpVariable.dicts('x', vertices, 0, 1, LpContinuous)#, LpInteger)

# The variable 'prob' is created
prob = LpProblem("vertex_cover", LpMinimize)

# The objective function is entered: the total number of large rolls used * the fixed cost of each
prob += lpSum([x[i] * costs[i] for i in vertices]), "Cover cost"

prob += x['a']+x['b']>=1
prob += x['a']+x['c']>=1
prob += x['a']+x['d']>=1
prob += x['a']+x['e']>=1
prob += x['b']+x['c']>=1

prob.solve()

for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen
print("Cover Costs = ", value(prob.objective))

