import Fuzzy as fz
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# This is a demo of the Fuzzy code
# It creates some fuzzy sets, does some implications, creates 2 Takagi Sugeno Systems and calculates output from output that I had done by hand and from the notes
# Does some plotting
# This was done to make sure that my fuzzy sets were implemented properly
# If changed to True, will do all of the above

if False:
    a = fz.FuzzySet(2,6,10)
    b = fz.FuzzySet(3,4,5,7)

    c = fz.implicate(a,b,lambda x,y : min((1 - x + y),1)) # Lukasiewicz
    d = fz.implicate(a,b,lambda x,y : min(x,y)) # Larson
    e = fz.implicate(a,b,lambda x,y : x * y) # Goguen

    fig = plt.figure()
    a.plot("Basic A set")
    b.plot("Basic B set")
    c.plot("Lukasiewicz Implication")
    #d.plot("Larson Rule")
    #e.plot("Goguen Implication")
    plt.legend()
    plt.show()

    a1 = fz.FuzzySet(1,2,3)
    a2 = fz.FuzzySet(2,4,6)

    new_fig = plt.figure()
    a1.plot()
    a2.plot()
    plt.show()

    ts = fz.TakagiSugeno()
    ts.add_fuzzy_input('A',a1)
    ts.add_fuzzy_input('A',a2)
    ts.add_output_rule(lambda x : 2 * x - 2)
    ts.add_output_rule(lambda x : x + 1)
    print(ts(2.5))
    print(ts)


    a3 = fz.FuzzySet(2,4,6)
    b3 = fz.FuzzySet(4,5,7)
    a4 = fz.FuzzySet(4,5,8)
    b4 = fz.FuzzySet(3,4,6)

    ts1 = fz.TakagiSugeno()
    ts1.add_fuzzy_input('A',a3)
    ts1.add_fuzzy_input('B',b3)
    ts1.add_fuzzy_input('A',a4)
    ts1.add_fuzzy_input('B',b4)
    ts1.add_output_rule(lambda x,y : 2 * x - y + 1)
    ts1.add_output_rule(lambda x,y : x - y + 2)
    print(ts1(5,4.5))

    print("\n\n\n")
    print(ts1)

    thing = ts1.plot(25,5)
    plt.show()

# This is the real section of the project
# We setup the function, then approximate it
# Then we plot it to look at it

t = fz.TakagiSugeno()
t.add_fuzzy_input('a',fz.FuzzySet(1,2,3))
t.add_fuzzy_input('a',fz.FuzzySet(2,3,5))
t.add_fuzzy_input('b',fz.FuzzySet(2,4,6))
t.add_fuzzy_input('b',fz.FuzzySet(4,6,10))
t.add_output_rule(lambda x,y: 1.5 * x + .75 * y + 1)
t.add_output_rule(lambda x,y: 4/3 * x + 2/3 * y + 1)

print(t(2.5,5))

# Function to approximate
# Some functions I found interesting to look at
#func = lambda x,y : np.sin(x * y) # Use n = 50 and d = 1 for better results
#func = lambda x,y : np.tanh(x * y)
#func = lambda x,y : np.sinh(x * y) # Use n = 50 and d = 1 for better results
func = lambda x,y : x * y
#func = lambda x,y : np.sin(x/y) if y else 0
#func = lambda x,y : np.sin(x) * np.cos(y) # Use n = 50 and d = 1 for better results

# Creates a Takagi Sugeno system for the function passed
# Works with any dimension function
ts = fz.approximate_function(func)

# Also can take a specific domain max (and min) and step size
#ts = fz.approximate_function(func,1,.01)


# Plots the Takagi Sugeno system
# Function works like this:
# The first number is how many points to generate; in our case it will be n^2 since there will be n points in each domain for x and y
# The second number is the domain max (and min) to plot from
# Decreasing the domain size increases speed of the plot, but leads for more jarring edges

n = 50
d = 5

ts.plot(n,d)
plt.show()

print(ts)
