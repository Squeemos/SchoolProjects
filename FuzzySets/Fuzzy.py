import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import inspect
import re

def is_number(s : object) -> bool:
    '''Checks to see if the input is a number or not'''
    try:
        float(s)
        return True
    except ValueError:
        return False

class FuzzySet(object):
    def __init__(self,*args,**kwargs):
        '''Creates a fuzzy set from a set of inputs/outputs
           Will contain members:
              - type   : What type of fuzzy set. Empty/Singleton/Interval/Triangular/Trapezoidal/Function
              - lr     : If the set is defined with an LR function
              - pt     : The points of the domain, different based on type
              - func   : Function for output
              - l_func : The function for the left
              - r_func : The function for the right'''
        if len(args) == 0:
            self.type = "empty"
            self.lr = False
            self.func = lambda x : 0
        elif len(args) == 1:
            # Only 1 argument passed
            if callable(args[0]):
                self.type = "function"
                self.lr = False
                self.func = args[0]
            elif is_number(args[0]):
                self.type = "singleton"
                self.lr = False
                self.pt = args[0]
                self.func = lambda x: 1 if x == self.pt else 0
        elif len(args) == 2:
            self.type = "interval"
            self.lr = False
            self.pt = args[0:2]
            self.func = lambda x: 1 if x >= self.pt[0] and x <= self.pt[1] else 0
        elif len(args) == 3:
            self.type = "triangular"
            self.lr = True
            self.pt = args[0:3]
            self.l_func = lambda x: (x - self.pt[0]) / (self.pt[1] - self.pt[0])
            self.r_func = lambda x: (self.pt[2] - x) / (self.pt[2] - self.pt[1])
            if "L" in kwargs:
                self.l_func = lambda x : kwargs["L"]((x - self.pt[0]) / (self.pt[1] - self.pt[0]))
            if "R" in kwargs:
                self.r_func = lambda x : kwargs["R"]((self.pt[2] - x) / (self.pt[2] - self.pt[1]))
        elif len(args) == 4:
            self.type = "trapezoidal"
            self.lr = True
            self.pt = args[0:4]
            self.l_func = lambda x: (x - self.pt[0]) / (self.pt[1] - self.pt[0])
            self.r_func = lambda x: (self.pt[3] - x) / (self.pt[3] - self.pt[2])
            if "L" in kwargs:
                self.l_func = lambda x : kwargs["L"]((x - self.pt[0]) / (self.pt[1] - self.pt[0]))
            if "R" in kwargs:
                self.r_func = lambda x : kwargs["R"]((self.pt[3] - x) / (self.pt[3] - self.pt[2]))

    def __call__(self,input):
        '''Gives the output given from input'''
        output = 0
        if self.lr:
            if self.type == "triangular":
                if input >= self.pt[0] and input <= self.pt[1]:
                    output =  self.l_func(input)
                elif input >= self.pt[1] and input <= self.pt[2]:
                    output =  self.r_func(input)
                else:
                    output = 0
            elif self.type == "trapezoidal":
                if input >= self.pt[0] and input <= self.pt[1]:
                    output =  self.l_func(input)
                elif input >= self.pt[1] and input <= self.pt[2]:
                    output =  1
                elif input >= self.pt[2] and input <= self.pt[3]:
                    output =  self.r_func(input)
                else:
                    output =  0
        else:
            output = self.func(input)
        if output >= 1:
            return 1
        elif output <= 0:
            return 0
        else:
            return output

    def __str__(self):
        if self.type == "empty":
            return f"(-∞,∞)"
        elif self.type == "function":
            return f"Not done"
        elif self.type == "triangular":
            return f"{tuple(self.pt)}"
        elif self.type == "singleton":
            return f"({self.pt})"
        elif self.type == "trapezoidal":
            return f"{tuple(self.pt)}"
        elif self.type == "interval":
            return f"{tuple(self.pt)}"

    def plot(self,label=None):
        '''Not well put together yet.
           Plots the fuzzy set with some domain
           Can be used to plot multiple together'''
        ax = plt.gca()
        if self.type == "empty":
            domain = np.linspace(-25,25,1000)
            range = np.linspace(0,0,1000)
        elif self.type == "singleton":
            domain = np.linspace(self.pt - 5, self.pt + 5,1000)
            range = np.linspace(0,0,100)
            plt.scatter(self.pt,1)
        elif self.type == "interval":
            spread = self.pt[1] - self.pt[0]
            domain = np.linspace(self.pt[0] - spread,self.pt[1] + spread)
            range = np.array([self.__call__(x) for x in domain])
        elif self.type == "triangular":
            spread = self.pt[2] - self.pt[0]
            dom = np.linspace(self.pt[0] - spread,self.pt[2] + spread,1000)
            domain = np.append(dom,self.pt[1])
            domain = np.sort(domain)
            range = np.array([self.__call__(x) for x in domain])
        elif self.type == "trapezoidal":
            spread = self.pt[3] - self.pt[0]
            domain = np.linspace(self.pt[0] - spread,self.pt[3] + spread,1000)
            range = np.array([self.__call__(x) for x in domain])
        elif self.type == "function":
            domain = np.linspace(-10,10,1000)
            range = np.array([self.__call__(x) for x in domain])
        plt.plot(domain,range,label=label)
        return ax

def implicate(first,second,func):
    '''Implicates two fuzzy sets with a function
       func should have 2 variables'''
    assert func.__code__.co_argcount == 2, f"Passed function should have 2 arguments. Got {func.__code__.co_argcount}."
    return FuzzySet(lambda a: func(first(a), second(a)))

class TakagiSugeno(object):
    def __init__(self):
        '''Generates the containers for the inputs and outputs'''
        self.inputs = {}
        self.outputs = []

    def add_fuzzy_input(self,key,input):
        '''Adds a fuzzy input with specific input'''
        if key not in self.inputs:
            self.inputs[key] = []
        self.inputs[key].append(input)

    def add_output_rule(self,output):
        '''Adds an output rule to the system
           Can be a function (typically piece-wise linear) or constants (piece-wise constant)
           Do not mix constants and functions though, since they are handled separately'''
        self.outputs.append(output)


    def __call__(self,*args):
        '''Obtain the crisp output from the Takagi Sugeno System'''

        assert len(self.outputs) != 0, "You must have output rules to get an output"
        assert len(self.inputs) != 0, "You must have input rules to get an output"

        firing_levels = []
        function_outputs = []

        if callable(self.outputs[0]):
            for level in range(len(self.inputs[list(self.inputs.keys())[0]])): # What 'i' we are at
                # Current firing levels for this 'i' value
                # Gives each fuzzy set the proper input from the args
                current_level_firing = [self.inputs[key][level](input) for key,input in zip(self.inputs.keys(),args)]
                firing_levels.append(min(current_level_firing)) # Add the minimum firing level

                function_outputs.append(self.outputs[level](*args)) # Add the function output of this level
        else:
            for level in range(len(self.inputs[list(self.inputs.keys())[0]])): # What 'i' we are at
                # Current firing levels for this 'i' value
                # Gives each fuzzy set the proper input from the args
                current_level_firing = [self.inputs[key][level](input) for key,input in zip(self.inputs.keys(),args)]
                firing_levels.append(min(current_level_firing)) # Add the minimum firing level

                function_outputs.append(self.outputs[level]) # Add the output of this level

        # Defuzzification
        top = np.sum([a*b for a,b in zip(firing_levels,function_outputs)])
        bot = np.sum(firing_levels)
        np.seterr('raise') 
        try:
            return top/bot
        except ZeroDivisionError:
            return 0
        except FloatingPointError:
            return 0

    def plot(self,n,loc):
        '''Only works in 2 dimensions for now'''
        x = np.linspace(-loc,loc,n)
        y = np.linspace(-loc,loc,n)
        X,Y = np.meshgrid(x,y)
        Z = np.array([[self(u,v) for v in y] for u in x]).flatten().reshape((n,n))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z,antialiased=False)
        return ax

    def __str__(self):
        string = f""
        for index in range(len(self.inputs[list(self.inputs.keys())[0]])):
            string += f"If "
            for key in self.inputs:
                string += f"{key} is {self.inputs[key][index]} "

            if callable(self.outputs[index]):
                base_function = str(inspect.getsource(self.outputs[index]))
                if 'lambda' in base_function:
                    current_string = str(re.split('[:]',base_function)[1])
                    current_string = current_string[:-2]
                    string += f"then output is{current_string}\n"
            else:
                string += f"then output is {self.outputs[index]}\n"
        return string

def approximate_function(func,loc=5,step=.25):
    '''Given a function, approximate it using a Takagi Sugeno System
       The loc is the max domain number. Generates a min of the domain with -loc. Domain looks like [-loc,loc]
       Step is how big the step size is in the domain'''
    ts = TakagiSugeno()

    # How many inputs the function has
    num_domains = func.__code__.co_argcount
    # Genereates the domains for each argument in the function
    domains = [np.arange(-loc,loc,step) for _ in range(num_domains)]

    # All possible combinations of domain inputs
    combinations = list(itertools.product(*domains))

    # For every combination in the domain
    for pair in combinations:
        # For each argument in the function
        for index,arg in enumerate(func.__code__.co_varnames):
            # Add a fuzzy set with the key of the argument name
            # Creates a triangular fuzzy set with values : (a - step, a, a + step)
            ts.add_fuzzy_input(arg,FuzzySet(pair[index]-step,pair[index],pair[index]+step))
        # Adds crisp output for the current layer (A_1, B_1, etc) based on the function
        ts.add_output_rule(func(*pair))
    return ts
