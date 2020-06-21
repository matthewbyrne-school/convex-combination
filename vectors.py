'''
Vectors
'''

# Imports
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from random import choice as ch
from numpy import sin, cos, tan, sqrt, pi
import numpy as np


# Timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        a = time.time()
        output = func(*args, **kwargs)
        b = time.time()

        print(f"{func.__name__} executed in {(b-a)*1000}s")
        return output

    return wrapper

# Main class
class Vector:
    dimensions = ["x", "y", "z", "t"]
    #@timer
    def __init__(self, *coordinates):
        self.x = 0
        self.y = 0
        self.z = 0
        self.t = 0

        self.coords = list(coordinates)
        self.dim = len(coordinates)

        for idx, dim in enumerate(Vector.dimensions):
            try:
                exec(f"self.{dim} = coordinates[{idx}]")

            except:
                break


    def __add__(self, other):
        if type(other) is int or type(other) is float:
            a = [other] + [0 for x in range(len(self.coords)-1)]

        if self.dim != other.dim:
            raise ValueError("Cannot add vectors of different dimensions")
        
        new = []

        for i, j in zip(self.coords, other.coords):
            new.append(i+j)

        return Vector(*new)

    def __getitem__(self, name):
        return self.coords[name]

    def __mul__(self, other):
        if type(other) is int or type(other) is float:
            return Vector(*[coord*other for coord in self.coords])

        elif type(other) is Vector:
            new = Vector(*[0 for _ in self.coords])
            for idx, i in enumerate(self.coords):
                new.coords[idx] = i*other[idx]

            return new

        elif type(other) is Matrix:
            return other * self

    def dot(self, other):
        total = 0
        for i, j in zip(self.coords, other.coords):
            total += i*j 

        return total

    #@timer
    def __str__(self):
        output = {}

        if self.x:
            output["i"] = self.x

        if self.y:
            output["j"] = self.y

        if self.z:
            output["k"] = self.z

        if self.t:
            output["t"] = self.t

        return " + ".join(map(str, [f"{v}{k}" for k,v in output.items()])).replace("+ -", "- ")

    def __repr__(self):
        return f"<{', '.join(map(str, self.coords))}>"

    def __pow__(self, power):
        product = self

        for _ in range(power-1):
            product = self * product

        return product

    def __truediv__(self, other):
            a = self * other
            b = other ** 2

            return a/b

    def magnitude(self):
        return sqrt(sum([i**2 for i in self.coords]))

    def __int__(self):
        return int(self.magnitude())

    def __float__(self):
        return self.magnitude()

    def plot(self, colour="r", start=(0, 0), figure=plt):
    	#figure.arrow(start[0], start[1], self.x, self.y, color=colour, length_includes_head=True, head_width=0.25, head_length=0.5)
        figure.plot([start[0], self.x], [start[1], self.y], color=colour)

    def vignette(self, figure=plt):
        X, Y = np.meshgrid(np.linspace(-100 * self.x, 100 * self.x, 5 * self.x), np.linspace(-100 * self.y, 100 * self.y, 5 * self.y))
        U, V = self.x * X, self.y * Y
        figure.quiver(X, Y, U, V)

    def quiver(self, figure=plt):
        X, Y = np.meshgrid(np.linspace(-100 * self.x, 100 * self.x, 5 * self.x), np.linspace(-100 * self.y, 100 * self.y, 5 * self.y))
        U, V = self.x, self.y
        figure.quiver(X, Y, U, V)


# Algebraic vector 3D
class AVector:
    def __init__(self, x:str, y:str, z:str):
        self.x = x
        self.y = y
        self.z = z

    def quiver(self, figure):
        X, Y, Z = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
        U = eval(self.x.replace("x", "X"))
        V = eval(self.y.replace("y", "Y"))
        W = eval(self.z.replace("z", "Z"))
        figure.quiver(X, Y, Z, U, V, W)


    # Evaluating the endpoint of the vector
    def evaluate(self, x:int, y:int, z:int):
        X = eval(self.x)
        Y = eval(self.y)
        Z = eval(self.z)

        return (X, Y, Z)


# Matrix
class Matrix:
    def __init__(self, *rows):
        self.matrix = list(rows)
        self.m = len(self.matrix)
        self.n = len(self.matrix[0])

    def __add__(self, other):
        if self.m != other.m or self.n != other.n:
            raise ValueError("Cannot add matrices of different dimensions")

        new = []

        for idx, row in enumerate(self.matrix):
            new.append([])
            for jdx, item in enumerate(row):
                new[idx].append(item + other.matrix[idx][jdx])

        return Matrix(*new)

    def __getitem__(self, name):
        return self.matrix[name]

    def __str__(self):
        output = "Matrix:"
        for row in self.matrix:
            output += "\n" + ", ".join(map(str, row))

        return output

    def rotate(self):
        new = Matrix(*[[0 for _ in self.matrix] for _ in self.matrix[0]]) # creating a new matrix with dimensions n*m rather thnan m*n

        for m, row in enumerate(self.matrix):
            for n, col in enumerate(row):
                new[n][m] = col # takes coord (m, n) of old matrix and maps it to (n, m) of new matrix

        return new


    def __mul__(self, other):
        if type(other) is int:
            new = []
            for idx, row in enumerate(self.matrix):
                new.append([])
                for jdx, item in enumerate(row):
                    new[idx].append(item*other)
            return Matrix(*new)

        elif type(other) is Vector:
            
            coords = []

            for row in self.matrix:
                total = 0
                
                for coord, item in zip(other.coords, row):
                    total += coord*item
                
                coords.append(total)

            return Vector(*coords)


        elif type(other) is Matrix:
            if self.n == other.m:
                new = Matrix(*[[0 for _ in other.matrix[0]] for _ in other.matrix])

                rotated = other.rotate()

                for i in range(new.m):
                    for j in range(new.n):
                        total = 0

                        for x, y in zip(self[i], rotated[j]):
                            total += x*y

                        new[i][j] = total
            
                return new


            elif self.m == other.n:
                return other * self

            else:
                raise ValueError("Cannot multiply two matrices of incompatable dimensions")
        
        


if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    v3 = AVector("sin(x)", "cos(y)", "tan(z)")
    v3.quiver(ax)

    a, b, c = v3.evaluate(1, 1, 1)
    print(a, b, c)

    ax.set_xlabel("sin(x)")
    ax.set_ylabel("cos(y)")
    ax.set_zlabel("tan(z)")


    plt.show()


