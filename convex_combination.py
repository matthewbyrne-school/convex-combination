'''
Convex Combination
'''

# Imports
from vectors import Vector
import tkinter as tk
from random import random as rand, randint as r
import matplotlib.pyplot as plt
from math import acos, pi

# Convex Combination class
class Convex_Combination:
	def __init__(self, vectA, vectB):
		self.A, self.B = vectA, vectB
	
	def getVector(self, α, β):
		if (α+β) == 1 and α>=0 and α>=0: return self.A*α + self.B*β
		else: raise ValueError("Values for α & β in Convex Combination are invalid.")
	
	def randomVector(self): return self.getVector(r:=rand(), 1-r)

	def randomVectorSample(self, n):
		h = []
		
		for _ in range(n):
			h.append(self.randomVector())

		return h

	def getAngle(self):
		x = self.A.dot(self.B)
		y = float(self.A) * float(self.B)
		return acos(x/y) 



# Main Bit
if __name__ == "__main__":
	root = tk.Tk()

	def generate():
		a = Vector(int(x1.get()), int(y1.get())) # Vector A is sampled from left column of inputs
		b = Vector(int(x2.get()), int(y2.get())) # Vector B is sampled from right column of inputs

		#print(a, b)

		c = Convex_Combination(a, b)

		C = c.randomVectorSample(4)
		try:
			plt.clf()

		except:
			pass

		for x in C:
			x.plot()

		a.plot(colour="b")
		b.plot(colour="b")

		plt.plot([a.x, b.x], [a.y, b.y], "b--")
		plt.title("Result")

		plt.show()

	xLabel = tk.Label(root, text="x")
	X = tk.Label(root, text="x")
	yLabel = tk.Label(root, text="y")
	Y = tk.Label(root, text="y")

	aLabel = tk.Label(root, text="A")
	bLabel = tk.Label(root, text="B")

	x1 = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL)
	y1 = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL)

	x2 = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL)
	y2 = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL)
	b1 = tk.Button(root, text="Go", command=generate)
	
	aLabel.grid(row=0, column=0); bLabel.grid(row=0, column=2)
	x1.grid(row=1, column=0); X.grid(row=1, column=1); x2.grid(row=1, column=2)
	y1.grid(row=2, column=0); Y.grid(row=2, column=1); y2.grid(row=2, column=2)
	b1.grid(row=3, column=1)

	root.title("VCC Settings")

	root.mainloop()