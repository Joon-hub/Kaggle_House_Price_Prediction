import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title('Simple Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()
