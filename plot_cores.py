import numpy as np
import matplotlib.pyplot as plt

times = [29.501, 17.575, 11.423, 8.906, 9.296, 9.224, 8.587, 9.831]
cores = [1, 2, 3, 4, 5, 6, 7, 8]

plt.plot(cores, times)
plt.xlabel("Number of cores")
plt.ylabel("Time, s")
plt.grid(True)
for i in range(7):
    plt.text(cores[i],times[i],str(times[i]))
plt.show()
