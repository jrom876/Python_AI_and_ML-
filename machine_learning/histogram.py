## Histogram
## https://www.w3schools.com/python/python_ml_mean_median_mode.asp
## https://machinelearningmastery.com/start-here/

import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 250)
plt.hist(x, 5)
plt.show()

# ~ y = numpy.random.uniform(0.0, 5.0, 100000)
# ~ plt.hist(y, 100)
# ~ plt.show()
