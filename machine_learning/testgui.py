
### pip install PyQt6

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')

print(matplotlib.get_backend())
plt.plot((1, 4, 6))
plt.show()
