
### pip install PyQt6

import matplotlib
import matplotlib.pyplot as plt

# ~ matplotlib.use('Agg')
matplotlib.use('QtAgg')
# ~ matplotlib.use('TkAgg')
# ~ matplotlib.use('SVG')

print(matplotlib.get_backend())
plt.plot((1, 4, 6))
plt.show()
