# nel file __init__.py del tuo modulo

import matplotlib.pyplot as plt
from cycler import cycler

# Prendi il ciclo colori attuale
current_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Modifica i primi 3 colori a piacere (esempio con rosso, verde, blu)
new_colors = ['#0000ff', '#ff0000', '#00ff00'] + current_colors[3:]

# Imposta il nuovo ciclo colori
plt.rcParams['axes.prop_cycle'] = cycler(color=new_colors)
