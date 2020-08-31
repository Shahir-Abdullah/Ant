
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

import math


def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


style.use('fivethirtyeight')

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
filepath = "images/foodvstime.txt"

widthbar = .3


def animate(i):
    graph_data = open(filepath, 'r').read()
    lines = graph_data.split('\n')
    t = []
    food1 = []
    food2 = []
    food3 = []
    lifea = []
    lifeb = []
    lifec = []
    lifed = []
    lifee = []
    lifef = []
    lifeg = []
    lifeh = []
    lifej = []
    lifek = []
    ants = [7, 8, 5, 5, 7, 5, 4, 7, 9, 7]
    smellpowers = [7, 8, 5, 5, 7, 5, 4, 7, 9, 7]
    life = [20, 15.9, 10, 40, 30, 20, 35, 25, 15.6, 40]
    foodearned = []

    antsx = np.arange(len(ants))  # the label locations
    for line in lines:
        if len(line) > 1:
            y1, y2, y3, la, lb, lc, ld, le, lf, lg, lh, lj, lk, a, b, c, d, e, f, g, h, j, k, x = line.split(
                ',')
            t.append(float(x))

            food1.append(truncate(float(y1), 3))
            food2.append(truncate(float(y2), 3))
            food3.append(truncate(float(y3), 3))

            lifea.append(float(la))
            lifeb.append(float(lb))
            lifec.append(float(lc))
            lifed.append(float(ld))
            lifee.append(float(le))
            lifef.append(float(lf))
            lifeg.append(float(lg))
            lifeh.append(float(lh))
            lifej.append(float(lj))
            lifek.append(float(lk))

            foodearned.clear()

            foodearned.append(a)
            foodearned.append(b)
            foodearned.append(c)
            foodearned.append(d)
            foodearned.append(e)
            foodearned.append(f)
            foodearned.append(g)
            foodearned.append(h)
            foodearned.append(j)
            foodearned.append(k)

    '''axs[1, 1].clear()
    axs[1, 0].clear()'''
    axs[0].clear()
    axs[1].clear()
    '''axs[0, 1].clear()'''
    # plot time signal:
    axs[0].set_title("Time vs Food Amount")
    axs[0].plot(t, food1, t, food2, t, food3)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Food Amount")
    
    axs[1].set_title("Ant's Life")
    axs[1].plot(t, lifea, t, lifeb, t, lifec, t, lifed, t,
                   lifee, t, lifef, t, lifeg, t, lifeh, t, lifej, t, lifek)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Life")
    '''
    axs[1, 0].bar(antsx, foodearned, widthbar, label='Amount eaten')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[1, 0].set_ylabel('Food Earned')
    axs[1, 0].set_title('Smellpower vs Foodearned')
    axs[1, 0].set_xticks(antsx)
    axs[1, 0].set_xticklabels(ants)
    axs[1, 0].legend()

    axs[1, 1].bar(antsx, foodearned, widthbar, label='Food Earned')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[1, 1].set_ylabel('Food Earned')
    axs[1, 1].set_title('Initial Life vs Foodearned')
    axs[1, 1].set_xticks(antsx)
    axs[1, 1].set_xticklabels(ants)
    axs[1, 1].legend()
    '''
    fig.tight_layout()


ani = animation.FuncAnimation(fig, animate, interval=100)
fig.tight_layout()
plt.show()
