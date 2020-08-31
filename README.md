# Ant's movement and their way of searching for food visualized by creating Ant agent and ploting in matplotlib

![](images/summary.gif)

I created Simple reflex agents to replicate ants and their behavior. The agent smells food and takes the food back to the nest. While returning they drop a chemical trail which lets other ants to sniff and find the food source.

Although simple reflex agent but collectively they act in a very sophisticated manner.

The more ants go to a food source they reinforce the chemical trail.

I used several death rate and life expectency to see their behavior.

It's harder to create a stable trail if the food is distant.

# Simulation

![](images/simulation.gif)

# Data visualization

![](images/datavisualization.gif)

# Testing different types of death rate

1. Death rate 5 units per 100 milisecond
   ![](images/deathrate5.gif)

1. Death rate 2 units per 100 milisecond
   ![](images/deathrate2.gif)

1. Death rate 1.5 units per 100 milisecond
   ![](images/deathrate1.5.gif)

1. Death rate 1 units per 100 milisecond
   ![](images/deathrate1.gif)

# Files

1. 'ant.py' is the main file.
2. For finding the shortest distance, 'ant.py' uses 'search.py' file. Searching is done by "A star" path finding algorithm. 'search.py' takes the current coordinate and destiantion coordinate as the input, and returns the list of coordinates as the path.
3. 'dataplot.py' is a script to plot the data in matplotlib. It uses the 'foodvstime.txt' file as input. 'foodvstime.txt' is being updated every 100 milisecond by the 'ant.py' script.

# Instructions

1. To run the simulation, just type 'python ant.py' in terminal.
2. To see real time data visualization, simultaneously run the comman 'python dataplot.py' on another terminal.
