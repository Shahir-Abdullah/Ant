import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import search
import random
import time
import math

# fixing random state for reproducibility
np.random.seed(19680801)

# truncate function


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


# txt file to store the data
filepath = 'data/foodvstime.txt'

fig, (ax) = plt.subplots(1, 1, figsize=(10, 10))
# 20X20 grid
N = 20

# grid class


class Grid:

    grid = np.zeros((N, N))

    @staticmethod
    def display():  # currently not using
        colors = np.random.rand(N)
        # for i in range(0, N):
        #print(np.arange(0,N,1), Grid.grid[i,0:N])
        #ax1.scatter(np.full((1, N), i),Grid.grid[i,0:N])

    @staticmethod
    # initializing the given cell to a smell value
    def smellarea(x, y, smellradius, smellpower):
        x1 = x - smellradius  # starting from the top left corner
        y1 = y - smellradius
        x2 = x + smellradius
        y2 = y + smellradius

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > 19:
            x2 = 19
        if y2 > 19:
            y2 = 19

        for i in range(x1, x2+1):
            for j in range(y1, y2+1):
                Grid.grid[i, j] = smellpower

    @staticmethod
    # the ant calls this function in order to select the cell with the highest smell value
    def smellzone(x, y, smellradius):

        x0 = x - smellradius
        y0 = y - smellradius
        x1 = x + smellradius
        y1 = y + smellradius

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > 19:
            x1 = 19
        if y1 > 19:
            y1 = 19

        maxvalue = -100
        maxx = 0
        maxy = 0

        for i in range(x0, x1+1):
            for j in range(y0, y1+1):
                if i == x or j == y:
                    continue
                else:
                    if Grid.grid[i, j] > maxvalue:
                        maxvalue = Grid.grid[i, j]
                        maxx = i
                        maxy = j
                        # print("maxvalue ", maxvalue, " maxx ", maxx, " maxy ", maxy)

        return maxx, maxy


class House:  # nest class
    def __init__(self, x, y, r, color):
        self.x = x
        self.y = y
        self.r = r
        self.color = color

    def spawn(self):  # ploting
        circle = plt.Circle((self.x, self.y), self.r, color=self.color)
        ax.add_artist(circle)


class Sugar:  # the food source class, has a center x,y and radius and color

    def __init__(self, x, y, r, color):
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.evaporation_rate = .1  # it's actually decomposition rate
        self.foodamount = 10  # amount of food
        self.foodradius = 1  # radius of the foodsource
        self.smellfactor = 3  # smell factor is the smell value, 3 means the highest
        self.foodr = True  # a bool value to know the ant's has entered the radius of the food
        # adjecent food with more amount but less taste
        Grid.smellarea(self.x, self.y, self.foodradius, self.smellfactor-1)
        # center food with less amount but more taste
        Grid.smellarea(self.x, self.y, self.foodradius-1, self.smellfactor)

    def foodeaten(self):  # function called each time a small amount of food is being eaten by the and

        if self.foodamount > 0:

            self.foodamount -= .5  # value reduces by .5
            Grid.smellarea(self.x, self.y, self.foodradius, self.smellfactor -
                           self.evaporation_rate-.1)  # less tasty food eaten less
            Grid.smellarea(self.x, self.y, self.foodradius-1,
                           self.smellfactor-self.evaporation_rate)  # tastier food first
            self.smellfactor = self.smellfactor - self.evaporation_rate
        else:
            # negating the smell value so that the ant's don't get stuck into local maxima when the food is finished,
            Grid.smellarea(self.x, self.y, 2, -1)
            Grid.smellarea(self.x, self.y, 1, -2)
            self.foodr = False  # i forgot why i did that

    # decomposition function, the radius gets decreased, it's different than foodeaten because it's for the plotting (visual part)
    def decomposition(self, lifespan):
        if lifespan == 200:
            print("food destroyed\n")
            Grid.smellarea(self.x, self.y, self.foodradius, -1)

    def spawn(self):  # plot function
        if self.foodr == True:
            circle = plt.Circle((self.x, self.y), self.r, color=self.color)
            ax.add_artist(circle)
        else:
            if self.r > 0:
                self.r -= .1
            else:
                self.r = 0
            circle = plt.Circle((self.x, self.y), self.r, color=self.color)
            ax.add_artist(circle)


class Smell:  # the smell class
    evaporation_rate = .0001  # evaporation rate
    xpath = []  # in xpath all the travelled x coordinates are stored
    ypath = []  # in xpath all the travelled y coordinates are stored
    diffusedx = []  # the diffused x coordinates which are adjacent to the travelled cells, this will get smell value but less
    diffusedy = []
    # not necesseray, used it to store the lengths of paths each time food was found
    pathlength = []
    diffusion_rate = .001  # diffusion rate of the chemical trails
    trailgone = 80  # this is the amount of diffused cells that are gonna be removed after the smell value decreases to a certain value

    @staticmethod
    def drawpath():  # plotting

        ax.scatter(Smell.xpath, Smell.ypath, 75, "#FF6B6B", alpha=0.5)

    @staticmethod
    # called by the ant class each time food is found, it adds the cell to draw the trail path
    def addpath(path=None, x=None, y=None):
        Smell.pathlength.append(len(path))  # not necessary now
        if path != None:
            for coordinates in path:
                Smell.xpath.append(coordinates[0])
                Smell.ypath.append(coordinates[1])
                Smell.diffusion(coordinates[0], coordinates[1])
        if x != None:
            Smell.xpath.append(x)
        if y != None:
            Smell.ypath.append(y)

    @staticmethod
    def evaporate():
        if len(Smell.xpath) > 0:
            for i in range(0, len(Smell.xpath)-1):
                Grid.grid[Smell.xpath[i], Smell.ypath[i]
                          ] -= Smell.evaporation_rate

            xlen = len(Smell.xpath)
            # once the smell values of a cell reach a certain point, all the cells are assigned negative smell value representing decomposition
            for i in range(0, xlen-1):
                if Grid.grid[Smell.xpath[i], Smell.ypath[i]] <= -0.001:
                    Smell.xpath[i] = -1
                    Smell.ypath[i] = -1

            copyxpath = []
            copyypath = []

            for i in range(0, xlen-1):
                if Smell.xpath[i] != -1:
                    copyxpath.append(Smell.xpath[i])
                    copyypath.append(Smell.ypath[i])

            Smell.xpath = copyxpath
            Smell.ypath = copyypath

    @staticmethod
    def diffusion(x, y):  # diffusion function, x, y's adjacent cells are assigned a smell value
        Smell.diffusedx.append(x)
        Smell.diffusedy.append(y)
        if x+1 <= 19 and y+1 <= 19:
            Grid.grid[x+1, y+1] += Smell.diffusion_rate
            Smell.diffusedx.append(x+.3)
            Smell.diffusedy.append(y+.3)
        if x+1 <= 19 and y-1 >= 0:
            Grid.grid[x+1, y-1] += Smell.diffusion_rate
            Smell.diffusedx.append(x+.3)
            Smell.diffusedy.append(y-.3)
        if x-1 >= 0 and y+1 <= 19:
            Grid.grid[x-1, y+1] += Smell.diffusion_rate
            Smell.diffusedx.append(x-.3)
            Smell.diffusedy.append(y+.3)
        if x-1 >= 0 and y-1 >= 0:
            Grid.grid[x-1, y-1] += Smell.diffusion_rate
            Smell.diffusedx.append(x-.3)
            Smell.diffusedy.append(y-.3)

    @staticmethod
    def spawdiffusion():  # plotting of the diffused cells

        ax.scatter(Smell.diffusedx, Smell.diffusedy, 55, "#FF6B6B", alpha=0.3)

    @staticmethod
    def evaporateall():  # this is called in order to decrease the smell value of the diffused cell, evaporations of the cells

        while len(Smell.diffusedx) >= Smell.trailgone:
            if Smell.diffusedx:
                Smell.diffusedx.pop(0)
                Smell.diffusedy.pop(0)


class Ant:  # ant class
    # initialized with a coordinate and smell power which is a radius of it's sensory nerve, a life
    def __init__(self, x, y, smellpower, life):
        self.x = x
        self.y = y
        self.destx = None  # finds the highest cell with smell value and assigns dest
        self.desty = None
        self.smellpower = smellpower
        self.food = False  # food found bool value
        self.life = life
        self.foodearned = 0  # amount of food eaten

    def smellsetzero(self):  # not using
        self.smellx = None
        self.smelly = None

    def randomwalk(self):  # not necessary, used it to test random movement

        self.destx = random.randint(self.x, self.x+self.smellpower)
        self.desty = random.randint(self.x, self.x+self.smellpower)

        if self.destx < 0:
            self.destx = 1
        if self.destx > 19:
            self.destx = 18
        if self.desty < 0:
            self.desty = 1
        if self.desty > 19:
            self.desty = 18

        path = search.main(self.x, self.y, self.destx, self.desty)

        path.reverse()
        return path

    def foundfood(self, sugar):# food searching function, called each time, if food found, sets the destination to home, else finds the next best smell value cell

        if self.food == True:  # found the food
            self.destx = 9
            self.desty = 10
            self.life += 50
            self.foodearned += 1
            self.food = False
            # the search function which runs a star searching to fiind the shortest path
            path = search.main(self.x, self.y, self.destx, self.desty)
            path.reverse()
            Smell.addpath(path)
        else:
            

            self.destx, self.desty = Grid.smellzone(
                self.x, self.y, self.smellpower)
            # print(self.destx, self.desty, "\n")
            if self.destx < 0:
                self.destx = random.randint(1, 5)
            if self.destx > 19:
                self.destx = random.randint(16, 19)
            if self.desty < 0:
                self.desty = random.randint(1, 5)
            if self.desty > 19:
                self.desty = random.randint(16, 19)

            '''print(self.x, self.y, self.destx, self.desty)'''
            # check funtion to know it has arrived at the destination or not, for the plot
            if (self.destx-sugar[0].x)**2 + (self.desty-sugar[0].y)**2 <= sugar[0].r**2:
                '''print("foound food 1")'''
                sugar[0].foodeaten()
                self.food = True
            elif (self.destx-sugar[1].x)**2 + (self.desty-sugar[1].y)**2 <= sugar[1].r**2:
                '''print("found food 2")'''
                sugar[1].foodeaten()
                self.food = True
            elif (self.destx-sugar[2].x)**2 + (self.desty-sugar[2].y)**2 <= sugar[2].r**2:
                '''print("found food 3")'''
                sugar[2].foodeaten()
                self.food = True

            path = search.main(self.x, self.y, self.destx, self.desty)

        if path == None:
            print("we are at the destination")
            exit(1)

        # time.sleep(.08)
        path.reverse()
        return path




house = House(9, 10, 2, 'b')  # x, y, radius, color

sugar1 = Sugar(3, 10, 1.2, 'r')  # x, y, radius, color
sugar2 = Sugar(17, 3, 1.4, 'r')
sugar3 = Sugar(8, 18, 1.4, 'r')

a = Ant(9, 10, 7, 20)  # x, y, smellpower, lifespan
b = Ant(10, 9, 8, 15)
c = Ant(10, 10, 5, 10)
d = Ant(11, 11, 5, 40)
e = Ant(10, 9, 7, 30)
f = Ant(10, 9, 5, 20)
g = Ant(9, 11, 4, 35)
h = Ant(9, 10, 7, 25)
j = Ant(10, 9, 9, 15)
k = Ant(10, 10, 7, 40)

patha = []
pathb = []
pathc = []
pathd = []
pathe = []
pathf = []
pathg = []
pathh = []
pathk = []
pathj = []


p = (3, 1)  # dummy value
pb = (4, 2)
pc = (1, 1)
pd = (1, 1)
pe = (4, 2)
pf = (1, 1)
pg = (1, 1)
ph = (4, 2)
pk = (1, 1)
pj = (1, 1)

foodlifespan = 1
gridshow = 0

start = time.time()
eva = 100


def animate(i):

    ax.clear()
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])

    global patha, pathb, pathc, pathd, pathe, pathf, pathg, pathh, pathk, pathj
    global p, pb, pc, pd, pe, pf, pg, ph, pk, pj
    global foodlifespan

    a.life -= 1 #life decreases by 1 each 100 milisecond 
    b.life -= 1
    c.life -= 1
    d.life -= 1
    e.life -= 1
    f.life -= 1
    g.life -= 1
    h.life -= 1
    k.life -= 1
    j.life -= 1

    if patha and a.life > 0:  # when food found, patha is the path to home from the food source
        p = patha[-1]
        patha.pop()
        a.x = p[0]
        a.y = p[1]

    elif a.life > 0:  # food searhing
        patha = a.foundfood([sugar1, sugar2, sugar3])

    else:  # ant's life has ended
        print("ant 1 died....... :( \n")

    if pathb and b.life > 0:
        pb = pathb[-1]
        pathb.pop()
        b.x = pb[0]
        b.y = pb[1]
        '''Grid.grid[b.x, b.y] = Grid.grid[b.x, b.y] + .01
        Smell.diffusion(b.x, b.y)'''

    elif b.life > 0:
        pathb = b.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 2 died........ :( \n")

    if pathc and c.life > 0:
        pc = pathc[-1]
        pathc.pop()
        c.x = pc[0]
        c.y = pc[1]
        '''Grid.grid[c.x, c.y] = Grid.grid[c.x, c.y] + .01
        Smell.diffusion(c.x, c.y)'''
    elif c.life > 0:
        pathc = c.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 3 died........... :( \n")

    if pathd and d.life > 0:
        pd = pathd[-1]
        pathd.pop()
        d.x = pd[0]
        d.y = pd[1]
        '''Grid.grid[d.x, d.y] = Grid.grid[d.x, d.y] + .01
        Smell.diffusion(d.x, d.y)'''
    elif d.life > 0:
        pathd = d.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 4 died............. :( \n")

    if pathe and e.life > 0:
        pe = pathe[-1]
        pathe.pop()
        e.x = pe[0]
        e.y = pe[1]
        '''Grid.grid[e.x, e.y] = Grid.grid[e.x, e.y] + .01
        Smell.diffusion(e.x, e.y)'''
    elif e.life > 0:
        pathe = e.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 5 died............. :( \n")

    if pathf and f.life > 0:
        pf = pathf[-1]
        pathf.pop()
        f.x = pf[0]
        f.y = pf[1]

    elif f.life > 0:
        pathf = f.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 6 died............. :( \n")

    if pathg and g.life > 0:
        pg = pathg[-1]
        pathg.pop()
        g.x = pg[0]
        g.y = pg[1]
        '''Grid.grid[g.x, g.y] = Grid.grid[g.x, g.y] + .01
        Smell.diffusion(g.x, g.y)'''
    elif g.life > 0:
        pathg = g.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 7 died............. :( \n")

    if pathh and h.life > 0:
        ph = pathh[-1]
        pathh.pop()
        h.x = ph[0]
        h.y = ph[1]
        '''Grid.grid[h.x, h.y] = Grid.grid[h.x, h.y] + .01
        Smell.diffusion(h.x, h.y)'''
    elif h.life > 0:
        pathh = h.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 8 died............. :( \n")

    if pathk and k.life > 0:
        pk = pathk[-1]
        pathk.pop()
        k.x = pk[0]
        k.y = pk[1]
        '''Grid.grid[k.x, k.y] = Grid.grid[k.x, k.y] + .01
        Smell.diffusion(k.x, k.y)'''
    elif k.life > 0:
        pathk = k.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 9 died............. :( \n")

    if pathj and j.life > 0:
        pj = pathj[-1]
        pathj.pop()
        j.x = pj[0]
        j.y = pj[1]
        '''Grid.grid[j.x, j.y] = Grid.grid[j.x, j.y] + .01
        Smell.diffusion(j.x, j.y)'''
    elif j.life > 0:
        pathj = j.foundfood([sugar1, sugar2, sugar3])

    else:
        print("ant 10 died............. :( \n")

    if a.life > 0:  # this is just to design the ant's in matplotlib, don't worry about it
        ax.scatter(p[0], p[1], 100, 'r', '4')
        ax.scatter(p[0]+.1, p[1]+.1, 10, 'r', 'o')
        ax.scatter(p[0]+.1, p[1]-.1, 10, 'r', 'o')
        ax.scatter(p[0]-.1, p[1]+.1, 10, 'r', 'o')
        ax.scatter(p[0]-.1, p[1]-.1, 10, 'r', 'o')
    if b.life > 0:
        ax.scatter(pb[0], pb[1], 100, 'r', '4')
        ax.scatter(pb[0]+.1, pb[1]+.1, 10, 'r', 'o')
        ax.scatter(pb[0]+.1, pb[1]-.1, 10, 'r', 'o')
        ax.scatter(pb[0]-.1, pb[1]+.1, 10, 'r', 'o')
        ax.scatter(pb[0]-.1, pb[1]-.1, 10, 'r', 'o')
    if c.life > 0:
        ax.scatter(pc[0], pc[1], 100, 'r', '4')
        ax.scatter(pc[0]+.1, pc[1]+.1, 10, 'r', 'o')
        ax.scatter(pc[0]+.1, pc[1]-.1, 10, 'r', 'o')
        ax.scatter(pc[0]-.1, pc[1]+.1, 10, 'r', 'o')
        ax.scatter(pc[0]-.1, pc[1]-.1, 10, 'r', 'o')
    if d.life > 0:
        ax.scatter(pd[0], pd[1], 100, 'r', '4')
        ax.scatter(pd[0]+.1, pd[1]+.1, 10, 'r', 'o')
        ax.scatter(pd[0]+.1, pd[1]-.1, 10, 'r', 'o')
        ax.scatter(pd[0]-.1, pd[1]+.1, 10, 'r', 'o')
        ax.scatter(pd[0]-.1, pd[1]-.1, 10, 'r', 'o')
    if e.life > 0:
        ax.scatter(pe[0], pe[1], 100, 'r', '4')
        ax.scatter(pe[0]+.1, pe[1]+.1, 10, 'r', 'o')
        ax.scatter(pe[0]+.1, pe[1]-.1, 10, 'r', 'o')
        ax.scatter(pe[0]-.1, pe[1]+.1, 10, 'r', 'o')
        ax.scatter(pe[0]-.1, pe[1]-.1, 10, 'r', 'o')
    if f.life > 0:
        ax.scatter(pf[0], pf[1], 100, 'r', '4')
        ax.scatter(pf[0]+.1, pf[1]+.1, 10, 'r', 'o')
        ax.scatter(pf[0]+.1, pf[1]-.1, 10, 'r', 'o')
        ax.scatter(pf[0]-.1, pf[1]+.1, 10, 'r', 'o')
        ax.scatter(pf[0]-.1, pf[1]-.1, 10, 'r', 'o')
    if g.life > 0:
        ax.scatter(pg[0], pg[1], 100, 'r', '4')
        ax.scatter(pg[0]+.1, pg[1]+.1, 10, 'r', 'o')
        ax.scatter(pg[0]+.1, pg[1]-.1, 10, 'r', 'o')
        ax.scatter(pg[0]-.1, pg[1]+.1, 10, 'r', 'o')
        ax.scatter(pg[0]-.1, pg[1]-.1, 10, 'r', 'o')
    if h.life > 0:
        ax.scatter(ph[0], ph[1], 100, 'r', '4')
        ax.scatter(ph[0]+.1, ph[1]+.1, 10, 'r', 'o')
        ax.scatter(ph[0]+.1, ph[1]-.1, 10, 'r', 'o')
        ax.scatter(ph[0]-.1, ph[1]+.1, 10, 'r', 'o')
        ax.scatter(ph[0]-.1, ph[1]-.1, 10, 'r', 'o')
    if k.life > 0:
        ax.scatter(pk[0], pk[1], 100, 'r', '4')
        ax.scatter(pk[0]+.1, pk[1]+.1, 10, 'r', 'o')
        ax.scatter(pk[0]+.1, pk[1]-.1, 10, 'r', 'o')
        ax.scatter(pk[0]-.1, pk[1]+.1, 10, 'r', 'o')
        ax.scatter(pk[0]-.1, pk[1]-.1, 10, 'r', 'o')
    if j.life > 0:
        ax.scatter(pj[0], pj[1], 100, 'r', '4')
        ax.scatter(pj[0]+.1, pj[1]+.1, 10, 'r', 'o')
        ax.scatter(pj[0]+.1, pj[1]-.1, 10, 'r', 'o')
        ax.scatter(pj[0]-.1, pj[1]+.1, 10, 'r', 'o')
        ax.scatter(pj[0]-.1, pj[1]-.1, 10, 'r', 'o')

    sugar1.decomposition(foodlifespan)
    sugar2.decomposition(foodlifespan)
    sugar3.decomposition(foodlifespan)

    foodlifespan += 1

    house.spawn()

    sugar1.spawn()
    sugar2.spawn()
    sugar3.spawn()
    Smell.drawpath()
    Smell.evaporate()
    Smell.spawdiffusion()
    Smell.evaporateall()

    # data writing into a text file which will be used by the dataplot.py script
    ft = open(filepath, "a")
    end = time.time()
    t1 = end - start
    lives = str(a.life) + ", " + str(b.life) + ", " + str(c.life) + ", " + str(d.life) + ", " + str(e.life) + \
        ", " + str(f.life) + ", " + str(g.life) + ", " + \
        str(h.life) + ", " + str(j.life) + ", " + str(k.life) + ", "
    foodearneddata = str(a.foodearned) + ", " + str(b.foodearned) + ", " + str(c.foodearned) + ", " + str(d.foodearned) + ", " + str(e.foodearned) + \
        ", " + str(f.foodearned) + ", " + str(g.foodearned) + ", " + \
        str(h.foodearned) + ", " + str(j.foodearned) + \
        ", " + str(k.foodearned) + ", "
    data = str(sugar1.foodamount) + ", " + str(sugar2.foodamount) + ", " + \
        str(sugar3.foodamount) + ", " + lives + \
        foodearneddata + str(t1) + str("\n")
    # print(data)
    ft.write(data)

    # Grid.display()


# this calls the animate function in each 100 milisecond
ani = animation.FuncAnimation(fig, animate, interval=100)

plt.show()


print("============================\n")
# Grid.display()
print("============================\n")
