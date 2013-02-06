__author__ = 'Crystal'

import random
from random import choice
import networkx as nx
import matplotlib.pyplot as plt
import operator
from collections import Counter



G = nx.read_gml("fb.gml")
#G = nx.barabasi_albert_graph(160, 1)
num_of_all = len(G.nodes())

degreeList= [G.degree(node) for node in G.nodes_iter()]

map = Counter(degreeList)

plt.hist(nx.degree_histogram(G),bins=num_of_all,log=True)
plt.savefig("histogram.png")


def getMaxLabelOfNeighbours(neighboursList):
    labels = [names[i] for i in neighboursList]
    return sorted(Counter(labels).iteritems(), key=operator.itemgetter(1),reverse=True) #count label occurrence in neighbours and sort

def setLabel(i):
    neighboursList = adjacency_list[i]
    cLabels  = getMaxLabelOfNeighbours(neighboursList)

    if not cLabels:
        return names[i]

    if len(cLabels)>=2 and cLabels[0][1] >= cLabels[1][1]: #most labels are cLabels[0][0]
        return cLabels[0][0]
    else:                                              #else return random value
        return choice(cLabels)[0]

def oneInCorrectGroup(i):
    neighboursList = adjacency_list[i]
    cLabels  = getMaxLabelOfNeighbours(neighboursList)

    if len(cLabels)>=2 and cLabels[0][1] >= cLabels[1][1] and names[i]==cLabels[0][0]:
        return True
    elif len(cLabels)==1 and names[i]==cLabels[0][0]:
        return True
    return False

def allInCorrectGroup():
    for i in range(len(adjacency_list)):
        if not oneInCorrectGroup(i):
            return False
    return True


adjacency_list = G.adjacency_list()     #seznam sosedov
names = {}                              #zacetna imena
for i in range(len(adjacency_list)):
    names[i] = str(i)

setOfVertex = [i for i in range(len(adjacency_list)) ]
random.shuffle(setOfVertex)   #shuffled to work randomly

t=0
while not allInCorrectGroup() and t<50:
    for i in setOfVertex:
        names[i] = setLabel(i)
    t+=1



plt.close()
pos = nx.spring_layout(G)
#nx.draw_networkx(G, labels=names, node_size=4,vmin=0,vmax=10,font_size=3,style="dashdot", width=1)
pos=nx.spring_layout(nx.connected_component_subgraphs(G)[0])
nx.draw_networkx_nodes(nx.connected_component_subgraphs(G)[0],pos,node_size=5,linewidths=0.3,data=True)
nx.draw_networkx_edges(nx.connected_component_subgraphs(G)[0],pos,width = 0.1,alpha=0.3)
nx.draw_networkx_labels(nx.connected_component_subgraphs(G)[0],pos,font_size=0.8,font_color="blue",labels=names)
plt.savefig("G.pdf")

setOfNames = set([ i[1] for i in names.iteritems()])
print "stevilo odkritih skupin: ",len(setOfNames),"(",
for i in setOfNames:
    print i,
print ")"

