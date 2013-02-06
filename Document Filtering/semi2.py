import math
from collections import Counter
import os
from PIL import Image,ImageDraw
import operator
from unidecode import unidecode

__author__ = 'Crystal'

def frequency(string):
    """
    return dict of frequncy for each pair of letters
    """
    string=string.lower()
    string=["%s%s " %(string[i],string[i+1]) for i in range(len(string)-1)]
    return Counter(string)

def absoluteDist(a):
    a= sum(x*x for x in a.values())
    a=math.sqrt(a)
    return a

def dotProduct(x1,x2):
    return sum(v*x2.get(k,0) for k,v in x1.items())

def cosDist(a,b):
    return 1-dotProduct(a,b)/(absoluteDist(a)*absoluteDist(b))

class bicluster:
    def __init__(self, left=None, right=None, distance=0.0, id=None, list=None):
        if not list: list = []
        self.left=left
        self.right=right
        self.id=id
        self.distance=distance
        self.list=list

def distancebetweengroups(lista,listb):
    """
    Calculate distance between two groups
    """
    avgdist=0
    for a in lista:
        for b in listb:
            avgdist+=cosDist(languagedict[a],languagedict[b])
    if lista and listb:
        return (avgdist*1.0)/(len(lista)*len(listb))
    return 0

def hcluster(rows):
    """
    Adding countrys to groups until one final "unvisible" group. Return final group. From here we can access left and right node.
    """
    distances={}
    currentclustid=-1
    # Clusters are initially just the rows
    clust=[bicluster(id=i,list=[langs[i]]) for i in range(len(rows))]
    while len(clust)>1:
        lowestpair=(0,1)
        closest=distancebetweengroups(clust[0].list,clust[1].list)
        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
            # distances is the cache of distance calculations
                if (clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)]=distancebetweengroups(clust[i].list,clust[j].list)

                d=distances[(clust[i].id,clust[j].id)]

                if d<closest:
                    closest=d
                    lowestpair=(i,j)

        # create the new cluster
        newcluster=bicluster(
            left=clust[lowestpair[0]],
            right=clust[lowestpair[1]],
            distance=closest*1500,
            id=currentclustid,
            list = clust[lowestpair[0]].list+clust[lowestpair[1]].list
        )
        # cluster ids that weren't in the original set are negative
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    return clust[0]

def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left is None and clust.right is None: return 1
    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left)+getheight(clust.right)

def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left is None and clust.right is None: return 0
    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left),getdepth(clust.right))*100+clust.distance*100

def drawdendrogram(clust,labels,jpeg='clusters.jpg'):
    # height and width
    h=getheight(clust)*20
    w=1100
    depth=getdepth(clust)*100
    # width is fixed, so scale distances accordingly
    scaling=float(w-150)/depth
    # Create a new image with a white background
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)
    #draw.line((0,h/2,10,h/2),fill=(0,0,0))
    # Draw the first node
    drawnode(draw,clust,10,(h/2),scaling,labels)
    img.save(jpeg,'JPEG')

def drawnode(draw,clust,x,y,scaling,labels):
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        # Line length
        ll=clust.distance
        #true distance
        #distance = clust.left.dist #distancebetweengroups(clust.left.list, clust.right.list);
        # Vertical line from this cluster to children
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(0,0,0))
        # Horizontal line to left item
        draw.line((x,top+h1/2,x+ll-clust.left.distance,top+h1/2),fill=(0,0,0))
        # Horizontal line to right item
        draw.line((x,bottom-h2/2,x+ll-clust.right.distance,bottom-h2/2),fill=(0,0,0))

        draw.text((x+1,bottom-h2/2),str(round(clust.distance/1500,3)),(0,0,0))
        # Call the function to draw the left and right nodes
        drawnode(draw,clust.left,x+ll-clust.left.distance,top+h1/2,scaling,labels)
        drawnode(draw,clust.right,x+ll-clust.right.distance,bottom-h2/2,scaling,labels)
    else:
        # If this is an endpoint, draw the item label
        draw.line((x,y,x+clust.distance,y),fill=(0,0,0))
        draw.text((x+clust.distance,y-7),labels[clust.id],(255,0,0))

#get all files in array
path = 'langs/'
langs = os.listdir(path)
languagedict = dict([(i, frequency(unidecode(open(path + i).read().decode("utf-8")))) for i in langs])
#print languagedict

drawdendrogram(hcluster(langs),langs,jpeg='languageGroup-dendrogram.jpg')

####################################################################################
###DETECT LANGUAGE OF FILE DETECT.TXT###############################################
####################################################################################

detectlanguage = frequency(unidecode(open("detect.txt").read().decode("utf-8")))
detectdistances={}
for i in langs:
    detectdistances[i] = cosDist(detectlanguage,languagedict[i])
sortedDistances = sorted(detectdistances.iteritems(), key=operator.itemgetter(1))
sumall = sum([i[1] for i in sortedDistances])

print "brobabal language in detect.txt file is:"
print "Probability for %s is:" % sortedDistances[0][0], (1-sortedDistances[0][1])*100/sumall,"%"
print "Probability for %s is:" % sortedDistances[1][0], (1-sortedDistances[1][1])*100/sumall,"%"
print "Probability for %s is:" % sortedDistances[2][0], (1-sortedDistances[2][1])*100/sumall,"%"
#######################################################################################
#######################################################################################
#######################################################################################

print "\ndistance between 2 groups:"
#fin vs hungary
print "fin vs hungary", cosDist(frequency(unidecode(open(path+"hng.txt").read().decode("utf-8"))),frequency(unidecode(open(path+"fin.txt").read().decode("utf-8"))))
print  "eng vs perz", cosDist(frequency(unidecode(open(path+"eng.txt").read().decode("utf-8"))),frequency(unidecode(open("prsn.txt").read().decode("utf-8"))))
print  "germany vs perz", cosDist(frequency(unidecode(open(path+"ger.txt").read().decode("utf-8"))),frequency(unidecode(open("prsn.txt").read().decode("utf-8"))))