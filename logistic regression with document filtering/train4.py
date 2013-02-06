#pass 466192d3
from __future__ import print_function
import copy
import csv
import re
import math
from collections import Counter
import numpy
import scipy.sparse
import scipy.optimize._zeros
from scipy.optimize import fmin_l_bfgs_b

def filtertxt(str):
    str = str.lower()
    str = re.sub('[!?,."):/\-\'(\^123456789]', '',str)
    str = str.replace('  ', ' ')
    return str

def frequency(string):
    """
    return dict of frequncy for each pair of letters
    """
    string=["%s%s%s%s" %(string[i],string[i+1],string[i+2],string[i+3]) for i in range(len(string)-4)]
    return Counter(string)

def absoluteDist(a):
    a= sum(x*x for x in a.values())
    a=math.sqrt(a)
    return a

def dotProduct(x1,x2):
    return sum(v*x2.get(k,0) for k,v in x1.items())

def cosDist(a,b):
    x = absoluteDist(a)*absoluteDist(b)
    if(x>0):
        return 1-dotProduct(a,b)/x
    else:
        return 0

#######################
##LOGISTIC REGRESSION##
#######################

def h(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    vec = 1.0/(1+numpy.exp(x.dot(-theta)))
    return vec
def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    m=y.size
    theta1= copy.deepcopy(theta)
    theta1[0]=0
    J=-sum(y * numpy.log(h(X,theta))+ (1 - y) * numpy.log(1- h(X,theta)))/m  + (1.0*lambda_/(2*m))*sum(theta1**2)
    return J

def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """
    m=y.size
    grad = ((X.T).dot(1.0*y-h(X,theta)).T)/m
    theta1=copy.deepcopy(theta)
    theta1[0]=0
    regularization = (lambda_/m)*theta
    return  -(grad+regularization)

class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x =scipy.sparse.hstack(([1.], x))
        p1 = h(x, self.th) #verjetno razreda 1
        return [ 1-p1, p1 ]

class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = scipy.sparse.hstack((numpy.ones((X.shape[0],1)), X))

        #optimizacija
        theta = fmin_l_bfgs_b(cost,
            x0=numpy.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)

def test_cv(learner, X, y, k=5):
    j=0
    e=0
    vector= []#numpy.zeros(len(y))
    ratio = 1.*len(X)/k

    #k-cross
    for e in range(k):

        newX,testX,newY,testY = [],[],[],[]

        #divide for learning and testing
        for i in range(len(X)):
            if(e*ratio > i or (e+1)*ratio<=i):
                newX.append(X[i])
                newY.append(y[i])
            else:
                testX.append(X[i])
                testY.append(y[i])

        classifier1 = learner(numpy.asarray(newX), numpy.asarray(newY))

        for i,el in enumerate(testX):
            i+=e*ratio
            cli = classifier1(el)
            vector.append(cli)
    return numpy.asarray(vector)

def test_learning(learner, X, y):
    c = learner(X,y)
    results = [ c(x) for x in X ]
    return results

def CA(y, res):
    vec = numpy.zeros(len(y))
    for i,cl in enumerate(res):
        prediction = 0
        if(cl[1]>cl[0]):
            if(y[i]==1):
                vec[i]=cl[1]
            else:
                vec[i]=1-cl[1]
        else:
            if(y[i]==0):
                vec[i]=cl[0]
            else:
                vec[i]=1-cl[0]
    return sum(vec)/len(vec)

def test_learning(learner, X, y):
    c = learner(X,y)
    results = [ c(x) for x in X ]
    return results






#################################################################################
f = open("predictions.tab","w+")
f.close


print("reading file train.tab")
filename    = "train.tab"
inputfile   = csv.reader(open(filename), delimiter='\t')
linelist    = [line for line in inputfile]
linelist    = linelist[1:]
# id   set  score / answer

#filter special chars and optimize

print("filtering file")
for i in linelist:
    i[4]=filtertxt(i[4])


#create dict of id: frequency(windows length=4)
for index in range(10):
    languagedict=[0]*10
    scoredict=[0]*10
    X=[]
    y0=[0]*10
    y1=[0]*10
    y2=[0]*10
    y3=[0]*10
    print("generate X", index)

    languagedict = dict([(i[0], frequency(i[4])) for i in linelist if int(i[1])==index+1])  #id: frequency
    scoredict = dict([(i[0], i[2]) for i in linelist if int(i[1])==index+1])                 #id: score
    #all attruutes
    attribs=[]
    for j,elj in enumerate(languagedict.keys()): # j-index,  elj==id
        for i,eli in enumerate(languagedict[elj]): # i-index, eli==attribute
            attribs.append(eli)
    attribs = set(attribs)
    print(len(attribs),len(languagedict.keys()))
    #create matrix X, score vector y, ids vecotr
    #X= scipy.sparse.csr_matrix((len(attribs),len(languagedict.keys())))
    row=[]
    column=[]
    value=[]

    y0=numpy.zeros(shape=len(languagedict.keys()))
    y1=numpy.zeros(shape=len(languagedict.keys()))
    y2=numpy.zeros(shape=len(languagedict.keys()))
    y3=numpy.zeros(shape=len(languagedict.keys()))

    ids=numpy.zeros(shape=len(languagedict.keys()))
    for j,elj in enumerate(languagedict.keys()):
        ids[j]=elj
        yValue = int(scoredict[elj])

        y0[j]=0
        y1[j]=0
        y2[j]=0
        y3[j]=0
        if yValue==0:
            y0[j]=1
        elif yValue==1:
            y1[j]=1
        elif yValue==2:
            y2[j]=1
        elif yValue==3:
            y3[j]=1
        for i, el in enumerate(attribs):
            if(languagedict[elj][el]>0):
                value.append(languagedict[elj][el])
                row.append(j)
                column.append(i)

    print("dotle")
    X=scipy.sparse.csr_matrix((value, [ row, column]))

    classifier0=[0]*10
    classifier1=[0]*10
    classifier2=[0]*10
    classifier3=[0]*10

    languagedict=0
    scoredict=0
    learner = LogRegLearner(lambda_=7)

    print("create model ",index, "0")
    classifier0 = learner(X,y0) #dobimo model


    print("create model ",index, "1")
    classifier1 = learner(X,y1) #dobimo model

    print("create model ",index, "2")
    classifier2 = learner(X,y2) #dobimo model

    if(index+1 in [1,2,5,6]):
        print("create model ",index, "3")
        classifier3 = learner(X,y3) #dobimo model


    print("read test")
    filename1    = "test.tab"
    inputfile1   = csv.reader(open(filename1), delimiter='\t')
    linelist1    = [line for line in inputfile1]
    linelist1    = linelist1[1:]

    for i in linelist1:
        i[2]=filtertxt(i[2])

    languagedict1=[0]*10
    print("generate X1",index)
    languagedict1 = dict([(i[0], frequency(i[2])) for i in linelist1 if int(i[1])==index+1])  #id: frequency

    #create matrix X, score vector y, ids vecotr
    print("prediction")
    f = open("predictions.tab","a+")

    row=[]
    column=[]
    value=[]
    ids=numpy.zeros(shape=len(languagedict1.keys()))
    for j,elj in enumerate(languagedict1.keys()):
        ids[j]=elj
        X=[0]*len(attribs)
        for i, el in enumerate(attribs):
            if(languagedict1[elj][el]>0):
                X[i]=languagedict1[elj][el]

        if(index+1 in [1,2,5,6]):
            tab = [classifier0(X)[1],classifier1(X)[1],classifier2(X)[1],classifier3(X)[1] ]
        else:
           tab = [classifier0(X)[1],classifier1(X)[1],classifier2(X)[1]]
        print(str(int(ids[j]))+"\t"+str(tab.index(max(tab))),file=f)


    f.close