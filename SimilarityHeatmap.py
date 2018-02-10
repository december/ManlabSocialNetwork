import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm 
from matplotlib import axes
import math
import random
import sys
import os
import seaborn as sns
import pandas as pd

realdic = {} #from id to its cascade dic
relation = {} #from id to follower id
authordic = {} #from tweet id to author id
cnt = 0 #cascade number

simulation = False
filename = 'Real'
if len(sys.argv) > 1:
    filename = sys.argv[1]
    simulation = True
'''
def draw_heatmap(data, xlabels, ylabels, name):
    cmap = cm.Blues    
    figure=plt.figure(facecolor='w')
    ax=figure.add_subplot(2,1,1,position=[0.1,0.15,0.8,0.8])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    vmax=data[0][0]
    vmin=data[0][0]
    for i in data:
        for j in i:
            if j>vmax:
                vmax=j
            if j<vmin:
                vmin=j
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
    plt.savefig(prefix+'similarity/'+filename+'_'+name+'_2D.png')
'''

def calcPJ(x, y):
    p = 0
    j = 0
    xlist = x.keys()
    ylist = y.keys()
    xset = set(xlist)
    yset = set(ylist)
    inter = list(xset.intersection(yset))
    if not len(inter) == 0:
        union = list(xset.union(yset))
        j = len(inter) * 1.0 / len(union)
    #top = 0
    #for key in inter:
    #   top += x[key] * y[key]
    #top *= cnt
    #xsum = sum(x.values())
    #ysum = sum(y.values())
    xlen = len(xlist)
    ylen = len(ylist)
    top = len(inter) * cnt
    top -= xlen * ylen
    bottom = math.sqrt(cnt * xlen - xlen ** 2) * math.sqrt(cnt * ylen - ylen ** 2)
    p = top * 1.0 / bottom
    return p, j

prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
users = 7268
realdata = list()
relation_prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
if not simulation:
    namelist = os.listdir(path)
    for name in namelist:
        fr = open(path+name, 'r')
        realdata.extend(fr.readlines())
        fr.close()
else:
    fr = open(prefix+'result/'+filename+'.detail', 'r')
    realdata = fr.readlines()
    fr.close()

n = len(realdata)
i = 0
while i < n:
    temp = realdata[i].split('\t')
    number = int(temp[1]) + 1
    for j in range(i+1, i+number):
        data = realdata[j].split('\t')
        if not realdic.has_key(data[1]):
            realdic[data[1]] = {}
        realdic[data[1]][cnt] = 1
        #if not realdic[data[1]].has_key(cnt)
        #   realdic[data[1]][cnt] = 1
        #else:
        #   realdic[data[1]][cnt] += 1
    cnt += 1
    i += number
m = len(realdic)
print 'Construct vectors finished.'
print cnt
print m

p_matrix = np.zeros((m, m))
j_matrix = np.zeros((m, m))
x = np.linspace(1, m, m)
pointlist = realdic.keys()

for i in range(m):
    for j in range(i+1, m):
        pij, jij = calcPJ(realdic[pointlist[i]], realdic[pointlist[j]])
        p_matrix[i][j] = pij
        j_matrix[i][j] = jij
    print i
#draw_heatmap(p_matrix, x, x, 'pearson')
#draw_heatmap(j_matrix, x, x, 'jaccard')
df = pd.DataFrame(p_matrix)
sns.heatmap(df, vmin=np.min(p_matrix), vmax=np.max(p_matrix))
plt.savefig(prefix+'similarity/'+filename+'_pearson_2D.png')
plt.cla()
#df = pd.DataFrame(j_matrix)
sns.heatmap(df, vmin=np.min(p_matrix), vmax=np.max(p_matrix))
plt.savefig(prefix+'similarity/'+filename+'_jaccard_2D.png')
plt.cla()
