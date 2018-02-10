import sys
import scipy as sp
import numpy as np
import numpy.random

single = True
filename = int(sys.argv[1])
ts = 0
te = int(float(sys.argv[2])) * 86400
sims = int(sys.argv[3])
if filename < 0:
	single = False

users = 7268
vnum = 7268
enum = 7268
edgemap = {}
uid = list() #from user index in this ego network to user id
iddic = {} #from user id to user index in this ego network
tweetdic = {} #from tweet id to the user index of its author
number = 0 #total number of tweeters
k = 1.2

def GetIET(l):
	p = numpy.random.rand()
	t = -1 * np.log(1-p) / l
	return round(t)

def GetPhi(p1, p2, p3, p4, p5, tau, v):
	if tau == 0:
		return p1[v]
	if tau == 1:
		return p2[v]
	if tau == 2:
		return p3[v]
	if tau == 3:
		return p4[v]
	return p5[v]

def GetTau(p1, p2, p3, p4, p5, v):
	p = numpy.random.rand()
	if p <= p1[v]:
		return 0
	p -= p1[v]
	if p <= p2[v]:
		return 1
	p -= p2[v]
	if p <= p3[v]:
		return 2
	p -= p3[v]
	if p <= p4[v]:
		return 3
	return 4	

def GetLog(r, p, u, t, c, d): #root_tweet, parent_tweet, parent_user, parent_time, tau, cascade log, depth
	global number
	if not edgemap.has_key(u):
		return c
	for f in edgemap[u]:
		see = t + GetIET(omega[f])
		if see > te:
			continue
		#thres = d ** -x[edgemap[u][f]] * pi[edgemap[u][f]]
		realpi = pi[edgemap[u][f]]
		if d > 1:
			realpi = x[edgemap[u][f]] * k ** -(d - 1)
		if np.random.rand() <= realpi * 0.1:
			current = number
			tweetdic[current] = f
			number += 1
			temp = list()
			temp.append(current)
			temp.append(uid[f])
			temp.append(see)
			temp.append(p)
			temp.append(uid[u])
			c.append(temp)
			c = GetLog(r, current, f, see, c, d+1)
	return c

prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'

lbddic = {}
fr = open(prefix+'lambda_Poisson'+suffix, 'r')
lbdlist = fr.readlines()
postlist = list()
for i in range(users):
	temp = lbdlist[i].split('\t')
	lbddic[int(temp[0])] = float(temp[1])
	postlist.append(int(temp[0]))	
fr.close()

if single:
	prefix = prefix + 'single_user_parameter_notopic/'
	suffix = '_' + str(filename) + suffix
else:
	suffix = '_notopic' + suffix
if len(sys.argv) > 4:
	prefix += sys.argv[4] + '/'

fr = open(prefix+'omega_Poisson'+suffix, 'r')
omglist = fr.readlines()
vnum = len(omglist)

lbd = np.zeros(vnum) #parameter lambda which have calculated before
omega = np.zeros(vnum) #parameter omega
#phi1 = np.zeros(vnum) + 0.2 #one of topic distribution
#phi2 = np.zeros(vnum) + 0.2 #one of topic distribution
#phi3 = np.zeros(vnum) + 0.2 #one of topic distribution
#phi4 = np.zeros(vnum) + 0.2 #one of topic distribution
#phi5 = np.zeros(vnum) + 0.2 #one of topic distribution

for i in range(vnum):
	temp = omglist[i].split('\t')
	uid.append(temp[0])
	iddic[int(temp[0])] = i
	omega[i] = float(temp[1])
fr.close()
#print iddic

#for key in lbddic:
#	lbd[iddic[key]] = lbddic[key]

fr = open(prefix+'pi_Poisson'+suffix, 'r')
pilist = fr.readlines()
enum = len(pilist)

pi = np.zeros(enum) #parameter pi (based on edges), row is sender while col is receiver
x = np.zeros(enum) #parameter x (based on edges), row is sender while col is receiver

for i in range(enum):
	temp = pilist[i].split('\t')
	row = iddic[int(temp[0])]
	col = iddic[int(temp[1])]
	if not edgemap.has_key(row):
		edgemap[row] = {}
	edgemap[row][col] = i
	pi[i] = float(temp[2])
fr.close()

fr = open(prefix+'x_Poisson'+suffix, 'r')
xlist = fr.readlines()
for i in range(enum):
	temp = xlist[i].split('\t')
	x[i] = float(temp[2])
fr.close()

print 'Finished reading..'
prefix = '../../cascading_generation_model/simulation/'
suffix = '.detail'
if single:
	prefix += str(filename) + '/'

for j in range(sims):
	number = 0
	behavior = list()
	print 'Generation ' + str(j+1) + ' begins...'
	casnum = 0
	totalnum = 0
	for i in range(users):
		if single and i != 0:
			continue
		l = lbddic[postlist[i]]
		ts = GetIET(l)
		newi = iddic[postlist[i]]
		print i
		while ts < te:
			casnum += 1
			tweetdic[number] = newi
			root = number
			number += 1
			cascade = list()
			temp = list()
			temp.append(root)
			temp.append(uid[newi])
			temp.append(ts)
			temp.append(-1)
			temp.append(-1)
			cascade.append(temp)
			#tau = GetTau(phi1, phi2, phi3, phi4, phi5, i)
			cascade = GetLog(root, root, newi, ts, cascade, 1)
			cascade = sorted(cascade, key=lambda c:c[2])
			size = len(cascade)
			temp = list()
			temp.append(root)
			temp.append(size)
			behavior.append(temp)
			behavior.extend(cascade)
			totalnum += size
			iet = GetIET(l)
			ts += iet
	print casnum
	print totalnum
	fw = open(prefix+str(j)+suffix, 'w')
	for item in behavior:
		fw.write(str(item[0]))
		n = len(item)
		for k in range(1, n):
			fw.write('\t')
			fw.write(str(item[k]))
		fw.write('\n')
	fw.close()

print 'Finished generation task...'
