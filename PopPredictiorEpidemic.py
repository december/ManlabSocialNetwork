import sys     
import scipy as sp
import scipy.stats
import numpy as np
import numpy.random

def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()
import signal
signal.signal(signal.SIGINT, debug_signal_handler)


ts = 0
te = 5 * 86400

users = 7268
vnum = 7268
enum = 7268
edgemap = {}
uid = list() #from user index in this ego network to user id
iddic = {} #from user id to user index in this ego network
idlist = list()
tweetdic = {} #from tweet id to the user index of its author
number = 0 #total number of tweeters
k = 1.05
#normal_rand = scipy.stats.truncnorm.rvs(0, 1, loc=0, scale=1, size=100000000)
#nrpos = 0

def GetIET(l):
	#global nrpos
	p = numpy.random.rand()
	#p = normal_rand[nrpos]
	#nrpos += 1
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

def GetExpect(u, d, rp, s): #root_tweet, parent_tweet, parent_user, parent_time, tau, cascade log, depth
	if d >= 20 or rp <= 5e-2:
		return s
	if not edgemap.has_key(u):
		return s
	for f in edgemap[u]:
		#psaw = 1 - np.exp(-omega[f]*te)
		#psaw = 1
		realpi = beta0[edgemap[u][f]]
		if d > 1:
			realpi = beta1[edgemap[u][f]]		
		p = realpi * (1 - gamma[u])
		s += rp * p
		#s = GetExpect(f, d+1, rp * p, s)
	return s

def Select(prusc, pop, selection, depdic):
	while pop > len(selection) and len(prusc) > 0:
		maximum = max(prusc.values())
		user = max(prusc.items(), key=lambda x: x[1])[0]
		selection.add(user)
		prusc.pop(user)
		if not edgemap.has_key(user):
			continue
		for f in edgemap[user]:
			if not f in selection:
				prusc[f] = (1 - gamma[user]) * beta0[edgemap[user][f]]
				depdic[f] = depdic[user] + 1
	return selection

prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'

fr = open(prefix+'Popularity'+suffix, 'r')
questions = fr.readlines()
pop = list()
for line in questions:
	pop.append(line[:-1].split('\t')[1:])
fr.close()

fr = open(prefix+'Popularity_answer'+suffix, 'r')
questions = fr.readlines()
pop_answer = list()
for line in questions:
	pop_answer.append(line[:-1].split('\t')[:-1])
fr.close()
'''
lbddic = {}
fr = open(prefix+'lambda_Poisson'+suffix, 'r')
lbdlist = fr.readlines()
postlist = list()
for i in range(users):
	temp = lbdlist[i].split('\t')
	lbddic[int(temp[0])] = float(temp[1])
	postlist.append(int(temp[0]))
fr.close()
'''
#if int(sys.argv[4]) == 0:
#	suffix = '_notopic.detail'
prefix += sys.argv[1] + '/'

fr = open(prefix+'gamma'+suffix, 'r')
omglist = fr.readlines()
vnum = len(omglist)

lbd = np.zeros(vnum) #parameter lambda which have calculated before
gamma = np.zeros(vnum) #parameter omega

for i in range(vnum):
	temp = omglist[i].split('\t')
	uid.append(temp[0])
	iddic[int(temp[0])] = i
	idlist.append(temp[0])
	gamma[i] = float(temp[1])
fr.close()
#print iddic

#for key in lbddic:
#	lbd[iddic[key]] = lbddic[key]

fr = open(prefix+'beta0'+suffix, 'r')
pilist = fr.readlines()
enum = len(pilist)

beta0 = np.zeros(enum) #parameter pi (based on edges), row is sender while col is receiver
beta1 = np.zeros(enum) #parameter pi (based on edges), row is sender while col is receiver

for i in range(enum):
	temp = pilist[i].split('\t')
	row = iddic[int(temp[0])]
	col = iddic[int(temp[1])]
	if not edgemap.has_key(row):
		edgemap[row] = {}
	edgemap[row][col] = i
	beta0[i] = float(temp[2])
fr.close()

fr = open(prefix+'beta1'+suffix, 'r')
xlist = fr.readlines()
for i in range(enum):
	temp = xlist[i].split('\t')
	beta1[i] = float(temp[2])
fr.close()
#x += 1

print 'Finished reading..'
#prefix = '../../cascading_generation_model/simulation/'
#suffix = '.detail'
accuracy = list()
mae = list()
expect_pop = {}
n = len(pop)
right = 0
for i in range(n):
	line = pop[i]
	flag = False
	poineer = list()
	for j in line:
		if j == 1:
			flag = True
			break
		poineer.append(iddic[int(j)])
	if flag:
		continue
	s = GetExpect(poineer[0], 1, 1, 0)
	'''
	for tau in range(5):
		d = tau + 1
		if tau > 0:
			d = 1			
		s += GetExpect(poineer[tau], d, 1, 0) + 1
	s = s / 5
	
	mape = abs(pop_answer[i] - s) * 1.0 / pop_answer[i]
	accuracy.append(mape)
	mae.append(abs(pop_answer[i] - s))	
	'''
	panumer = int(pop_answer[i][0])
	print str(s) + '\t' + str(pop_answer[i][0]) + '\t' + idlist[poineer[0]]
	mae.append(abs(panumer - s))
	total += panumer
	#print i
	#print i

#print accuracy
#print len(accuracy)
#print sum(accuracy) / len(accuracy)
print sum(mae) / len(mae)
print len(mae)
#print right