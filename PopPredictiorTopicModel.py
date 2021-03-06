import sys     
import scipy as sp
import scipy.stats
import numpy as np
import numpy.random
from random import choice

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

def GetExpect(u, tau, d, rp, s): #root_tweet, parent_tweet, parent_user, parent_time, tau, cascade log, depth
	if d >= 20 or rp <= 1e-2:
		return s
	if not edgemap.has_key(u):
		return s
	for f in edgemap[u]:
		psaw = 1 - np.exp(-omega[f]*te)
		#psaw = 1
		#realpi = pi[edgemap[u][f]]
		#if d > 1:
		realpi = x[edgemap[u][f]] * k ** -(d - 1)		
		p = psaw * realpi * GetPhi(phi1, phi2, phi3, phi4, phi5, tau, f)
		s += rp * p
		#s = GetExpect(f, tau, d+1, rp * p, s)
	return s

def GetRanking(u, taudb, cset, uid):
	result = {}
	#print uid
	for f in edgemap[u]:
		if not f in cset:
			continue
		p = 0
		iterset = [1,3,4]
		if uid == '1':
			iterset = [1,2,4]
		for tau in iterset:
			#p += pi[edgemap[u][f]] * GetPhi(phi1, phi2, phi3, phi4, phi5, tau, f) * taudb[tau]
			#p = np.log(GetPhi(phi1, phi2, phi3, phi4, phi5, tau, f)) + np.log(taudb[tau]) + np.log(x[edgemap[u][f]])
			#p = np.log(taudb[tau]) + np.log(x[edgemap[u][f]])
			p = np.log(x[edgemap[u][f]]) + np.log(taudb[tau]) * 200000
			if uid == '1':
				p = np.log(x[edgemap[u][f]]) + np.log(taudb[tau]) * 200000
		result[f] = p
	return sorted(result.iteritems(), key=lambda d:d[1], reverse=True)

def Select(prusc, pop, selection, depdic, infer):
	while pop > len(selection) and len(prusc) > 0:
		maximum = max(prusc.values())
		user = max(prusc.items(), key=lambda x: x[1])[0]
		selection.add(user)
		prusc.pop(user)
		if not edgemap.has_key(user):
			continue
		for f in edgemap[user]:
			if not f in selection:
				prusc[f] = maximum * (1 - np.exp(-omega[f]*te)) * x[edgemap[user][f]] * GetPhi(phi1, phi2, phi3, phi4, phi5, infer, f) * k ** -depdic[user]
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

fr = open(prefix+'Participation_answer'+suffix, 'r')
questions = fr.readlines()
par_answer = list()
for line in questions:
	par_answer.append(line[:-1].split('\t')[0][:-1])
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

fr = open(prefix+'omega_Poisson'+suffix, 'r')
omglist = fr.readlines()
vnum = len(omglist)

#lbd = np.zeros(users) #parameter lambda which have calculated before
omega = np.zeros(vnum) #parameter omega
phi1 = np.zeros(vnum) #one of topic distribution
phi2 = np.zeros(vnum) #one of topic distribution
phi3 = np.zeros(vnum) #one of topic distribution
phi4 = np.zeros(vnum) #one of topic distribution
phi5 = np.zeros(vnum) #one of topic distribution
phi1_post = np.zeros(vnum) #one of topic distribution
phi2_post = np.zeros(vnum) #one of topic distribution
phi3_post = np.zeros(vnum) #one of topic distribution
phi4_post = np.zeros(vnum) #one of topic distribution
phi5_post = np.zeros(vnum) #one of topic distribution

for i in range(vnum):
	temp = omglist[i].split('\t')
	uid.append(temp[0])
	iddic[int(temp[0])] = i
	idlist.append(temp[0])
	omega[i] = float(temp[1])
fr.close()
#print iddic

#for key in lbddic:
#	lbd[iddic[key]] = lbddic[key]

for i in range(5):
	fr = open(prefix+'phi'+str(i)+'_Poisson_post'+suffix, 'r')
	philist = fr.readlines()
	for j in range(vnum):
		temp = philist[j].split('\t')
		if i == 0:
			phi1_post[j] = float(temp[1])
		if i == 1:
			phi2_post[j] = float(temp[1])
		if i == 2:
			phi3_post[j] = float(temp[1])
		if i == 3:
			phi4_post[j] = float(temp[1])
		if i == 4:
			phi5_post[j] = float(temp[1])
	fr.close()

for i in range(5):
	fr = open(prefix+'phi'+str(i)+'_Poisson'+suffix, 'r')
	philist = fr.readlines()
	for j in range(vnum):
		temp = philist[j].split('\t')
		if i == 0:
			phi1[j] = float(temp[1])
		if i == 1:
			phi2[j] = float(temp[1])
		if i == 2:
			phi3[j] = float(temp[1])
		if i == 3:
			phi4[j] = float(temp[1])
		if i == 4:
			phi5[j] = float(temp[1])
	fr.close()

fr = open(prefix+'pi_Poisson'+suffix, 'r')
pilist = fr.readlines()
enum = len(pilist)

pi = np.zeros(enum) #parameter pi (based on edges), row is sender while col is receiver
x = np.zeros(enum) #parameter x (based on edges), row is sender while col is receiver

edgelist = list()
for i in range(enum):
	temp = pilist[i].split('\t')
	edgelist.append(temp[0] + '\t' + temp[1] + '\t')
	row = iddic[int(temp[0])]
	col = iddic[int(temp[1])]
	if not edgemap.has_key(row):
		edgemap[row] = {}
	edgemap[row][col] = i
	#phisum = phi1[row] * phi1[col] + phi2[row] * phi2[col] + phi3[row] * phi3[col] + phi4[row] * phi4[col] + phi5[row] * phi5[col]
	#pi[i] = min(float(temp[2]) / phisum ** 3, 1)
	pi[i] = float(temp[2])
fr.close()

fr = open(prefix+'x_Poisson'+suffix, 'r')
xlist = fr.readlines()
for i in range(enum):
	temp = xlist[i].split('\t')
	x[i] = float(temp[2])
fr.close()
#x += 1

print 'Finished reading..'
#prefix = '../../cascading_generation_model/simulation/'
#suffix = '.detail'
accuracy = list()
mae = list()
expect_pop = {}
answer = list()
right = 0
wrong = 0
threshold = 20
bigger = 0
smaller = 0
n = len(pop)
total = 0
random = list()
list1 = list()
list2 = list()
for i in range(n):
	line = pop[i]
	ans = par_answer[i].split(',')
	if ans[0] == '':
		continue
	flag = False
	poineer = list()
	for j in line:
		#if j == '1':
		#	flag = True
		#	break
		poineer.append(iddic[int(j)])
	if flag:
		continue
	expect_pop[poineer[0]] = [1]
	for ui in range(1, 5):
		expect_pop[poineer[0]].append(GetExpect(poineer[0], ui, 1, 1, 0))
	#print expect_pop[poineer[0]]
	delta = 0
	infer = list()
	p = expect_pop[poineer[0]]
	psum = expect_pop[poineer[0]][1] + expect_pop[poineer[0]][3] + expect_pop[poineer[0]][4] + expect_pop[poineer[0]][0] + expect_pop[poineer[0]][2]
	for j in range(1, 5):
		p[j] = expect_pop[poineer[0]][j] / psum
	for j in range(5):
		#mul = np.log(expect_pop[poineer[0]][j]) * 5
		mul = 0
		for tau in range(1, 5):
			mul += np.log(GetPhi(phi1, phi2, phi3, phi4, phi5, j, poineer[tau]))
		infer.append(mul)
	norm = infer[3] + infer[4] + infer[1] + infer[0] + infer[2]
	for j in range(5):
		infer[j] = infer[j] / norm
	#p = [0, 3,2,4,1]
	
	norm = infer[3] * p[3] + infer[4] * p[4] + infer[1] * p[1] + infer[2] * p[2] + infer[0] * p[0]
	for j in range(5):
		infer[j] = infer[j] * p[j] / norm
	realset = set(ans)
	choiceset = set()
	for item in realset:
		choiceset.add(iddic[int(item)])
	while len(choiceset) < 2 * len(realset):
		choiceset.add(choice(edgemap[poineer[0]].keys()))
	ranking = GetRanking(poineer[0], infer, choiceset, idlist[poineer[0]])
	
	topnum = len(ans)
	simset = set()
	
	for j in range(topnum):
		simset.add(idlist[ranking[j][0]])
	inter = realset & simset
	acu = len(inter) * 1.0 / len(realset)
	random.append(len(realset) * 1.0 / len(ranking))
	print str(len(inter)) + '\t' + str(len(realset)) + '\t' + str(len(ranking)) + '\t' + str(acu) + '\t' + idlist[poineer[0]]
	if idlist[poineer[0]] == '1':
		list1.append(acu)
	else:
		list2.append(acu)
	'''

	s = 0
	for ui in range(1, 5):
		s += expect_pop[poineer[0]][ui] * infer[ui]
	
	s = 0
	temps = 0
	num = 0
	for tau in range(5):
		d = tau + 1
		if tau > 0:
			d = 1
		for ui in range(0, 5):
			if ui == 0:
				continue
			s += GetExpect(poineer[tau], ui, d, 1, 0) / 4
	'''
		#s += GetExpect(poineer[tau], 4, d, 1, 0)
	#s += 10
	#s = s / 5
	#print i
	'''
	panumer = int(pop_answer[i][0])
	
	
	if pop_answer[i] > threshold:
		bigger += 1
	else:
		smaller += 1

	if (s > threshold) == (pop_answer[i] > threshold):
		right += 1
	else:
		wrong += 1 
	
	answer.append(infer)
	'''
	#mape = abs(panumer - s) * 1.0 / (panumer + 5)
	accuracy.append(acu)
	#accuracy.append(mape)
	
	#mae.append(abs(panumer - s))
	#total += panumer
	
	#print i

#print accuracy
#print expect_pop
#print answer
#print len(mae)
print len(accuracy)
print sum(accuracy) / len(accuracy)
print sum(list1) / len(list1)
print sum(list2) / len(list2)
print sum(random) / len(random)
#print sum(mae) / len(mae)
#print total
#print str(right) + '\t' + str(wrong)
#print str(bigger) + '\t' + str(smaller)
#print right
