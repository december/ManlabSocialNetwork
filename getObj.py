import sys
import scipy as sp
import numpy as np
import scipy.optimize
import numpy.random
import datetime

single = True
filename = int(sys.argv[1])
if filename < 0:
	single = False

users = 7268
allusers = 7268
ts = 0 #start timestamps
te = 864000 #end timestamps
uid = list() #from user index to user id
iddic = {} #from user id to user index
friend = {} #from user id to its followers' user id
rusc = {} #from cascade id to rusc sets and records
nrusc = {} #from cascade id to nrusc sets and records
depth = {} #from tweet id to depth
author = {} #from tweet id to user id
timestamp = {} #from tweet id to timestamp
posts = {} #from user index to post times
q = {} #from cascade id to q function
lc = {} #from cascade id to log-likelihood function value
cdic = {} #from cascade id to cascade index
clist = list() #from cascade index to cascade id
edgemap = {} #from relations to the index of edge
vdic = {} #from user index to the index of point parameter 
edic = {} #from the index of edge to the index of edge parameter
vlist = list() #from the index of point parameter to user index
elist = list() #from the index of edge parameter to the index of edge
vnum = 0
enum = 0
cnum = 0
pos = 0
poslist = list()
total = 0
iters = 1 #iteration times in each M-steps
alpha = 0.00001 #learning rate for optimizer

gamma = -1.0 #log barrier
epsilon = 1.0 #when will EM stop
lbd = np.zeros(users) #parameter lambda which have calculated before


def Select(omega, pi, x, phi1, phi2, phi3, phi4, phi5):
	newomega = list()
	newp = list()
	newx = list()
	philist = list()
	for i in range(5):
		philist.append(list())
	for i in range(vnum):
		newomega.append(omega[vlist[i]])
	for i in range(enum):
		newp.append(pi[elist[i]])
	for i in range(enum):
		newx.append(x[elist[i]])
	for i in range(vnum):
		philist[0].append(phi1[vlist[i]])
	for i in range(vnum):
		philist[1].append(phi2[vlist[i]])	
	for i in range(vnum):
		philist[2].append(phi3[vlist[i]])	
	for i in range(vnum):
		philist[3].append(phi4[vlist[i]])	
	for i in range(vnum):
		philist[4].append(phi5[vlist[i]])		
	return newomega, newp, newx, philist

def Phi(theta1, theta2, theta3, theta4, idx):
	if idx == 0:
		return np.cos(theta1) * np.cos(theta1)
	if idx == 1:
		return np.sin(theta1) * np.sin(theta1) * np.cos(theta2) * np.cos(theta2)
	if idx == 2:
		return np.sin(theta1) * np.sin(theta1) * np.sin(theta2) * np.sin(theta2) * np.cos(theta3) * np.cos(theta3)
	if idx == 3:
		return np.sin(theta1) * np.sin(theta1) * np.sin(theta2) * np.sin(theta2) * np.sin(theta3) * np.sin(theta3) * np.cos(theta4) * np.cos(theta4)
	return np.sin(theta1) * np.sin(theta1) * np.sin(theta2) * np.sin(theta2) * np.sin(theta3) * np.sin(theta3) * np.sin(theta4) * np.sin(theta4)

def LnLc(omega, pi, x, philist, c, tau): #ln fromulation of one cascades's likelihood on tau(do not include part of Q)
	uc = vdic[iddic[author[c]]]
	s = np.log(lbd[vlist[uc]]) + np.log(philist[tau][uc])
	for item in rusc[c]:
		edge = item[0]
		u = item[3]
		s += np.log(omega[u]) - omega[u] * item[1] + tf.log(pi[edge]) - item[2] * np.log(x[edge]) + np.log(philist[tau][u])
	for item in nrusc[c]:
		edge = item[0]
		u = item[3]
		exponent = max(-1 * omega[u] * item[1], -100)
		estimate = np.exp(exponent) - 1
		#print edgemap[uc][u]
		result = 1 + pi[edge] * x[edge] ** (-1 * item[2]) * philist[tau][u] * estimate
		s += np.log(result)
	return s

def QF(omega, pi, x, philist, c): #calculate q funciton with tricks
	for i in range(5):
		lc[c][i] = LnLc(omega, pi, x, philist, c, i)
	for i in range(5):
		s = 0
		for j in range(5):
			s += tf.exp(lc[c][j] - lc[c][i])
		q[c][i] = 1 / s

def ObjF(omega, pi, x, philist): #formulation of objective function (include barrier) (the smaller the better)
	#global total
	#total += 1
	noreply = 0
	'''
	print 'Begin'
	print omega
	print x
	print pi
	'''
	#obj = (np.log(omega+10**-5).sum() + np.log(x+10**-5).sum() + np.log(1-pi+10**-5).sum() + np.log(pi+10**-5).sum()) * gamma #need to be fixxed
	obj = 0
	for c in q:
		if len(rusc[c]) == 0:
			if noreply == 0:
				for i in range(5):
					noreply -= q[c][i] * LnLc(omega, pi, x, philist, c, i)
					tmp = q[c][i] * tf.log(q[c][i])
					noreply += tf.cast(tmp, dtype=tf.float64)
			obj += noreply
			continue
		for i in range(5):
			obj -= q[c][i] * LnLc(omega, pi, x, philist, c, i)
			tmp = q[c][i] * tf.log(q[c][i])
			obj = obj + tf.cast(tmp, dtype=tf.float64)
	#if total % 10000 == 0:
	#	print 'No.' + str(total) + ' times: ' + str(obj)
	return obj

def EStep(omega, pi, x, philist): #renew q and lc
	#print [len(omega), len(pi), len(x)]
	#print [len(oc), len(pc), len(xc)]
	#count = 0
	for c in q:
		QF(oc, pc, xc, philist, c)
		#count += 1
		#print count

def SingleObj(data, u):
	global vnum, enum, cnum
	n = len(data)
	#last = int(data[1].split('\t')[2])
	i = 0
	while i < n:
		temp = data[i].split('\t')
		number = int(temp[1]) + 1
		rusc[temp[0]] = list()
		nrusc[temp[0]] = list()
		clist.append(temp[0])
		cdic[temp[0]] = cnum
		cnum += 1
		q[temp[0]] = list()
		lc[temp[0]] = list()
		for j in range(5):
			q[temp[0]].append(0.2)
			lc[temp[0]].append(0.0)
		casdic = {} #from tweet id to user id who replied it with which tweet id
		for j in range(i+1, i+number):
			tweet = data[j].split('\t')
			#print tweet
			author[tweet[0]] = tweet[1]
			timestamp[tweet[0]] = int(float(tweet[2]))
			if not vdic.has_key(iddic[tweet[1]]):
				vdic[iddic[tweet[1]]] = vnum
				vnum += 1
				vlist.append(iddic[tweet[1]])
			if not casdic.has_key(tweet[0]):
				casdic[tweet[0]] = {}
			if tweet[3] == '-1':
				depth[tweet[0]] = 0
			else:
				depth[tweet[0]] = depth[tweet[3]] + 1
				casdic[tweet[3]][tweet[1]] = tweet[0]
		for item in casdic:
			#print item
			#print author[item]
			#print friend[author[item]]
			if not friend.has_key(author[item]):
				continue
			for f in friend[author[item]]:
				if not edic.has_key(edgemap[iddic[author[item]]][iddic[f]]):
					edic[edgemap[iddic[author[item]]][iddic[f]]] = enum
					enum += 1
					elist.append(edgemap[iddic[author[item]]][iddic[f]])
				if not vdic.has_key(iddic[f]):
					vdic[iddic[f]] = vnum
					vnum += 1
					vlist.append(iddic[f])
				info = list()
				if f in casdic[item]: #this person retweeted it
					info.append(edic[edgemap[iddic[author[item]]][iddic[f]]])
					info.append(timestamp[casdic[item][f]] - timestamp[item])
					info.append(depth[item])
					info.append(vdic[iddic[f]])
					rusc[temp[0]].append(info)
				else: #this person did not retweet it
					info.append(edic[edgemap[iddic[author[item]]][iddic[f]]])
					info.append(te - timestamp[item])
					info.append(depth[item])
					info.append(vdic[iddic[f]])
					nrusc[temp[0]].append(info)
		i += number		


#Get lambda value
print 'Preparatory work begins...'
prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'
fr = open(prefix+'lambda_Poisson'+suffix, 'r')
lbdlist = fr.readlines()
for i in range(users):
	temp = lbdlist[i].split('\t')
	uid.append(temp[0])
	iddic[temp[0]] = i
	lbd[i] = float(temp[1])
fr.close()

#Get post times
fr = open(prefix+'posttimes'+suffix, 'r')
post = fr.readlines()
for i in range(users):
	temp = post[i].split('\t')
	posts[iddic[temp[0]]] = int(temp[1])
fr.close()

#Give initial value and construct relation
print 'Construct relation network and give initial value...'

pi = list() #parameter pi (based on edges), row is sender while col is receiver
x = list() #parameter x (based on edges), row is sender while col is receiver
fr = open(prefix+'relations'+suffix, 'r')
relation = fr.readlines()
n = len(relation)
i = 0
while i < n:
	temp = relation[i].split('\t')
	number = int(temp[1]) + 1
	friend[temp[0]] = list()
	if not iddic.has_key(temp[0]):
			iddic[temp[0]] = allusers
			uid.append(temp[0])
			allusers += 1
	for j in range(i+1, i+number):
		fd = relation[j].split('\t')
		if not iddic.has_key(fd[1]):
			iddic[fd[1]] = allusers
			uid.append(fd[1])
			allusers += 1
		if not edgemap.has_key(iddic[temp[0]]):
			edgemap[iddic[temp[0]]] = {}
		edgemap[iddic[temp[0]]][iddic[fd[1]]] = pos
		pos += 1
		if iddic[temp[0]] >= users or int(fd[2]) == 0:
			pi.append(10 ** -5)
		else:
			pi.append(min(1-10**-5, int(fd[2]) * 1.0 / posts[iddic[temp[0]]]))
		x.append(1.0)
		friend[temp[0]].append(fd[1])
	i += number
fr.close()
pi = np.array(pi)
pi = np.arccos(np.sqrt(pi))
x = np.array(x)

omega = np.zeros(allusers) #parameter omega
phi1 = np.zeros(allusers) #one of spherical coordinates of phi distribution
phi2 = np.zeros(allusers) #one of spherical coordinates of phi distribution
phi3 = np.zeros(allusers) #one of spherical coordinates of phi distribution
phi4 = np.zeros(allusers) #one of spherical coordinates of phi distribution
phi5 = np.zeros(allusers) #one of spherical coordinates of phi distribution

'''
tr = list()
for i in range(4):
	tr.append(np.random.rand())
print tr
theta1 += np.arccos(np.sqrt(tr[0]))
theta2 += np.arccos(np.sqrt(tr[1]))
theta3 += np.arccos(np.sqrt(tr[2]))
theta4 += np.arccos(np.sqrt(tr[3]))
'''
#Read personal cascade file
print 'Read behavior log...'
for i in range(users):
	if single and i != filename:
		continue
	fr = open(prefix+'single_user_post/'+str(i)+'_'+uid[i]+'_syn'+suffix, 'r')
	single = fr.readlines()
	SingleObj(single, i)
	fr.close()

if single:
	prefix = prefix + 'single_user_parameter/'
	suffix = '_' + str(filename) + suffix

fr = open(prefix+'omega_Poisson'+suffix, 'r')
omglist = fr.readlines()
vpnum = len(omglist)

for i in range(vpnum):
	temp = omglist[i].split('\t')
	omega[iddic[temp[0]]] = float(temp[1])
fr.close()
#print iddic

for i in range(5):
	fr = open(prefix+'phi'+str(i)+'_Poisson'+suffix, 'r')
	philist = fr.readlines()
	for j in range(vpnum):
		temp = philist[j].split('\t')
		if i == 0:
			phi1[iddic[temp[0]]] = float(temp[1])
		if i == 1:
			phi2[iddic[temp[0]]] = float(temp[1])
		if i == 2:
			phi3[iddic[temp[0]]] = float(temp[1])
		if i == 3:
			phi4[iddic[temp[0]]] = float(temp[1])
		if i == 4:
			phi5[iddic[temp[0]]] = float(temp[1])
	fr.close()

fr = open(prefix+'pi_Poisson'+suffix, 'r')
pilist = fr.readlines()
epnum = len(pilist)

for i in range(enum):
	temp = pilist[i].split('\t')
	row = iddic[int(temp[0])]
	col = iddic[int(temp[1])]
	egidx = edgemap[row][col]
	if not edic.has_key(egidx):
		continue
	pi[edic[egidx]] = float(temp[2])
fr.close()

fr = open(prefix+'x_Poisson'+suffix, 'r')
xlist = fr.readlines()
for i in range(enum):
	temp = xlist[i].split('\t')
	row = iddic[int(temp[0])]
	col = iddic[int(temp[1])]
	egidx = edgemap[row][col]
	if not edic.has_key(egidx):
		continue
	x[edic[egidx]] = float(temp[2])
fr.close()

omega, pi, x, philist = Select(omega, pi, x, phi1, phi2, phi3, phi4, phi5)
print 'There are ' + str(vnum * 5) + ' point parameters and ' + str(enum * 2) + ' edge parameters to be learned...'

EStep(omega, pi, x, philist)
obj = ObjF(omega, pi, x, philist)
print obj
