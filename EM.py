import scipy as sp
import numpy as np

users = 7268
ts = 1321286400 #start timestamps
te = 1322150400 #end timestamps
uid = list()
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
edgemap = {} #from relations to the index of parameter pi and x
pos = 0

gamma = 1.0 #log barrier

lbd = np.zeros(users) #parameter lambda which have calculated before
omega = np.zeros(users) #parameter omega
theta1 = np.zeros(users) #one of spherical coordinates of phi distribution
theta2 = np.zeros(users) #one of spherical coordinates of phi distribution
theta3 = np.zeros(users) #one of spherical coordinates of phi distribution
theta4 = np.zeros(users) #one of spherical coordinates of phi distribution
pi = list() #parameter pi (based on edges), row is sender while col is receiver
x = list() #parameter x (based on edges), row is sender while col is receiver

def Resolver(param):


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

def LnLc(omega, pi, x, theta1, theta2, theta3, theta4, c, tau): #ln fromulation of one cascades's likelihood on tau(do not include part of Q)
	uc = iddic[author[c]]
	s = np.log(lbd[uc]) + np.log(Phi(theta1, theta2, theta3, theta4, tau)[uc])
	for item in rusc[c]:
		u = iddic[author[item[0]]]
		s += np.log(omega[u]) - omega[u] * item[1] + np.log(pi[edgemap[uc][u]]) - item[2] * np.log(x[edgemap[uc][u]]) + np.log(Phi(tau)[u])
	for item in nrusc[c]:
		u = iddic[author[item[0]]]
		exponent = -1 * omega[u] * item[1]
		estimate = -1
		if exponent >= -100:
			estimate = np.exp(exponent) - 1
		result = 1 + pi[edgemap[uc][u]] * x[edgemap[uc][u]] ** (-1 * item[2]) * Phi(theta1, theta2, theta3, theta4, tau)[u] * estimate
		s += np.log(result)
	return s

def QF(omega, pi, x, theta1, theta2, theta3, theta4, c): #calculate q funciton with tricks
	for i in range(5):
		lc[c][i] = LnLc(omega, pi, x, theta1, theta2, theta3, theta4, c, i)
	for i in range(5):
		s = 0
		for j in range(5):
			s += np.exp(lc[c][j] - lc[c][i])
		q[c][i] = 1 / s

def ObjF(omega, pi, x, theta1, theta2, theta3, theta4): #formulation of objective function (include barrier) (the smaller the better)
	obj = (np.log(omega).sum() + np.log(x).sum() + np.log(1-pi).sum() + np.log(pi).sum()) * gamma #need to be fixxed
	for c in q:
		for i in range(5):
			obj -= q[c][i] * LnLc(omega, pi, x, theta1, theta2, theta3, theta4, c, i)
			obj += q[c][i] * np.log(q[c][i])
	return obj

def EStep(): #renew q and lc
	for c in q:
		QF(omega, pi, x, theta1, theta2, theta3, theta4, c)

def MStep(): #optimize parameters to achieve smaller obj
	

def SingleObj(data, u):
	n = len(data)
	last = int(data[1].split('\t')[2])
	i = 0
	while i < n:
		temp = data[i].split('\t')
		tm = int(data[i+1].split('\t')[2])
		number = int(temp[1]) + 1
		rusc[temp[0]] = list()
		nrusc[temp[0]] = list()
		q[temp[0]] = list()
		lc[temp[0]] = list()
		for j in range(5):
			q[temp[0]].append(0)
			lc[temp[0]].append(0)
		casdic = {} #from tweet id to user id who replied it with which tweet id
		for j in range(i+1, i+number):
			tweet = data[i].split('\t')
			author[tweet[0]] = tweet[1]
			timestamp[tweet[0]] = int(tweet[2])
			if tweet[3] == '-1':
				depth[tweet[0]] = 0
				casdic[tweet[0]] = {}
			else:
				depth[tweet[0]] = depth[tweet[3]] + 1
				casdic[tweet[3]][tweet[1]] = tweet[0]
		for item in casdic:
			for f in friend[author[item]]:
				info = list()
				if f in casdic[item]: #this person retweeted it
					info.append(iddic[f])
					info.append(timestamp[casdic[item][f]] - timestamp[item])
					info.append(depth[item])
					rusc[temp[0]].append(info)
				else: #this person did not retweet it
					info.append(iddic[f])
					info.append(te - timestamp[item])
					info.append(depth[item])
					nrusc[temp[0]].append(info)
		i += number		


#Get lambda value
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
omega += sum(lbd) * 100 / users
theta1 += np.arccos(np.sqrt(0.2))
theta2 += np.arccos(np.sqrt(0.25))
theta3 += np.arccos(np.sqrt(1.0 / 3))
theta4 += np.arccos(np.sqrt(0.5))

fr = open(prefix+'relations'+suffix, 'r')
relation = fr.readlines()
n = len(relation)
i = 0
while i < n:
	temp = relation[i].split('\t')
	number = int(temp[1]) + 1
	friend[temp[0]] = list()
	for j in range(i+1, i+number):
		fd = relation[j].split('\t')
		if not edgemap.has_key(iddic[temp[0]]):
			edgemap[iddic[temp[0]]] = {}
		edgemap[iddic[temp[0]]][iddic[fd[1]]] = pos
		pos += 1
		pi.append(int(fd[2]) * 1.0 / posts[iddic[temp[0]]])
		x.append(1.0)
		friend[temp[0]].append(fd[1])
	i += number
pi = np.array(pi)
x = np.array(x)

#Read personal cascade file
n = len(uid)
for i in range(n):
	fr = open(prefix+'single_user_post/'+str(i)+'_'+uid[i]+suffix, 'w')
	single = fr.readlines()
	SingleObj(single, i)




