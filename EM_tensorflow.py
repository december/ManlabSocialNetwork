import sys
import scipy as sp
import numpy as np
import tensorflow as tf
import scipy.optimize
import numpy.random
import datetime

single = True
filename = int(sys.argv[1])
if filename < 0:
	single = False

users = 7268
allusers = 7268
ts = 1321286400 #start timestamps
te = 1322150400 #end timestamps
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
alpha = 0.0000001 #learning rate for optimizer

gamma = -1.0 #log barrier
epsilon = 10.0 #when will EM stop
lbd = np.zeros(users) #parameter lambda which have calculated before

def Joint(omega, pi, x, theta1, theta2, theta3, theta4):
	param = np.append(omega, pi)
	param = np.append(param, x)
	param = np.append(param, theta1)
	param = np.append(param, theta2)
	param = np.append(param, theta3)
	param = np.append(param, theta4)
	return param

def Resolver(param):
	omega = param[:poslist[0]]
	pi = param[poslist[0]:poslist[1]]
	x = param[poslist[1]:poslist[2]]
	theta1 = param[poslist[2]:poslist[3]]
	theta2 = param[poslist[3]:poslist[4]]
	theta3 = param[poslist[4]:poslist[5]]
	theta4 = param[poslist[5]:]
	return omega, pi, x, theta1, theta2, theta3, theta4

def Select(omega, pi, x, theta1, theta2, theta3, theta4):
	p = list()
	for i in range(vnum):
		p.append(omega[vlist[i]])
	for i in range(enum):
		p.append(pi[elist[i]])
	for i in range(enum):
		p.append(x[elist[i]])
	for i in range(vnum):
		p.append(theta1[vlist[i]])	
	for i in range(vnum):
		p.append(theta2[vlist[i]])
	for i in range(vnum):
		p.append(theta3[vlist[i]])
	for i in range(vnum):
		p.append(theta4[vlist[i]])
	return Resolver(np.array(p))

def Phi(theta1, theta2, theta3, theta4, idx):
	if idx == 0:
		return tf.cos(theta1) * tf.cos(theta1)
	if idx == 1:
		return tf.sin(theta1) * tf.sin(theta1) * tf.cos(theta2) * tf.cos(theta2)
	if idx == 2:
		return tf.sin(theta1) * tf.sin(theta1) * tf.sin(theta2) * tf.sin(theta2) * tf.cos(theta3) * tf.cos(theta3)
	if idx == 3:
		return tf.sin(theta1) * tf.sin(theta1) * tf.sin(theta2) * tf.sin(theta2) * tf.sin(theta3) * tf.sin(theta3) * tf.cos(theta4) * tf.cos(theta4)
	return tf.sin(theta1) * tf.sin(theta1) * tf.sin(theta2) * tf.sin(theta2) * tf.sin(theta3) * tf.sin(theta3) * tf.sin(theta4) * tf.sin(theta4)

def Phi_np(theta1, theta2, theta3, theta4, idx):
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
	s = tf.log(lbd[vlist[uc]]) + tf.log(philist[tau][uc])
	for item in rusc[c]:
		edge = item[0]
		u = item[3]
		s += tf.log(omega[u]) - omega[u] * item[1] + tf.log(pi[edge]) - item[2] * tf.log(x[edge]) + tf.log(philist[tau][u])
	for item in nrusc[c]:
		edge = item[0]
		u = item[3]
		exponent = tf.maximum(-1 * omega[u] * item[1], -100)
		estimate = tf.exp(exponent) - 1
		#print edgemap[uc][u]
		result = 1 + pi[edge] * x[edge] ** (-1 * item[2]) * philist[tau][u] * estimate
		s += tf.log(result)
	return s

def QMatrix():
	n = len(q)
	qmx = list()
	for i in range(n):
		for j in range(5):
			qmx.append(q[clist[i]][j])
	qmx = tf.stack(qmx, 0)
	return tf.reshape(qmx, shape=(n, 5))

def QF(omega, pi, x, philist, c): #calculate q funciton with tricks
	for i in range(5):
		lc[c][i] = LnLc(omega, pi, x, philist, c, i)
	for i in range(5):
		s = 0
		for j in range(5):
			s += tf.exp(lc[c][j] - lc[c][i])
		q[c][i] = 1 / s

def ObjF(param, qm): #formulation of objective function (include barrier) (the smaller the better)
	omega, pi, x, theta1, theta2, theta3, theta4 = Resolver(param)
	omega = tf.cos(omega) * tf.cos(omega)
	pi = tf.cos(pi) * tf.cos(pi)
	x = x * x
	philist = list()
	for i in range(5):
		philist.append(Phi(theta1, theta2, theta3, theta4, i))
	#global total
	#total += 1
	noreply = 0
	'''
	print 'Begin'
	print omega
	print x
	print pi
	'''
	obj = (tf.reduce_sum(tf.log(omega)) + tf.reduce_sum(tf.log(x)) + tf.reduce_sum(tf.log(1-pi)) + tf.reduce_sum(tf.log(pi))) * gamma #need to be fixxed
	#obj = 0
	for c in q:
		if len(rusc[c]) == 0:
			if noreply == 0:
				for i in range(5):
					noreply -= qm[cdic[c]][i] * LnLc(omega, pi, x, philist, c, i)
					tmp = qm[cdic[c]][i] * tf.log(qm[cdic[c]][i])
					noreply += tf.cast(tmp, dtype=tf.float64)
			obj += noreply
			continue
		for i in range(5):
			obj -= qm[cdic[c]][i] * LnLc(omega, pi, x, philist, c, i)
			tmp = qm[cdic[c]][i] * tf.log(qm[cdic[c]][i])
			obj = obj + tf.cast(tmp, dtype=tf.float64)
	#if total % 10000 == 0:
	#	print 'No.' + str(total) + ' times: ' + str(obj)
	return obj

def EStep(omega, pi, x, theta1, theta2, theta3, theta4): #renew q and lc
	#print [len(omega), len(pi), len(x)]
	omega = tf.cos(omega) * tf.cos(omega)
	pi = tf.cos(pi) * tf.cos(pi)
	x = x * x
	#print [len(oc), len(pc), len(xc)]
	philist = list()
	for i in range(5):
		philist.append(Phi(theta1, theta2, theta3, theta4, i))
	#count = 0
	for c in q:
		QF(omega, pi, x, philist, c)
		#count += 1
		#print count
	return QMatrix()

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
			timestamp[tweet[0]] = int(tweet[2])
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
theta1 = np.zeros(allusers) #one of spherical coordinates of phi distribution
theta2 = np.zeros(allusers) #one of spherical coordinates of phi distribution
theta3 = np.zeros(allusers) #one of spherical coordinates of phi distribution
theta4 = np.zeros(allusers) #one of spherical coordinates of phi distribution

omega += sum(lbd) * 100 / users
omega = np.arccos(np.sqrt(omega))
'''
theta1 += np.arccos(np.sqrt(0.2))
theta2 += np.arccos(np.sqrt(0.25))
theta3 += np.arccos(np.sqrt(1.0 / 3))
theta4 += np.arccos(np.sqrt(0.5))
'''
tr = list()
for i in range(4):
	tr.append(np.random.rand())
print tr
theta1 += np.arccos(np.sqrt(tr[0]))
theta2 += np.arccos(np.sqrt(tr[1]))
theta3 += np.arccos(np.sqrt(tr[2]))
theta4 += np.arccos(np.sqrt(tr[3]))

#Read personal cascade file
print 'Read behavior log...'
for i in range(users):
	if single and i != filename:
		continue
	fr = open(prefix+'single_user_post/'+str(i)+'_'+uid[i]+suffix, 'r')
	singlefile = fr.readlines()
	SingleObj(singlefile, i)
	fr.close()
poslist.append(vnum)
poslist.append(vnum+enum)
poslist.append(vnum+enum*2)
for i in range(4):
	poslist.append(vnum*(i+2)+enum*2)
omega, pi, x, theta1, theta2, theta3, theta4 = Select(omega, pi, x, theta1, theta2, theta3, theta4)
print 'There are ' + str(vnum * 5) + ' point parameters and ' + str(enum * 2) + ' edge parameters to be learned...'
#Conduct EM algorithm
#QMatrix(q)
print 'EM algorithm begins...'
#print min(omega)
#print max(omega)
#print pi
cnt = 0
lastObj = np.exp(100)
param = Joint(omega, pi, x, theta1, theta2, theta3, theta4)
n = len(q)
p = tf.Variable(param, name='p')
qm = tf.placeholder(tf.float64, name='qm', shape=(n, 5))
optimizer = tf.train.GradientDescentOptimizer(alpha)
#optimizer = tf.train.AdamOptimizer(alpha)
target = ObjF(p, qm)
train = optimizer.minimize(target)
init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(device_count={"CPU":76})) as session:
	session.run(init)
	qf = EStep(omega, pi, x, theta1, theta2, theta3, theta4)
	while cnt < 100:
	#param = Joint(omega, pi, x, theta1, theta2, theta3, theta4)
	#start = datetime.datetime.now()
	#obj = ObjF(param)
	#end = datetime.datetime.now()
	#print (end - start).seconds
		out_qf = session.run(qf)
		print 'EStep ' + str(cnt+1) + ' finished...'
		for step in range(iters):
			session.run(train, feed_dict={qm:out_qf})
			newp = session.run(p, feed_dict={qm:out_qf})
			obj = session.run(target, feed_dict={qm:out_qf})
		print 'MStep ' + str(cnt+1) + ' finished...'
		print 'Objective function value: ' + str(obj)
		omega, pi, x, theta1, theta2, theta3, theta4 = Resolver(newp)
		if abs(lastObj) - obj < epsilon:
			break
		lastObj = obj	
		cnt += 1
		print 'Iteration ' + str(cnt) + ' finished...'
omega = np.cos(omega) * np.cos(omega)
pi = np.cos(pi) * np.cos(pi)
x = x * x

#Output parameters
if single:
	prefix = prefix + 'single_user_parameter/'
	suffix = '_' + str(filename) + suffix

print 'Output data files...'
fw = open(prefix+'omega_Poisson'+suffix, 'w')
for i in range(vnum):
	fw.write(uid[vlist[i]])
	fw.write('\t')
	fw.write(str(omega[i]))
	fw.write('\n')
fw.close()

fw = open(prefix+'pi_Poisson'+suffix, 'w')
for item in edgemap:
	for fd in edgemap[item]:
		if not edgemap[item][fd] in edic:
			continue
		fw.write(uid[item])
		fw.write('\t')
		fw.write(uid[fd])
		fw.write('\t')
		fw.write(str(pi[edic[edgemap[item][fd]]]))
		fw.write('\n')
fw.close()

fw = open(prefix+'x_Poisson'+suffix, 'w')
for item in edgemap:
	for fd in edgemap[item]:
		if not edgemap[item][fd] in edic:
			continue
		fw.write(uid[item])
		fw.write('\t')
		fw.write(uid[fd])
		fw.write('\t')
		fw.write(str(x[edic[edgemap[item][fd]]]))
		fw.write('\n')
fw.close()

for i in range(5):
	fw = open(prefix+'phi'+str(i)+'_Poisson'+suffix, 'w')
	phi = Phi_np(theta1, theta2, theta3, theta4, i)
	for j in range(vnum):
		fw.write(uid[vlist[j]])
		fw.write('\t')
		fw.write(str(phi[j]))
		fw.write('\n')
	fw.close()

