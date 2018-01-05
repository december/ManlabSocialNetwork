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
rusc = list() #info of rusc sets and records
nrusc = list() #info of nrusc sets and records
rusc_dic = {} #from cascade id to index list of rusc info
nrusc_dic = {} #from cascade id to index list of nrusc info
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
iters = 1000 #iteration times in each M-steps
alpha = 0.00001 #learning rate for optimizer

gamma = -1.0 #log barrier
epsilon = 1.0 #when will EM stop
lbd = np.zeros(users) #parameter lambda which have calculated before

def Joint(omega, pi, x):
	param = np.append(omega, pi)
	param = np.append(param, x)
	return param

def Resolver(param):
	omega = param[:poslist[0]]
	pi = param[poslist[0]:poslist[1]]
	x = param[poslist[1]:poslist[2]]
	return omega, pi, x

def Select(omega, pi, x):
	p = list()
	for i in range(vnum):
		p.append(omega[vlist[i]])
	for i in range(enum):
		p.append(pi[elist[i]])
	for i in range(enum):
		p.append(x[elist[i]])
	return Resolver(np.array(p))

def LnLc(omega, pi, x, c): #ln fromulation of one cascades's likelihood on tau(do not include part of Q)
	uc = vdic[iddic[author[c]]]
	s = tf.log(lbd[vlist[uc]])
	rc = tf.gather(rusc, rusc_dic[c], axis=0)
	nc = tf.gather(nrusc, nrusc_dic[c], axis=0)
	omega_rc = tf.gather(omega, tf.cast(rc[:, 3], dtype=tf.int64), axis=0)
	pi_rc = tf.gather(pi, tf.cast(rc[:, 0], dtype=tf.int64), axis=0)
	x_rc = tf.gather(x, tf.cast(rc[:, 0], dtype=tf.int64), axis=0)
	s += tf.reduce_sum(tf.log(omega_rc) - omega_rc * rc[:, 1] + tf.log(pi_rc) - rc[:, 2] * tf.log(x_rc))
	omega_nc = tf.gather(omega, tf.cast(nc[:, 3], dtype=tf.int64), axis=0)
	pi_nc = tf.gather(pi, tf.cast(nc[:, 0], dtype=tf.int64), axis=0)
	x_nc = tf.gather(x, tf.cast(nc[:, 0], dtype=tf.int64), axis=0)
	exponent = tf.maximum(-1 * omega_nc * nc[:, 1], -100)
	estimate = tf.exp(exponent) - 1
	result = 1 + pi_nc * x_nc ** (-1 * nc[:, 2]) * estimate
	s += tf.reduce_sum(tf.log(result))
	return s

def ObjF(param): #formulation of objective function (include barrier) (the smaller the better)
	omega, pi, x = Resolver(param)
	#omega = tf.cos(omega) * tf.cos(omega)
	#pi = tf.cos(pi) * tf.cos(pi)
	#x = x * x

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
				noreply -= LnLc(omega, pi, x, philist, c, i)
				tmp = tf.log(qm[cdic[c]][i])
				noreply += tf.cast(tmp, dtype=tf.float64)
			obj += noreply
			continue
		obj -= LnLc(omega, pi, x, philist, c, i)
		tmp = tf.log(qm[cdic[c]][i])
		obj = obj + tf.cast(tmp, dtype=tf.float64)
	#if total % 10000 == 0:
	#	print 'No.' + str(total) + ' times: ' + str(obj)
	return obj

def SingleObj(data, u):
	global vnum, enum, cnum, rusc_num, nrusc_num
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
		q[temp[0]] = list()
		lc[temp[0]] = list()
		for j in range(5):
			q[temp[0]].append(0.2)
			lc[temp[0]].append(0.0)
		lc[temp[0]] = np.array(lc[temp[0]])
		q[temp[0]] = np.array(q[temp[0]])
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
					rusc.append(info)
					rusc_dic[temp[0]].append(rusc_num)
					rusc_num += 1
				else: #this person did not retweet it
					info.append(edic[edgemap[iddic[author[item]]][iddic[f]]])
					info.append(te - timestamp[item])
					info.append(depth[item])
					info.append(vdic[iddic[f]])
					nrusc.append(info)
					nrusc_dic[temp[0]].append(nrusc_num)
					nrusc_num += 1
		cnum += 1
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
#pi = np.arccos(np.sqrt(pi))
x = np.array(x)

omega = np.zeros(allusers) #parameter omega
theta1 = np.zeros(allusers) #one of spherical coordinates of phi distribution
theta2 = np.zeros(allusers) #one of spherical coordinates of phi distribution
theta3 = np.zeros(allusers) #one of spherical coordinates of phi distribution
theta4 = np.zeros(allusers) #one of spherical coordinates of phi distribution

omega += sum(lbd) * 100 / users
#omega = np.arccos(np.sqrt(omega))

#Read personal cascade file
print 'Read behavior log...'
for i in range(users):
	if single and i != filename:
		continue
	fr = open(prefix+'single_user_post/'+str(i)+'_'+uid[i]+suffix, 'r')
	single = fr.readlines()
	SingleObj(single, i)
	fr.close()
poslist.append(vnum)
poslist.append(vnum+enum)
poslist.append(vnum+enum*2)
for i in range(4):
	poslist.append(vnum*(i+2)+enum*2)
omega, pi, x = Select(omega, pi, x)
print 'There are ' + str(vnum) + ' point parameters and ' + str(enum * 2) + ' edge parameters to be learned...'
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
rusc = tf.constant(rusc, dtype=tf.float64)
nrusc = tf.constant(nrusc, dtype=tf.float64)
for key in rusc_dic:
	rusc_dic[key] = tf.constant(rusc_dic[key], dtype=tf.int64)
	nrusc_dic[key] = tf.constant(nrusc_dic[key], dtype=tf.int64)
p = tf.Variable(param, name='p')
optimizer = tf.train.GradientDescentOptimizer(alpha)
#optimizer = tf.train.AdamOptimizer(alpha)
target = ObjF(p)
train = optimizer.minimize(target)
init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
	#param = Joint(omega, pi, x, theta1, theta2, theta3, theta4)
	#start = datetime.datetime.now()
	#obj = ObjF(param)
	#end = datetime.datetime.now()
	#print (end - start).seconds
	for step in range(iters):
		session.run(train)
		newp = session.run(p)
		obj = session.run(target)
		print 'Iteration ' + str(cnt+1) + ' finished...'
		if abs(lastObj) - obj < epsilon:
			break
		lastObj = obj
	print 'Objective function value: ' + str(obj)
	omega, pi, x = Resolver(newp)
#omega = np.cos(omega) * np.cos(omega)
#pi = np.cos(pi) * np.cos(pi)
#x = x * x

#Output parameters
if single:
	prefix = prefix + 'single_user_parameter_notopic/'
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

