import sys
import scipy as sp
import numpy as np
import tensorflow as tf
import scipy.optimize
import numpy.random
import datetime

def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()
import signal
signal.signal(signal.SIGINT, debug_signal_handler)

single = True
filename = int(sys.argv[1])
if filename < 0:
	single = False
alpha = float(sys.argv[2]) #learning rate for optimizer

users = 7268
allusers = 7268
ts = 1321286400 #start timestamps
te = 1322150400 #end timestamps
uid = list() #from user index to user id
iddic = {} #from user id to user index
friend = {} #from user id to its followers' user id
rusc = list() #info part of rusc sets and records
nrusc = list() #info part of nrusc sets and records
rusc_id = list() #id part of rusc sets and records
nrusc_id = list() #id part of nrusc sets and records
rusc_dic = list() #from cascade id to index list of rusc info
nrusc_dic = list() #from cascade id to index list of nrusc info
begin_rusc = list()
end_rusc = list()
begin_nrusc = list()
end_nrusc = list()
depth = {} #from tweet id to depth
author = {} #from tweet id to user id
cascade_author = list()
timestamp = {} #from tweet id to timestamp
posts = {} #from user index to post times
q = list() #from cascade id to q function
tempq = list()
lc = list() #from cascade id to log-likelihood function value
cdic = {} #from cascade id to cascade index
clist = list() #from cascade index to cascade id
edgemap = {} #from relations to the index of edge
vdic = {} #from user index to the index of point parameter 
edic = {} #from the index of edge to the index of edge parameter
vlist = list() #from the index of point parameter to user index
vlist_tf = list()
elist = list() #from the index of edge parameter to the index of edge
vnum = 0
enum = 0
cnum = 0
rusc_num = 0
nrusc_num = 0
pos = 0
poslist = list()
total = 0
iters = 1 #iteration times in each M-steps

gamma = -1.0 #log barrier
epsilon = 10.0 #when will EM stop
lbd = np.zeros(users) #parameter lambda which have calculated before
count = 0

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
	p.append(x[0])
	return Resolver(np.array(p))

def LnLc(omega, pi, x, c): #ln fromulation of one cascades's likelihood on tau(do not include part of Q)
	uc = cascade_author[c]
	tmplbd = tf.log(lbd[vlist_tf[uc]])
	tmpphi = philist[uc]
	s = tf.cast(tf.log(tmpphi) + tmplbd, dtype=tf.float64)
	#print tf.shape(s)

	br = begin_rusc[c]
	bn = begin_nrusc[c]
	er = end_rusc[c]
	en = end_nrusc[c]
	rc = tf.gather(rusc, rusc_dic[br:er], axis=0)
	nc = tf.gather(nrusc, nrusc_dic[bn:en], axis=0)
	rc_id = tf.gather(rusc_id, rusc_dic[br:er], axis=0)
	nc_id = tf.gather(nrusc_id, nrusc_dic[bn:en], axis=0)

	omega_rc = tf.gather(omega, rc_id[:, 1], axis=0)
	pi_rc = tf.gather(pi, rc_id[:, 0], axis=0)
	#x_rc = tf.gather(x, rc_id[:, 0], axis=0)
	#phi_rc = tf.gather(philist, rc_id[:, 1], axis=0)
	oldtmp = tf.reduce_sum(tf.log(omega_rc) - omega_rc * rc[:, 0] + tf.log(pi_rc) - tf.log(rc[:, 1]) * x)
	#print oldtmp.get_shape()

	s += oldtmp
	#s += tf.reduce_sum(tf.log(phi_rc), 0)	

	omega_nc = tf.gather(omega, nc_id[:, 1], axis=0)
	pi_nc = tf.gather(pi, nc_id[:, 0], axis=0)
	#x_nc = tf.gather(x, nc_id[:, 0], axis=0)
	exponent = tf.maximum(-1 * omega_nc * nc[:, 0], -100)
	estimate = tf.exp(exponent) - 1
	tmp = pi_nc * nc[:, 1] ** (-1 * x) * estimate
	newtmp = tf.log(1 + tf.reshape(tmp, (-1, 1)))
	#print newtmp.get_shape()
	#print s.get_shape()
	s += tf.reduce_sum(newtmp, 0)

	return s

def printInfo(obj, i, noreply):
	print str(i) + ' ' + str(obj) + ' ' + str(noreply)

def cond(obj, i, noreply, omega, pi, x):
	return i < q.get_shape()[0]

def body(obj, i, noreply, omega, pi, x):
	#if rusc_dic[i].get_shape()[0] == 0:
	if begin_rusc[i] == end_rusc[i]:
		if noreply == 1:
			#noreply += tf.reduce_sum(qm[i] * tf.log(qm[i]))
			noreply -= tf.reduce_sum(LnLc(omega, pi, x, i))
		obj += noreply
	else:
		#obj += tf.reduce_sum(qm[i] * tf.log(qm[i]))
		obj -= tf.reduce_sum(LnLc(omega, pi, x, i))
	i += 1
	#tf.py_func(printInfo, [obj, i, noreply], tf.float64)
	return obj, i, noreply, omega, pi, x

def ObjF(param): #formulation of objective function (include barrier) (the smaller the better)
	omega, pi, x = Resolver(param)
	omega = tf.cos(omega) * tf.cos(omega)
	pi = tf.cos(pi) * tf.cos(pi)
	#x = x * x
	#global total
	#total += 1
	it = tf.cast(0, tf.int32)
	noreply = tf.cast(0.0, tf.float64)
	#it = tf.Variable(0)
	#noreply = tf.Variable(0.0)
	'''
	print 'Begin'
	print omega
	print x
	print pi
	'''
	obj = (tf.reduce_sum(tf.log(omega)) + tf.reduce_sum(tf.log(x)) + tf.reduce_sum(tf.log(1-pi)) + tf.reduce_sum(tf.log(pi))) * gamma #need to be fixxed
	#obj = 0
	newobj, _, _, _, _, _, _, _ = tf.while_loop(cond, body, [obj, it, noreply, omega, pi, x], parallel_iterations=80)
		
	#if total % 10000 == 0:
	#	print 'No.' + str(total) + ' times: ' + str(obj)
	return newobj

def SingleObj(data, u):
	global vnum, enum, cnum, rusc_num, nrusc_num
	n = len(data)
	#last = int(data[1].split('\t')[2])
	i = 0
	while i < n:
		temp = data[i].split('\t')
		number = int(temp[1]) + 1
		rusc_dic.append(list())
		nrusc_dic.append(list())
		clist.append(temp[0])
		cdic[temp[0]] = cnum
		q.append([0.2, 0.2, 0.2, 0.2, 0.2])
		#lc.append([0.0, 0.0, 0.0, 0.0, 0.0])
		#lc[temp[0]] = np.array(lc[temp[0]])
		#q[temp[0]] = np.array(q[temp[0]])
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
				info_id = list()
				if f in casdic[item]: #this person retweeted it
					info_id.append(edic[edgemap[iddic[author[item]]][iddic[f]]])
					info.append(timestamp[casdic[item][f]] - timestamp[item])
					info.append(depth[item])
					info_id.append(vdic[iddic[f]])
					rusc.append(info)
					rusc_id.append(info_id)
					rusc_dic[cdic[temp[0]]].append(rusc_num)
					rusc_num += 1
				else: #this person did not retweet it
					info_id.append(edic[edgemap[iddic[author[item]]][iddic[f]]])
					info.append(te - timestamp[item])
					info.append(depth[item])
					info_id.append(vdic[iddic[f]])
					nrusc.append(info)
					nrusc_id.append(info_id)
					nrusc_dic[cdic[temp[0]]].append(nrusc_num)
					nrusc_num += 1
		cnum += 1
		i += number		


#Get lambda value
starttime = datetime.datetime.now()
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
		#x.append(1.0)
		friend[temp[0]].append(fd[1])
	i += number
fr.close()
pi = np.array(pi)
pi = np.arccos(np.sqrt(pi))
x = np.array([3])

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
	fr = open(prefix+'single_user_post/'+str(i)+'_'+uid[i]+suffix, 'r')
	singlefile = fr.readlines()
	SingleObj(singlefile, i)
	fr.close()
poslist.append(vnum)
poslist.append(vnum+enum)
poslist.append(vnum+enum+1)
for i in range(4):
	poslist.append(vnum*(i+2)+enum+1)
omega, pi, x = Select(omega, pi, x)
print 'There are ' + str(vnum) + ' point parameters and ' + str(enum + 1) + ' edge parameters to be learned...'
#Conduct EM algorithm
#QMatrix(q)
for c in clist:
	cascade_author.append(vdic[iddic[author[c]]])
print 'EM algorithm begins...'
#print min(omega)
#print max(omega)
#print pi
cnt = 0
lastObj = np.exp(100)
param = Joint(omega, pi, x)
n = len(q)
#lc = np.array(lc)
#q = np.array(q)
#lc = tf.convert_to_tensor(np.array(lc.values()), dtype=tf.float64)
#q = tf.convert_to_tensor(np.array(q), dtype=tf.float64)


temp_rusc = list()
temp_pos = 0
for l in rusc_dic:
	begin_rusc.append(temp_pos)
	temp_pos += len(l)
	end_rusc.append(temp_pos)
	temp_rusc.extend(l)

temp_nrusc = list()
temp_pos = 0
for l in nrusc_dic:
	begin_nrusc.append(temp_pos)
	temp_pos += len(l)
	end_nrusc.append(temp_pos)
	temp_nrusc.extend(l)

#rusc_dic = np.array(rusc_dic.values())
#nrusc_dic = np.array(nrusc_dic.values())
cascade_author = tf.constant(cascade_author, dtype=tf.int32)
vlist_tf = tf.constant(vlist, dtype=tf.int32)
lbd = tf.constant(lbd, dtype=tf.float64)
rusc = tf.constant(rusc, dtype=tf.float64)
nrusc = tf.constant(nrusc, dtype=tf.float64)
rusc_id = tf.constant(rusc_id, dtype=tf.int32)
nrusc_id = tf.constant(nrusc_id, dtype=tf.int32)
rusc_dic = tf.constant(temp_rusc, dtype=tf.int32)
nrusc_dic = tf.constant(temp_nrusc, dtype=tf.int32)
begin_rusc = tf.constant(begin_rusc, dtype=tf.int32)
begin_nrusc = tf.constant(begin_nrusc, dtype=tf.int32)
end_rusc = tf.constant(end_rusc, dtype=tf.int32)
end_nrusc = tf.constant(end_nrusc, dtype=tf.int32)
#for key in rusc_dic:
#	rusc_dic[key] = tf.constant(rusc_dic[key], dtype=tf.int64)
#	nrusc_dic[key] = tf.constant(nrusc_dic[key], dtype=tf.int64)
print 'Graph construction completed.'
p = tf.Variable(param, name='p')
q = tf.Variable(np.array(q), dtype=tf.float64)
qm = tf.placeholder(tf.float64, name='qm', shape=(n, 5))
if alpha > 0:
	optimizer = tf.train.GradientDescentOptimizer(alpha)
else:
	optimizer = tf.train.AdamOptimizer(learning_rate=-alpha)
#optimizer = tf.train.AdamOptimizer(alpha)d
target = ObjF(p, qm)
train = optimizer.minimize(target)
init = tf.global_variables_initializer()
print 'Ready to calculate.'

if single:
	prefix = prefix + 'single_user_parameter/'
	suffix = '_' + str(filename) + suffix

def Output(omega, pi, x):
	print 'Output data files...'
	fw = open(prefix+'omega_Poisson_notopic'+suffix, 'w')
	for i in range(vnum):
		fw.write(uid[vlist[i]])
		fw.write('\t')
		fw.write(str(omega[i]))
		fw.write('\n')
	fw.close()

	fw = open(prefix+'pi_Poisson_notopic'+suffix, 'w')
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

	x = np.zeros(len(pi)) + x
	fw = open(prefix+'x_Poisson_notopic'+suffix, 'w')
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

with tf.Session() as session:
	session.run(init)
	#qf = EStep(omega, pi, x, theta1, theta2, theta3, theta4)
	print 'EStep part construction finished.'
	#total = begin_rusc.get_shape()[0]
	#same = 0
	#for i in range(total):
	#	if session.run(begin_rusc[i]) == session.run(end_rusc[i]):
	#		same += 1
	#print same
	while cnt < 100:
	#param = Joint(omega, pi, x, theta1, theta2, theta3, theta4)
	#start = datetime.datetime.now()
	#obj = ObjF(param)
	#end = datetime.datetime.now()
	#print (end - start).seconds
		#out_qf = session.run(qf)
		#print 'EStep ' + str(cnt+1) + ' finished...'
		for step in range(iters):
			session.run(train)
			newp = session.run(p)
			obj = session.run(target)
		#print 'MStep ' + str(cnt+1) + ' finished...'
		print 'Objective function value: ' + str(obj)
		if str(obj) == 'nan':
			break
		#print str(it) + ' ' + str(noreply)
		omega, pi, x = Resolver(newp)
		Output(np.cos(omega) * np.cos(omega), np.cos(pi) * np.cos(pi), x)
		#print omega[:10]
		if abs(lastObj) - obj < epsilon:
			break
		lastObj = obj	
		cnt += 1
		print 'Iteration ' + str(cnt) + ' finished...'
omega = np.cos(omega) * np.cos(omega)
pi = np.cos(pi) * np.cos(pi)
x = x

#Output parameters
Output(omega, pi, x)

endtime = datetime.datetime.now()
print 'Time consumed: ' + str(endtime - starttime) + ' (' + str(alpha) + ')'