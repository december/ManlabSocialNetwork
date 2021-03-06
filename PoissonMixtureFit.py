import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import scipy as sp
import numpy as np
import tensorflow as tf

users = 7625 #total number of users 
ts = 1321286400 #starting timestamps
uid = list() #id of the user
iddic = {} #from user id to user index
postlist = list() #from date to user index to post time
nummarix = list() #form date tp user index to number before time
for i in range(10):
	temp1 = list()
	temp2 = list()
	for j in range(users):
		temp1.append([])
		temp2.append(np.zeros(86400))
	postlist.append(temp1)
	nummarix.append(temp2)

s = np.zeros(5) #parameter scaler list
t = np.zeros(5) #parameter timecut list
t[0] = 0.25
t[1] = 0.375
t[2] = 0.5625
t[3] = 0.75
t[4] = 23.0 / 24
sum_iet = np.zeros(users) #sum of inter envet time
posts = np.zeros(users) #total posts of users
lnorderlist = np.zeros(7000) #ln result of order to int
indexlist = np.linspace(0, users-1, users, dtype=int)
lastlist = np.linspace(86439, 86400 * users - 1, users, dtype=int)
delta = 0.000000001 #when will the algorithm stop
alpha = float(sys.argv[1]) #learning rate
gamma = 1 #log barrier function

def Joint(s1, s2, s3, s4, s5, t1, t2, t3, t4, t5):
	param = np.append(s1, s2)
	param = np.append(param, s3)
	param = np.append(param, s4)
	param = np.append(param, s5)
	param = np.append(param, t1)
	param = np.append(param, t2)
	param = np.append(param, t3)
	param = np.append(param, t4)
	param = np.append(param, t5)
	return param

def Resolver(param):
	scaler = list()
	for i in range(5):
		scaler.append(param[i*users:(i+1)*users])
	timecut = list()
	for i in range(5):
		timecut.append(param[(i+5)*users:(i+6)*users])
	return scaler, timecut

def ObjLnPiQ(p, nm):
	global lnorderlist
	scaler = p[:5]
	timerate = p[5:]
	timepoint = tf.cast(timerate * 86400, tf.int32)
	ietlist = list()
	kmatrix = list()
	oneday = tf.constant(86400)
	for j in range(5):
		if j == 0:
			#print ietlist[j]
			#print timepoint[j].eval()
			ietlist.append(timepoint[0] + oneday - timepoint[-1])
			kmatrix.append(tf.gather(nm, timepoint[j], axis=1) + tf.gather(nm, 86399, axis=1) - tf.gather(nm, timepoint[-1], axis=1))
		else:
			ietlist.append(timepoint[j] - timepoint[j-1])
			kmatrix.append(tf.gather(nm, timepoint[j], axis=1) - tf.gather(nm, timepoint[j-1], axis=1))
	ietlist = np.array(ietlist)
	kmatrix = np.array(kmatrix)
	#print kmatrix
	r = 0

	for j in range(5):
		r += tf.reduce_sum(tf.gather(lnorderlist, kmatrix[j]))
		r += scaler[j] * tf.cast(ietlist[j], tf.float64) * 10
		tempr = tf.log(scaler[j]) + tf.log(tf.cast(ietlist[j], tf.float64))
		r -= tempr * tf.cast(tf.reduce_sum(kmatrix[j]), tf.float64)
	'''
	before = list()
	kmatrix = list()
	ietlist = np.zeros(5)
	scalerlist = list()
	for j in range(5):
		fixtp = timepoint[j*users:(j+1)*users] + indexlist * 86400
		before.append(tf.gather(nummarix, fixtp, axis=1))
		if j == 0:
			kmatrix.append(before[j])
			ietlist.append(timepoint[:users])
		else:
			kmatrix.append(before[j] - before[j-1])
			ietlist.append(timepoint[j*users:(j+1)*users] - timepoint[(j-1)*users:j*users])
		scalerlist.append(scaler[j*users:(j+1)*users])
	kmatrix[0] += tf.gather(nummarix, lastlist, axis=1) - kmatrix[-1]
	ietlist[0] += 86400 - timepoint[4*users:]
	lomatrix = tf.gather(lnorderlist, kmatrix)
	'''
	#r -= tf.reduce_sum(tf.transpose(kmatrix) * (tf.log(scaler) + tf.log(ietlist)))
	
	return r

def Derivative():
	return sum_iet - posts / lbd - gamma / lbd + gamma / (1 - lbd)

def DeltaSum():
	global nummarix
	for i in range(10):
		for j in range(users):
			for item in postlist[i][j]:
				nummarix[i][j][item] += 1
	nummarix = np.array(nummarix)
	nummarix = np.cumsum(nummarix, axis=2)
	#tempnum = list()
	#for i in range(10):
	#	tempnum.append(nummarix[i].flatten())
	#nummarix = np.array(tempnum)
	return

def CalcInitial(t):
	global nummarix
	timepoint = t * 86400
	timepoint = [int(k) for k in timepoint]
	inits = list()
	for i in range(users):
		inits.append(np.zeros(5))
		for j in range(5):
			totalk = 0
			for k in range(10):
				if j == 0:
					totalk += nummarix[k][i][timepoint[j]] + nummarix[k][i][-1] - nummarix[k][i][timepoint[-1]]
				else:
					totalk += nummarix[k][i][timepoint[j]] - nummarix[k][i][timepoint[j-1]]
			if j == 0:
				deltat = timepoint[j] + 86400 - timepoint[-1]
			else:
				deltat = timepoint[j] - timepoint[j-1]
			inits[i][j] = totalk * 1.0 / 10 / deltat
	return inits


def GiveMatrix(i):
	global nummarix
	temp = list()
	for j in range(10):
		temp.append(nummarix[j][i])
	return np.array(temp)

prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'
fr = open(prefix+'posttimes'+suffix, 'r')
post = fr.readlines()
for i in range(7268):
	temp = post[i].split('\t')
	uid.append(temp[0])
	posts[i] = int(temp[1])
fr.close()
#lbd += sum(posts) * 1.0 / 7268 / 86400 / 10
s += sum(posts) * 1.0 / 7268 / 86400 / 10
#print lnorder

cascades = list()
for i in range(100):
	fr = open(prefix+str(i)+suffix, 'r')
	cascade = fr.readlines()
	cascades.extend(cascade)
	fr.close()
n = len(cascades)
allusers = 0
i = 0
while i < n:
	line = cascades[i].split('\t')
	size = int(line[1]) + 1
	for j in range(i+1, i+size):
		userid = cascades[j].split('\t')[1]
		posttime = int(cascades[j].split('\t')[2])
		if not iddic.has_key(userid):
			iddic[userid] = allusers
			uid.append(userid)
			allusers += 1
		day = (posttime - ts) / 86400
		#print str(day) + ' ' + str(iddic[userid]) + str(i * 1.0 / n)
		#print iddic
		second = posttime % 86400
		postlist[day][iddic[userid]].append(second)
	i += size
#print allusers	
print 'Finish reading.'
DeltaSum()
lnorder = 0
for k in range(1, 7000):
	lnorder += np.log(k)
	lnorderlist[k] = lnorder
#print min(sum_iet)
print 'Finish calculating.'
'''
#Optimize manually
cnt = 0
lastobj = np.zeros(users) + 10000
while cnt < 100000:
	gd = Derivative()
	lbd = lbd - alpha * gd
	obj = ObjLnPiQ(lbd)
	if sum(lastobj - obj) < delta * users:
		break
	cnt += 1
	print sum(lastobj - obj) / users
	print cnt
	lastobj = obj
print lastobj
'''
#Optimize with tensorflow
scaler = list()
timecut = list()
origins = CalcInitial(t)
fw = open(prefix+'scaler_Origin'+suffix, 'w')
for i in range(users):
	average = origins[i][0] * (0.25 + 1.0 / 24) + origins[i][1] * 0.125 + origins[i][2] * 0.1875 + origins[i][3] * 0.1875 + origins[i][4] * (23.0 / 24 - 0.75)
	fw.write(uid[i])
	for j in range(5):
		fw.write('\t')
		fw.write(str(origins[i][j] / average))
	fw.write('\n')
fw.close()
print 'Begin to train.'
for i in range(users):
	cnt = 0
	lastobj = 10000000000
	param = np.append(origins[i], t)
	p = tf.Variable(param, name='p', dtype=tf.float64)
	nm = tf.placeholder(tf.int32, name='nm', shape=(10, 86400))
	#nummarix = tf.constant(nummarix, dtype=tf.int32)
	#alpha = tf.Variable(alpha, dtype=tf.float64)
	optimizer = tf.train.GradientDescentOptimizer(alpha)
	target = ObjLnPiQ(p, nm)
	train = optimizer.minimize(target)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		#param = tf.concat(tf.constant(origins[i], dtype=tf.float64), tf.constant(t, dtype=tf.float64))
		#p = tf.Variable(param, name='p', dtype=tf.float64)
		while cnt < 1000000:
			obj, newp, _ = session.run([target, p, train], feed_dict={nm:GiveMatrix(i)})
			#lbd = session.run(l)
			#obj = session.run(train)
			if abs(lastobj - obj) < 0.00001:
				break
			cnt += 1
			if cnt % 10000 == 0:
				print obj
			lastobj = obj
		print lastobj - obj
		print cnt
		print 'No. ' + str(i) + ' user learned.'
		scaler.append(newp[:5])
		timecut.append(newp[5:])
	print lastobj

print 'Begin to write.'
#scaler, timecut = Resolver(p)

fw = open(prefix+'lambda_Mixture'+suffix, 'w')
for i in range(users):
	fw.write(uid[i])
	for j in range(5):
		fw.write('\t')
		fw.write(str(scaler[i][j]))
	fw.write('\n')
fw.close()

fw = open(prefix+'timecut_Mixture'+suffix, 'w')
for i in range(users):
	fw.write(uid[i])
	for j in range(5):
		fw.write('\t')
		fw.write(str(int(timecut[i][j] * 86400)))
	fw.write('\n')
fw.close()

