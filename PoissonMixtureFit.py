import sys
import scipy as sp
import numpy as np
import tensorflow as tf

users = 7268 #total number of users 
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

s1 = np.zeros(users) #parameter scaler list
s2 = np.zeros(users) #parameter scaler list
s3 = np.zeros(users) #parameter scaler list
s4 = np.zeros(users) #parameter scaler list
s5 = np.zeros(users) #parameter scaler list
t1 = np.zeros(users) + 0.25
t2 = np.zeros(users) + 0.375
t3 = np.zeros(users) + 0.5625
t4 = np.zeros(users) + 0.75
t5 = np.zeros(users) + 23.0 / 24
sum_iet = np.zeros(users) #sum of inter envet time
posts = np.zeros(users) #total posts of users
lnorderlist = np.array(7000) #ln result of order to int
indexlist = np.linspace(0, 7267, 7268, dtype=int)
lastlist = np.linspace(86439, 86400 * 7268 - 1, 7268, dtype=int)
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
		scaler.append(param[i*allusers:(i+1)*allusers])
	timecut = list()
	for i in range(5):
		timecut.append(param[(i+5)*allusers:(i+6)*allusers])
	return scaler, timecut

def ObjLnPiQ(p):
	global gamma
	scaler = p[:5*allusers]
	timerate = p[5*allusers:]
	timepoint = int(timerate * 86400)
	before = list()
	kmatrix = list()
	ietlist = list()
	scalerlist = list()
	for j in range(5):
		fixtp = timepoint[j*allusers:(j+1)*allusers] + indexlist * 86400
		before.append(tf.gather(nummarix, fixtp, axis=1))
		if j == 0:
			kmatrix.append(before[j])
			ietlist.append(timepoint[:allusers])
		else:
			kmatrix.append(before[j] - before[j-1])
			ietlist.append(timepoint[j*allusers:(j+1)*allusers] - timepoint[(j-1)*allusers:j*allusers])
		scalerlist.append(scaler[j*allusers:(j+1)*allusers])
	kmatrix[0] += tf.gather(nummarix, lastlist, axis=1) - kmatrix[-1]
	ietlist[0] += 86400 - timepoint[4*allusers:]
	lomatrix = tf.gather(lnorderlist, kmatrix)
	r = scalerlist * ietlist + lomatrix - kmatrix * (tf.log(scalerlist) + tf.log(ietlist))
	return tf.reduce_sum(r)

def Derivative():
	return sum_iet - posts / lbd - gamma / lbd + gamma / (1 - lbd)

def DeltaSum():
	for i in range(10):
		for j in range(users):
			for item in postlist[i][j]:
				nummarix[i][j][item] += 1
	nummarix = np.array(nummarix)
	nummarix = np.cumsum(nummarix, axis=2)
	for i in range(10):
		nummarix [i] = nummarix[i].flatten()
	return

prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'
fr = open(prefix+'posttimes'+suffix, 'r')
post = fr.readlines()
for i in range(users):
	temp = post[i].split('\t')
	uid.append(temp[0])
	posts[i] = int(temp[1])
fr.close()
#lbd += sum(posts) * 1.0 / 7268 / 86400 / 10
s1 += posts * 1.0 / 7268 / 86400 / 10
s2 += posts * 1.0 / 7268 / 86400 / 10
s3 += posts * 1.0 / 7268 / 86400 / 10
s4 += posts * 1.0 / 7268 / 86400 / 10
s5 += posts * 1.0 / 7268 / 86400 / 10
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
		print str(day) + ' ' + str(iddic[userid])
		second = posttime % 86400
		postlist[day][iddic[userid]].append(second)
	i += size
DeltaSum()
lnorder = 0
for k in range(1, 7000):
	lnorder += np.log(k)
	lnorderlist[k] = lnorder
#print min(sum_iet)

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
cnt = 0
lastobj = 10000000000
param = Joint(s1, s2, s3, s4, s5, t1, t2, t3, t4, t5)
p = tf.Variable(param, name='p')
#alpha = tf.Variable(alpha, dtype=tf.float64)
optimizer = tf.train.GradientDescentOptimizer(alpha)
target = ObjLnPiQ(p)
train = optimizer.minimize(target)
init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
	while cnt < 100000000:
		obj, p, _ = session.run([target, p, train])
		#lbd = session.run(l)
		#obj = session.run(train)
		if lastobj - obj < 0.0000001:
			break
		cnt += 1
		if cnt % 10000 == 0:
			print obj
		lastobj = obj
	print lastobj - obj / users
	print cnt

print lastobj

scaler, timecut = Resolver(p)

fw = open(prefix+'lambda_Mixture'+suffix, 'w')
for i in range(users):
	fw.write(uid[i])
	for j in range(5):
		fw.write('\t')
		fw.write(str(scaler[j][i]))
	fw.write('\n')
fw.close()

fw = open(prefix+'timecut_Mixture'+suffix, 'w')
for i in range(users):
	fw.write(uid[i])
	for j in range(5):
		fw.write('\t')
		fw.write(str(int(timecut[j][i] * 86400)))
	fw.write('\n')
fw.close()

