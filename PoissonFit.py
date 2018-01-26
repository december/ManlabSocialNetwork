import sys
import scipy as sp
import numpy as np
import tensorflow as tf

users = 7268 #total number of users 
ts = 1321286400 #starting timestamps
uid = list() #id of the user
lbd = np.zeros(users) #parameter lambda list
sum_iet = np.zeros(users) #sum of inter envet time
posts = np.zeros(users) #total posts of users
lnorder = np.zeros(users) #ln result of order to posts
ptdic = {} #post time list
delta = 0.000000001 #when will the algorithm stop
alpha = float(sys.argv[1]) #learning rate
gamma = 1 #log barrier function

def ObjLnPiQ(lbd):
	global gamma
	r = lbd * sum_iet + lnorder - posts * tf.log(lbd) - posts * tf.log(sum_iet) - gamma * (tf.log(lbd) + tf.log(1 - lbd))
	return tf.reduce_sum(r)

def Derivative():
	return sum_iet - posts / lbd - gamma / lbd + gamma / (1 - lbd)

def DeltaSum(pt):
	s = pt[0] - ts
	n = len(pt)
	for i in range(1, n):
		s += pt[i] - pt[i-1]
	return s

prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'
fr = open(prefix+'posttimes'+suffix, 'r')
post = fr.readlines()
for i in range(users):
	temp = post[i].split('\t')
	uid.append(temp[0])
	posts[i] = int(temp[1])
fr.close()
lbd += sum(posts) * 1.0 / 7268 / 86400 / 4

for i in range(users):
	pos = int(posts[i] + 1)
	for j in range(2, pos):
		lnorder[i] += np.log(j)
#print lnorder

cascades = list()
for i in range(100):
	fr = open(prefix+str(i)+suffix, 'r')
	cascade = fr.readlines()
	cascades.extend(cascade)
	fr.close()
n = len(cascades)
i = 0
while i < n:
	line = cascades[i].split('\t')
	size = int(line[1]) + 1
	userid = cascades[i+1].split('\t')[1]
	posttime = int(cascades[i+1].split('\t')[2])
	if not ptdic.has_key(userid):
		ptdic[userid] = list()
	ptdic[userid].append(posttime)
	i += size;
i = 0
for i in range(users):
	sum_iet[i] = DeltaSum(ptdic[uid[i]])
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
l = tf.Variable(lbd, name='l')
#alpha = tf.Variable(alpha, dtype=tf.float64)
optimizer = tf.train.GradientDescentOptimizer(alpha)
target = ObjLnPiQ(l)
train = optimizer.minimize(target)
init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)
	while cnt < 100000000:
		obj, lbd, _ = session.run([target, l, train])
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

fw = open(prefix+'lambda_Poisson'+suffix, 'w')
for i in range(users):
	fw.write(uid[i])
	fw.write('\t')
	fw.write(str(lbd[i]))
	fw.write('\n')
fw.close()

