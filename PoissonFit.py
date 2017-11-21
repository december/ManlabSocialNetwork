import scipy as sp
import numpy as np

users = 7268 #total number of users 
ts = 1321286400 #starting timestamps
uid = list() #id of the user
lbd = np.zeros(users) #parameter lambda list
sum_iet = np.zeros(users) #sum of inter envet time
posts = np.zeros(users) #total posts of users
ptdic = {} #post time list
delta = 0.000000001 #when will the algorithm stop
alpha = 0.00000000001 #learning rate
gamma = 1 #log barrier function

def ObjLnPiQ():
	return lbd * sum_iet - posts * np.log(lbd) - gamma * (np.log(lbd) + np.log(1 - lbd))

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
lbd += sum(posts) * 1.0 / 7268 / 86400

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

cnt = 0
lastobj = np.zeros(users) + 10000
while cnt < 100000:
	gd = Derivative()
	lbd = lbd - alpha * gd
	obj = ObjLnPiQ()
	if sum(lastobj - obj) < delta * users:
		break
	cnt += 1
	print sum(lastobj - obj) / users
	print cnt
	lastobj = obj
print lastobj

fw = open(prefix+'lambda_Poisson'+suffix, 'w')
for i in range(users):
	fw.write(uid[i])
	fw.write('\t')
	fw.write(str(lbd[i]))
	fw.write('\n')
fw.close()

