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
gamma = 1 #log barrier function

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
lbd = posts * 1.0 / 7268 / 86400 / 10

fw = open(prefix+'lambda_Poisson'+suffix, 'w')
for i in range(users):
	fw.write(uid[i])
	fw.write('\t')
	fw.write(str(lbd[i]))
	fw.write('\n')
fw.close()

