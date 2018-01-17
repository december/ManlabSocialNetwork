import sys
import lda
import scipy as sp
import numpy as np
import tensorflow as tf
import scipy.optimize
import numpy.random
import datetime

prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'
path = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/single_user_post/'
iddic = {}
idlist = list()
docs = 0
users = 0

realdata = list()
namelist = os.listdir(path)
for name in namelist:
	fr = open(path+name, 'r')
	realdata.extend(fr.readlines())
	fr.close()

n = len(realdata)
i = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	docs += 1
	for j in range(i+1, i+number):
		data = realdata[j].split('\t')
		if not iddic.has_key(data[1]):
			iddic[data[1]] = users
			idlist.append(data[1])
			users += 1
	i += number

info = np.zeros((docs, users))
print 'Finish reading. ' + str(info.shape)
i = 0
cdoc = 0
while i < n:
	temp = realdata[i].split('\t')
	number = int(temp[1]) + 1
	for j in range(i+1, i+number):
		data = realdata[j].split('\t')
		info[cdoc][iddic(data[1])] += 1
	i += number
	cdoc += 1
model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)
model.fit(info)
print 'Finish training.'
topic = model.doc_topic_
fw = open(prefix+'lda'+suffix, 'w')
n = len(topic)
for i in range(n):
	fw.write(idlist[i])
	for j in range(5):
		fw.write('\t')
		fw.write(str(topic[i][j]))
fw.close()		

print 'Finish writing.'