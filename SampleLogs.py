import os

info_dic = {} #from user id to its info list(posts, retweets, retweeteds)
friend = {} #from user id to a dictionary which from its followers' user id to the retweet times

fr = open('../../cascading_generation_model/all/all.detail', 'r')
data = fr.readlines()
fr.close()

path = '../../cascading_generation_model/all/networks/'
prefix = '../../cascading_generation_model/all/behavior/'
namelist = os.listdir(path)
pointdic = {}
behaviordic = {}
for name in namelist
	fr = open(path+name, 'r')
	points = fr.readlines()
	fr.close()
	pointdic[name] = {}
	behaviordic[name] = list()
	for p in points:
		pointdic[name][p[:-1]] = 1
print 'Begin collecting...'

def CollectLog(info, k, m):
	for i in range(m):
		temp = info[i][:-1].split('\t')
		if pointdic[k].has_key(info[1]) and (info[4] == '-1' or pointdic[k].has_key(info[4])):
			behaviordic[k].append(info[i])

n = len(data)
print n
i = 0
cascade = 0
while i < n:
	temp = data[i].split('\t')
	number = int(temp[1]) + 1
	author = data[i+1].split('\t')[1]
	for key in pointdic:
		if pointdic[key].has_key(author):
			behaviordic[key].append(data[i])
			CollectLog(data[i+1:i+number], key, number)
	cascade += 1
	if cascade % 1000000 == 0:
		print str(cascade) + '(' + str(i) + ')'
	i += number
print 'Begin output...'

for name in namelist:
	fw = open(prefix+name, 'w')
	for line in behaviordic[name]:
		fw.write(line)
	fw.close()
