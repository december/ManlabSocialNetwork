import os

info_dic = {} #from user id to its info list(posts, retweets, retweeteds)
friend = {} #from user id to a dictionary which from its followers' user id to the retweet times

fr = open('../../cascading_generation_model/all/all.detail', 'r')
data = fr.readlines()
fr.close()

path = '../../cascading_generation_model/all/networks/'
prefix = '../../cascading_generation_model/all/behavior_origin/'
namelist = os.listdir(path)
pointdic = {}
behaviordic = {}
tweetdic = {}
maxsize = {}
totalnum = {}
for name in namelist:
	fr = open(path+name, 'r')
	points = fr.readlines()
	fr.close()
	pointdic[name] = {}
	maxsize[name] = 0
	totalnum[name] = 0
	behaviordic[name] = list()
	for p in points:
		pointdic[name][p[:-1]] = 1
print 'Begin collecting...'

def CollectLog(info, k, m, td):
	for i in range(m):
		temp = info[i][:-1].split('\t')
		if pointdic[k].has_key(temp[1]) and (temp[4] == '-1' or pointdic[k].has_key(temp[4])) and (temp[3] == '-1' or tweetdic.has_key(temp[3])):
			td.append(info[i])
			tweetdic[temp[0]] = 1
	return td

n = len(data)
print n
#i = 1589440404
i = 0
cascade = 0
wrongdata = 0

while i < n:
	temp = data[i].split('\t')
	while len(temp[1]) > 10:
		i += 1
		wrongdata += 1
		temp = data[i].split('\t')
	number = int(temp[1]) + 1
	author = data[i+1].split('\t')[1]
	for key in pointdic:
		if pointdic[key].has_key(author):
			tempdata = list()
			tweetdic = {}
			tempdata.append(temp[0])
			tempdata = CollectLog(data[i+1:i+number], key, number - 1, tempdata)
			length = len(tempdata) - 1
			tempdata[0] = tempdata[0] + '\t' + str(length) + '\n'
			maxsize[key] = max(maxsize[key], length)
			totalnum[key] += length
			behaviordic[key].extend(tempdata)
	cascade += 1
	if cascade % 1000000 == 0:
		print str(cascade) + '(' + str(i) + ')'
	i += number
print 'Begin output...'
print wrongdata

for name in namelist:
	fw = open(prefix+name+'_'+str(maxsize[name])+'_'+str(totalnum[name])+'.detail', 'w')
	for line in behaviordic[name]:
		fw.write(line)
	fw.close()
