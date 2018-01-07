import os

info_dic = {} #from user id to its info list(posts, retweets, retweeteds)
friend = {} #from user id to a dictionary which from its followers' user id to the retweet times

fr = open('../../cascading_generation_model/all/all.detail', 'r')
data = fr.readlines()
fr.close()
n = len(data)
i = 0
while i < n:
	temp = data[i].split('\t')
	number = int(temp[1]) + 1
	root = temp[0]
	for j in range(i+1, number):
		temp = data[j][:-1].split('\t')
		uid = temp[1]
		pid = temp[4]
		if not info_dic.has_key(uid):
			info_dic[uid] = [0, 0, 0]
		if pid == '-1':
			info_dic[uid][0] += 1
		else:
			if not info_dic.has_key(pid):
				info_dic[pid] = [0, 0, 0]
			if not friend.has_key(pid):
				friend[pid] = {}
			if not friend[pid].has_key(uid):
				friend[pid][uid] = 1
			else:
				friend[pid][uid] += 1
			info_dic[uid][1] += 1
			info_dic[pid][2] += 1
	i += number

fw = open('../../cascading_generation_model/all/all_relation.detail', 'w')
for key in friend:
	fw.write(key)
	fw.write('\t')
	fw.write(str(len(friend[key])))
	fw.write('\n')
	for fd in friend[key]:
		fw.write('\t')
		fw.write(fd)
		fw.write('\t')
		fw.write(str(friend[key][fd]))
		fw.write('\n')
fw.close()

fw = open('../../cascading_generation_model/all/all_info.detail', 'w')	
for key in info_dic:
	fw.write(key)
	fw.write('\t')
	fw.write(str(info_dic[key][0]) + '|' + str(info_dic[key][1]) + '|' + str(info_dic[key][2]))
	fw.write('\n')
fw.close()
