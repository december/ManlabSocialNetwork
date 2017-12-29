import os

infodic = {} #form tweet id to the info of it(user_id, tweet_time, reply_tweet_id, reply_user_id, root_tweet_id, root_user_id)
cascade = {} #from root id to the info of tweets in cascade(tweet_id, user_id, tweet_time, reply_tweet_id, reply_user_id)
tweetdic = {} #from real tweet id to our tweet id
userdic = {} #from real user id to our user id
prefix = '../../dataset/tencent_weibo_2011_11/'
namelist = os.listdir(prefix)
tweet_cnt = 0
user_cnt = 0
for name in namelist:
	if name.startswith('2011_'):
		fr = open(prefix+name, 'r')
		data = fr.readlines()
		fr.close()
		n = len(data)
		i = 0
		while i < n:
			if i + 12 >= n:
				break
			tid = data[i+1][2:-1]
			tm = data[i+2][2:-1]
			tu = data[i+3][2:-1]
			rt = data[i+9][2:-1]
			ru = data[i+10][2:-1]
			pt = data[i+11][2:-1]
			pu = data[i+12][2:-1]
			print str(tid) + ' ' + str(tm) + ' ' + str(tu) + ' ' + str(rt) + ' ' + str(ru) + ' ' + str(pt) + ' ' + str(pu)
			if tweetdic.has_key(tid):
				tid = tweetdic[tid]
			else:
				tweetdic[tid] = tweet_cnt
				tid = tweet_cnt
				tweet_cnt += 1
			if userdic.has_key(tu):
				tu = userdic[tu]
			else:
				userdic[tu] = user_cnt
				tu = user_cnt
				user_cnt += 1

			if int(pt) != 0 and int(pu) != 0:
				if tweetdic.has_key(rt):
					rt = tweetdic[rt]
				else:
					tweetdic[rt] = tweet_cnt
					rt = tweet_cnt
					tweet_cnt += 1
				if tweetdic.has_key(pt):
					pt = tweetdic[pt]
				else:
					tweetdic[pt] = tweet_cnt
					pt = tweet_cnt
					tweet_cnt += 1
				if userdic.has_key(ru):
					ru = userdic[ru]
				else:
					userdic[ru] = user_cnt
					ru = user_cnt
					user_cnt += 1
				if userdic.has_key(pu):
					pu = userdic[pu]
				else:
					userdic[pu] = user_cnt
					pu = user_cnt
					user_cnt += 1
			else:
				pt = -1
				pu = -1
				rt = tid
				ru = tu
			infodic[tid] = list()
			infodic[tid].append(tu)
			infodic[tid].append(int(tm))
			infodic[tid].append(pt)
			infodic[tid].append(pu)			
			infodic[tid].append(rt)
			infodic[tid].append(ru)
			while i < n and data[i][0] != '!':
				i += 1

infolist = sorted(infodic.items(), key=lambda d:d[1][1])
for info in infolist:
	if info[1][2] == -1:
		cascade[info[0]] = list()
		temp = list()
		temp.append(info[0])
		temp.extend(info[1][:4])
		cascade[info[0]].append(temp)
	else:
		if not cascade.has_key(info[1][4]):
			continue
		temp = list()
		temp.append(info[0])
		temp.extend(info[1][:4])
		cascade[info[1][4]].append(temp)
print len(cascade)
caslist = sorted(cascade.items(), key=lambda d:d[1][0][1])
fw = open('../../cascading_generation_model/all/all.detail', 'w')
for cas in caslist:
	fw.write(str(cas[0]))
	fw.write('\t')
	fw.write(str(len(cas[1])))
	fw.write('\n')
	for tweet in cas[1]:
		fw.write(str(tweet[0]))
		for i in range(1, 5):
			fw.write('\t')
			fw.write(str(tweet[i]))
		fw.write('\n')
fw.close()
