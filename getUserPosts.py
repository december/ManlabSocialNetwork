prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'
total = 100
logdic = {}
numberdic = {}
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
	if numberdic.has_key(userid):
		numberdic[userid] += 1
	else:
		numberdic[userid] = 1
	if not logdic.has_key(userid):
		logdic[userid] = list()
	for j in range(i, i + size):
		logdic[userid].append(cascades[j])
	i += size;

orderlist = sorted(numberdic.items(), key=lambda item:item[1], reverse=True)
i = 0;
while i < total:
	print orderlist[i][1]
	fw = open(prefix+'single_user_post/'+str(i)+'_'+str(orderlist[i][0])+suffix, 'w')
	for line in logdic[orderlist[i][0]]:
		fw.write(line)
	fw.close()
	i += 1
