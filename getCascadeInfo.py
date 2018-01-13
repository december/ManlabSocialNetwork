cascade = {}
select = list()

fr = open('../../cascading_generation_model/all/all.detail', 'r')
data = fr.readlines()
fr.close()
n = len(data)
i = 0
while i < n:
	temp = data[i].split('\t')
	number = int(temp[1])
	user = data[i+1].split('\t')[1]
	if not cascade.has_key(user):
		cascade[user] = {}
	if not cascade[user].has_key(number):
		cascade[user][number] = 1
	else:
		cascade[user][number] += 1
	i += number + 1

fw = open('../../ascading_generation_model/all/all_cascade.detail', 'w')
cnt = 0
for key in cascade:
	number = sum(cascade[key].values())
	size = max(cascade[key].keys())
	fw.write(key+'\t'+str(number)+'\t'+str(size)+'\n')
	if size > 100 and number > 300:
		cnt += 1
		select.append(key)
print cnt
fw.close()

fw = open('../../ascading_generation_model/all/select.detail', 'w')
for point in select:
	fw.write(point)
	fw.write('\n')
fw.close()
