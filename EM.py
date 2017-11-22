import scipy as sp
import numpy as np

users = 7268
uid = list();
iddic = {};
friend = {};
lbd = np.zeros(users) #parameter lambda which have calculated before
omega = np.zeros(users) #parameter omega
theta1 = np.zeros(users) #one of spherical coordinates of phi distribution
theta2 = np.zeros(users) #one of spherical coordinates of phi distribution
theta3 = np.zeros(users) #one of spherical coordinates of phi distribution
theta4 = np.zeros(users) #one of spherical coordinates of phi distribution
pi = sp.sparse.coo_matrix((users, users)) #parameter pi (based on edges), row is sender while col is receiver
x = sp.sparse.coo_matrix((users, users)) #parameter x (based on edges), row is sender while col is receiver

def Phi1():
	return np.cos(theta1) * np.cos(theta1)

def Phi2():
	return np.sin(theta1) * np.sin(theta1) * np.cos(theta2) * np.cos(theta2)

def Phi3():
	return np.sin(theta1) * np.sin(theta1) * np.sin(theta2) * np.sin(theta2) * np.cos(theta3) * np.cos(theta3)

def Phi4():
	return np.sin(theta1) * np.sin(theta1) * np.sin(theta2) * np.sin(theta2) * np.sin(theta3) * np.sin(theta3) * np.cos(theta4) * np.cos(theta4)

def Phi5():
	return np.sin(theta1) * np.sin(theta1) * np.sin(theta2) * np.sin(theta2) * np.sin(theta3) * np.sin(theta3) * np.sin(theta4) * np.sin(theta4)

def Q():


def EStep():


def MStep():


def SingleObj(data, u):
	




#Get lambda value
prefix = '../../cascading_generation_model/722911_twolevel_neighbor_cascades/'
suffix = '.detail'
fr = open(prefix+'lambda_Poisson'+suffix, 'r')
lbdlist = fr.readlines()
for i in range(users):
	temp = lbdlist[i].split('\t')
	uid.append(temp[0])
	iddic[temp[0]] = i
	lbd[i] = float(temp[1])
fr.close()

#Get post times
posts = {}
fr = open(prefix+'posttimes'+suffix, 'r')
post = fr.readlines()
for i in range(users):
	temp = post[i].split('\t')
	posts[temp[0]] = int(temp[1])
fr.close()

#Give initial value and construct relation
omega += sum(lbd) * 100 / users
theta1 += np.arccos(np.sqrt(0.2))
theta2 += np.arccos(np.sqrt(0.25))
theta3 += np.arccos(np.sqrt(1.0 / 3))
theta4 += np.arccos(np.sqrt(0.5))

fr = open(prefix+'relations'+suffix, 'r')
relation = fr.readlines()
n = len(relation)
i = 0
while i < n:
	temp = relation[i].split('\t')
	number = int(temp[1]) + 1
	friend[temp[0]] = list()
	for j in range(i+1, i+number):
		fd = relation[j].split('\t')
		pi[iddic[temp[0]], iddic[fd[1]]] = int(fd[2]) * 1.0 / posts[temp[0]]
		x[iddic[temp[0]], iddic[fd[1]]] = 1.0
		friend[temp[0]].append(fd[1])
	i += number

#Read personal cascade file
n = len(uid)
for i in range(n):
	fr = open(prefix+'single_user_post/'+str(i)+'_'+uid[i]+suffix, 'w')
	single = fr.readlines()




