# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import matplotlib.colors as colors
from pylab import *
import numpy as np
import matplotlib.colors as colors
np.seterr(divide='ignore', invalid='ignore')

# 设置最大的范围，保证画图的坐标轴一致性
REAL_MAX = {'size': 160, 'depth': 20, 'width': 140, 'Diameter': 23}

def showtime(s=''):
    print time.strftime('time: %Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' ' + str(s)


col_names = ['cascade_id', 'size', 'depth', 'width', 'Diameter']
input_folder = '/home/luyunfei/cascading_generation_model/simulation/info'  # 指定数据文件夹
# input_folder = '/home/hli/lyf_info'  # 指定数据文件夹
output_tmp_data = input_folder + '/col_data'
otuput_pic_folder = input_folder + '/pic'
files_name = os.listdir(input_folder)
files_name = [_ for _ in files_name if '.detail' in _]
files_name.sort()
data_len = len(col_names)
#assert len(files_name) == 3

if not os.path.exists(output_tmp_data):
    os.mkdir(output_tmp_data)


if not os.path.exists(otuput_pic_folder):
    os.mkdir(otuput_pic_folder)


# 这里先把数据提取成一个个csv
for f_name in files_name:  # 3
    f_path = '%s/%s' % (input_folder, f_name)
    for i in range(1, data_len):
        for j in range(1, data_len):
            if i >= j:  # C 42
                continue
            fout = open('%s/%s__%d__%d.csv' % (output_tmp_data, f_name.replace('.detail', ''), i, j), 'w')
            with open(f_path) as f:
                for line in f:
                    line = line.strip()
                    info = line.split()
                    x = int(info[i])
                    y = int(info[j])
                    print >> fout, '%d,%d' % (x, y)
            fout.close()

showtime('Finish pre data')

#  调试用打印矩阵
def print_matrix(data, path):
    with open('lyf_info/data/%s' % path, 'w') as f:
        for i in range(len(data)):
            for j in range(len(data[i])):
                print >> f, data[i][j],
            print >> f, ''


First = True
#  根据两个list画图
def plot_by_x_y(x_list, y_list, xy_f_name):
    _, x_name, y_name = xy_f_name.split('__')
    x_name = col_names[int(x_name)]
    y_name = col_names[int(y_name)]

    assert len(x_list) == len(y_list)

    x_min = min(x_list)
    x_max = max(max(x_list), REAL_MAX[x_name])
    y_min = min(y_list)
    y_max = max(max(y_list), REAL_MAX[y_name])

    print x_min, x_max, y_min, y_max
    data = [[0.0 for j in range(y_min,  y_max + 1)] for i in range(x_min, x_max + 1)]
    cnt = 0.0
    for i in range(len(x_list)):
        x = x_list[i] - x_min
        y = y_list[i] - y_min
        data[x][y] += 1
        cnt += 1
    # print_matrix(data, '%s_%s_%s.txt' % (_, x_name, y_name))
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] /= cnt
            data[i][j] = max(data[i][j], 1e-7)
    data = np.array(data)

    # pcolor(data)
    print data.max(), data.min()
    pcolor(data, norm=colors.LogNorm(vmin=data.min(), vmax=data.max()), cmap='magma')
    global First
    if First:
        colorbar()
        First = False

    # xticks(arange(x_min + 0.5, x_max + 1.5), range(x_min, x_max + 1, 10))
    # yticks(arange(y_min + 0.5, y_max + 1.5), range(y_min, y_max + 1, 10))
    # xticks(arange(x_min, x_max))
    # yticks(arange(y_min, y_max))
    # ylim(x_min, x_max)
    # xlim(y_min, y_max)
    ylim(x_min, REAL_MAX[x_name])
    xlim(y_min, REAL_MAX[y_name])
    xlabel(y_name)
    ylabel(x_name)
    # show()

    savefig('/home/luyunfei/cascading_generation_model/simulation/heat_pic/%s_%s_%s.eps' % (_, x_name, y_name), dpi=300)


xy_files_name = os.listdir(output_tmp_data)
xy_files_name = [_ for _ in xy_files_name if '.csv' in _ and '__' in _]
xy_files_name.sort()
#assert len(xy_files_name) == 30
print xy_files_name
for xy_f_name in xy_files_name:
    x_list = []
    y_list = []
    f_path = '%s/%s' % (output_tmp_data, xy_f_name)
    print 'do %s' % f_path
    with open(f_path) as f:

        for line in f:
            line = line.strip()
            x, y = line.split(',')
            try:
                x = int(x)
                y = int(y)
            except:
                continue
            x_list.append(x)
            y_list.append(y)
    print 'x_list len is %d, y_list len is %d' % (len(x_list), len(y_list))
    plot_by_x_y(x_list, y_list, xy_f_name.replace('.csv', ''))
    # break






