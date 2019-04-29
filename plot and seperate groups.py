
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import pylab as p
exp_data_left=np.load("/Users/xiangning/Desktop/MCB 111 project/20190329_2Ffish_2rigs_20pairs/60_scale/left_201903_extracted_x_y_ang_2_fish_20fish.npy")
exp_data_right=np.load("/Users/xiangning/Desktop/MCB 111 project/20190329_2Ffish_2rigs_20pairs/60_scale/right_201903_extracted_x_y_ang_2_fish_20fish.npy")


def distance(root, left_fish_names, right_fish_names, date):
    mean_set = []
    for i in range(len(left_fish_names)):

        left_path = os.path.join(root, left_fish_names[i])
        right_path = os.path.join(root, right_fish_names[i])
        data_left = np.load(left_path + 'left_' + date + '_extracted_x_y_ang_2min.npy')
        data_right = np.load(right_path + 'right_' + date + '_extracted_x_y_ang_2min.npy')
        left_x = []
        left_y = []
        left_angle = []
        frame_number = []
        a = 0
        for i in data_left:
            frame_number.append(a)
            a = a + 1
            left_x.append(i[1])
            left_y.append(i[2])
            if i[3] < 270:
                left_angle.append(90 - i[3])
            else:
                left_angle.append(450 - i[3])

        right_x = []
        right_y = []
        right_angle = []
        for i in data_right:
            right_x.append(i[1])
            right_y.append(i[2])
            if i[3] < 270:
                right_angle.append(90 - i[3])
            else:
                right_angle.append(450 - i[3])
        x_distance = []
        y_distance = []
        distance = []
        left_x2 = 60
        for i in range(len(data_left)):
            dis_x = data_right[i][1] + (left_x2 - data_left[i][1])
            dis_y = data_right[i][2] - data_left[i][2]
            dis = ((data_right[i][2] - data_left[i][2]) ** 2 + (
                        data_right[i][1] + (left_x2 - data_left[i][1])) ** 2) ** 0.5
            x_distance.append(dis_x)
            y_distance.append(dis_y)
            distance.append(dis)
        fish_df = pd.DataFrame(
            {'frame': frame_number, 'left_x': left_x, 'left_y': left_y, 'left_angle': left_angle, 'right_x': right_x,
             'right_y': right_y, 'right_angle': right_angle, 'x_distance': x_distance, 'y_distance': y_distance,
             'distance': distance})
        mean_set.append(np.mean(fish_df['distance']))

    return mean_set

exp_root='/Users/xiangning/Desktop/MCB 111 project/20190329_2Ffish_2rigs_20pairs/60_scale/'
exp_names=['20190302_04','20190302_05','20190302_06','20190302_07','20190302_08','20190302_09','20190302_12','20190329_01','20190329_02','20190329_03','20190329_04','20190329_05','20190329_06','20190329_07','20190329_08','20190329_09','20190329_11','20190329_12','20190329_13','20190329_14']
# exp_names=['20190329_01','20190329_02','20190329_03','20190329_04','20190329_05','20190329_06','20190329_07','20190329_08','20190329_09','20190329_11','20190329_12','20190329_13','20190329_14']


exp_list=distance(exp_root,exp_names,exp_names,'180422')

x1=[]
for i in range(len(exp_list)):
    x1.append(1)


print(exp_list)
plt.figure(figsize=(4,6))


plt.scatter(x=x1,y=exp_list,label='20fish',color='red',alpha=0.5)
# plt.scatter(x=x2,y=ctrl_list,label='ctrl',color='skyblue',alpha=0.5)
# plt.xticks(np.arange(4), (' ','exp','ctrl',' '))
plt.xticks(np.arange(3), (' ','fish',' '))


# plt.title('mean of mutual distance')
plt.ylabel('mean of mutual distance(mm)')
plt.legend()
plt.show()

fish_names = ['20190329_06', '20190329_07', '20190329_08', '20190329_11']
root_path = r"/Users/xiangning/Desktop/MCB 111 project/20190329_2Ffish_2rigs_20pairs/60_scale/"
dataset_left_all = []
dataset_right_all = []
for name in fish_names:
    left_path = os.path.join(root_path, name + 'left_180422_extracted_x_y_ang_2min.npy')
    right_path = os.path.join(root_path, name + 'right_180422_extracted_x_y_ang_2min.npy')
    left_data = np.load(left_path)
    dataset_left_all.extend(left_data)
    right_data = np.load(right_path)
    dataset_right_all.extend(right_data)

dataset_left_array = np.asarray(dataset_left_all)
dataset_right_array = np.asarray(dataset_right_all)

left_x = []
left_y = []
left_angle = []
frame_number = []
a = 0
for i in dataset_left_array:

    frame_number.append(a)
    a = a + 1
    left_x.append(i[1])
    left_y.append(i[2])
    if i[3] < 270:
        left_angle.append(90 - i[3])
    else:
        left_angle.append(450 - i[3])
right_x = []
right_y = []
right_angle = []
for i in dataset_right_array:
    right_x.append(i[1])
    right_y.append(i[2])
    if i[3] < 270:
        right_angle.append(90 - i[3])
    else:
        right_angle.append(450 - i[3])
x_distance = []

x_distance = []
y_distance = []
distance = []
left_x2 = 60
for i in range(len(dataset_right_array)):
    dis_x = dataset_right_array[i][1] + (left_x2 - dataset_left_array[i][1])
    dis_y = dataset_right_array[i][2] - dataset_left_array[i][2]
    dis = ((dataset_right_array[i][2] - dataset_left_array[i][2]) ** 2 + (
                dataset_right_array[i][1] + (left_x2 - dataset_left_array[i][1])) ** 2) ** 0.5
    x_distance.append(dis_x)
    y_distance.append(dis_y)
    distance.append(dis)
fish_df_population1_4fish = pd.DataFrame(
    {'frame': frame_number, 'left_x': left_x, 'left_y': left_y, 'left_angle': left_angle, 'right_x': right_x,
     'right_y': right_y, 'right_angle': right_angle, 'x_distance': x_distance, 'y_distance': y_distance,
     'distance': distance})
# fish_df_population1_4fish
fish_names = ['20190302_04', '20190302_05', '20190302_06', '20190302_07', '20190302_08', '20190302_09', '20190302_12',
              '20190329_01', '20190329_02', '20190329_03', '20190329_04', '20190329_05', '20190329_09', '20190329_12',
              '20190329_13', '20190329_14']
root_path = r"/Users/xiangning/Desktop/MCB 111 project/20190329_2Ffish_2rigs_20pairs/60_scale/"

dataset_left_all = []
dataset_right_all = []
for name in fish_names:
    left_path = os.path.join(root_path, name + 'left_180422_extracted_x_y_ang_2min.npy')
    right_path = os.path.join(root_path, name + 'right_180422_extracted_x_y_ang_2min.npy')
    left_data = np.load(left_path)
    dataset_left_all.extend(left_data)
    right_data = np.load(right_path)
    dataset_right_all.extend(right_data)

dataset_left_array = np.asarray(dataset_left_all)
dataset_right_array = np.asarray(dataset_right_all)

left_x = []
left_y = []
left_angle = []
frame_number = []
a = 0
for i in dataset_left_array:

    frame_number.append(a)
    a = a + 1
    left_x.append(i[1])
    left_y.append(i[2])
    if i[3] < 270:
        left_angle.append(90 - i[3])
    else:
        left_angle.append(450 - i[3])
right_x = []
right_y = []
right_angle = []
for i in dataset_right_array:
    right_x.append(i[1])
    right_y.append(i[2])
    if i[3] < 270:
        right_angle.append(90 - i[3])
    else:
        right_angle.append(450 - i[3])
x_distance = []
y_distance = []
distance = []
left_x2 = 60
for i in range(len(dataset_right_array)):
    dis_x = dataset_right_array[i][1] + (left_x2 - dataset_left_array[i][1])
    dis_y = dataset_right_array[i][2] - dataset_left_array[i][2]
    dis = ((dataset_right_array[i][2] - dataset_left_array[i][2]) ** 2 + (
                dataset_right_array[i][1] + (left_x2 - dataset_left_array[i][1])) ** 2) ** 0.5
    x_distance.append(dis_x)
    y_distance.append(dis_y)
    distance.append(dis)
fish_df_population2_16fish = pd.DataFrame(
    {'frame': frame_number, 'left_x': left_x, 'left_y': left_y, 'left_angle': left_angle, 'right_x': right_x,
     'right_y': right_y, 'right_angle': right_angle, 'x_distance': x_distance, 'y_distance': y_distance,
     'distance': distance})
# fish_df_population1_4fish
left_x = []
left_y = []
left_angle = []
frame_number = []
a = 0
for i in exp_data_left:

    frame_number.append(a)
    a = a + 1
    left_x.append(i[1])
    left_y.append(i[2])
    if i[3] < 270:
        left_angle.append(90 - i[3])
    else:
        left_angle.append(450 - i[3])
right_x = []
right_y = []
right_angle = []
for i in exp_data_right:
    right_x.append(i[1])
    right_y.append(i[2])
    if i[3] < 270:
        right_angle.append(90 - i[3])
    else:
        right_angle.append(450 - i[3])
x_distance = []

y_distance = []
distance = []
left_x2 = 60
for i in range(len(exp_data_left)):
    dis_x = exp_data_right[i][1] + (left_x2 - exp_data_left[i][1])
    dis_y = exp_data_right[i][2] - exp_data_left[i][2]
    dis = ((exp_data_right[i][2] - exp_data_left[i][2]) ** 2 + (
                exp_data_right[i][1] + (left_x2 - exp_data_left[i][1])) ** 2) ** 0.5
    x_distance.append(dis_x)
    y_distance.append(dis_y)
    distance.append(dis)
fish_df_exp = pd.DataFrame(
    {'frame': frame_number, 'left_x': left_x, 'left_y': left_y, 'left_angle': left_angle, 'right_x': right_x,
     'right_y': right_y, 'right_angle': right_angle, 'x_distance': x_distance, 'y_distance': y_distance,
     'distance': distance})
# fish_df_exp
#plot the two set
bins_number=50
plt.hist(bins=bins_number,x=fish_df_exp['distance'],density=True,alpha=0.1,color='red',label='20 fish')
plt.axvline(np.mean(fish_df_exp['distance']), linestyle='dashed', linewidth=1,color='red')
plt.title('mutual distance distribution')
plt.xlabel('distance(mm)')
plt.ylabel('probability')
from scipy.signal import savgol_filter


y,binEdges=np.histogram(fish_df_exp['distance'],density=True,bins=50)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
yhat = savgol_filter(y, 9, 3)
p.plot(bincenters,yhat,'-',color='red')

plt.legend()
p.show()

bins_number=50

plt.hist(bins=30,x=fish_df_population1_4fish['distance'],alpha=0.5,color='red',label='4 fish')
plt.hist(bins=30,x=fish_df_population2_16fish['distance'],alpha=0.1,color='green',label='16fish')
# plt.legend()
plt.title('mutual distance of 4 fish and 16 fish')
plt.xlabel('distance(mm)')
plt.ylabel('probability')

y,binEdges=np.histogram(fish_df_population2_16fish['distance'],bins=30)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
from scipy.signal import savgol_filter
yhat = savgol_filter(y, 3, 1)
p.plot(bincenters,yhat,'-',color='green',label='16 fish')
y,binEdges=np.histogram(fish_df_population1_4fish['distance'],30)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
yhat = savgol_filter(y, 3, 1)
p.plot(bincenters,yhat,'-',color='red',label='4 fish')
plt.xlim(0,130)
plt.legend()
p.show()





