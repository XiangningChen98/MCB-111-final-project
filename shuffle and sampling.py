import numpy as np              # this is the standard python library for doing data data analysis of matrices
import os                       # this is a helper library for the operating system, which allows to concatenate a path, for example
import pandas as pd
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

average=[]
average_x=[]
average_y=[]
for i in range(1000):
    left_x_random=[]
    left_y_random=[]
    left_angle_random=[]
    frame_number=[]
    a=0
    for i in range(1000):

        frame_number.append(a)
        a=a+1
        left_x_random.append(random.uniform(0, 60))
        left_y_random.append(random.uniform(0, 60))
        left_angle_random.append(random.uniform(0, 360))
    right_x_random=[]
    right_y_random=[]
    right_angle_random=[]
    for i in range(1000):
        right_x_random.append(random.uniform(0, 60))
        right_y_random.append(random.uniform(0, 60))
        right_angle_random.append(random.uniform(0, 360))
    x_distance_random=[]
    y_distance_random=[]
    distance_random=[]
    left_x2=60
    for i in range(len(right_x_random)):
        dis_x_random=right_x_random[i]+left_x2-left_x_random[i]
        dis_y_random=right_y_random[i]-left_y_random[i]
        dis_random = ((dis_x_random)**2+(dis_y_random)**2)**0.5
        x_distance_random.append(dis_x_random)
        y_distance_random.append(dis_y_random)
        distance_random.append(dis_random)
    fish_df_random=pd.DataFrame({'frame':frame_number,'left_x':left_x_random,'left_y':left_y_random,'left_angle':left_angle_random,'right_x':right_x_random,'right_y':right_y_random,'right_angle':right_angle_random,'x_distance':x_distance_random,'y_distance':y_distance_random,'distance':distance_random})
    m, s = stats.norm.fit(fish_df_random['distance'])
    m_x, s_x = stats.norm.fit(fish_df_random['x_distance'])
    m_y, s_y = stats.norm.fit(fish_df_random['y_distance'])
    average.append(m)
    average_x.append(m_x)
    average_y.append(m_y)
ave=pd.Series(average)
ave_x=pd.Series(average_x)
ave_y=pd.Series(average_y)
ave.hist(bins=30,density=True,color='orange',alpha=0.3)
range_plot = np.arange(60,68, 0.001)
m, s = stats.norm.fit(ave)
plt.plot(range_plot, norm.pdf(range_plot,m,s),label='fitted curve')
print('random distance ave: m is',m,'s is',s)
print(np.percentile(ave, 2.5), np.percentile(ave, 97.5))
plt.axvline(np.percentile(ave, 2.5), color='k', linestyle='dashed', linewidth=1,label='95% interval')
plt.axvline(np.percentile(ave, 97.5), color='k', linestyle='dashed', linewidth=1)
title='mean distribution of random distance '+str(np.percentile(ave, 2.5))+' '+str(np.percentile(ave, 97.5))
plt.title('distribution of mean distance of randomly sampling')
plt.axvline(61.72608798069477, linestyle='dashed', linewidth=1,color='red',label='exp')

plt.legend()
plt.xlabel('mean of distance')
plt.ylabel('probability')
plt.show()



#shuffle
root_path = "/Users/xiangning/Desktop/MCB 111 project/20190329_2Ffish_2rigs_20pairs/60_scale/"
fish_names=['20190302_04','20190302_05','20190302_06','20190302_07','20190302_08','20190302_09','20190302_12','20190329_01','20190329_02','20190329_03','20190329_04','20190329_05','20190329_06','20190329_07','20190329_08','20190329_09','20190329_11','20190329_12','20190329_13','20190329_14']
# fish_names=['20190329_01','20190329_02','20190329_03','20190329_04','20190329_05','20190329_06','20190329_07','20190329_08','20190329_09','20190329_11','20190329_12','20190329_13','20190329_14']


def load_data(root_path,fish_names,half):
    all_data=[]
    for fish_name in fish_names:
        path = os.path.join(root_path, fish_name+half+'_180422_extracted_x_y_ang_2min.npy')
#         path = os.path.join(root_path, fish_name+half+'_180422_extracted_x_y_ang.npy')


        data=np.load(path)
        all_data.append(data)
    all_data=np.asarray(all_data)
    return all_data
left_dataset=load_data(root_path=root_path,fish_names=fish_names,half='left')
right_dataset=load_data(root_path=root_path,fish_names=fish_names,half='right')
#the combine of all the data
all_dataset=[]
left_x2=60
dis_all_list=[]
for i in range(len(left_dataset)):
    fish=[]
    dis_all=0
    for j in range(len(left_dataset[i])):
        combine=np.concatenate((left_dataset[i][j], right_dataset[i][j][1:]), axis=None)
        dis_x=right_dataset[i][j][1]+(left_x2-left_dataset[i][j][1])
        combine = np.append(combine, dis_x)
        dis_y=right_dataset[i][j][2]-left_dataset[i][j][2]
        combine = np.append(combine, dis_y)
        dis = ((dis_x)**2+(dis_y)**2)**0.5
        dis_all=dis_all+dis
        combine = np.append(combine, dis)
        fish.append(combine)
    all_dataset.append(fish)
    dis_all_list.append(dis_all)


all_dataset=np.asarray(all_dataset)
# calculate the main of the real data


print(len(all_dataset))
mean_mutual_real = []
mean_x_real = []
mean_y_real = []
for i in all_dataset:
    #     print([item[7] for item in i])
    mean_mutual_dis = np.mean([item[9] for item in i])

    mean_mutual_real.append(mean_mutual_dis)
    mean_y_dis = np.mean([item[8] for item in i])
    mean_y_real.append(mean_y_dis)
    mean_x_dis = np.mean([item[7] for item in i])
    mean_x_real.append(mean_x_dis)

# print(mean_mutual_real,mean_x_real,mean_y_real)
x_mean_all_real = np.mean(mean_x_real)
y_mean_all_real = np.mean(mean_y_real)
mutual_mean_all_real = np.mean(mean_mutual_real)
print(x_mean_all_real, y_mean_all_real, mutual_mean_all_real)
#shift the data in every pair
def shift_data(data_left,data_right,time):
    left_x=[]
    left_y=[]
    left_angle=[]
    frame_number=[]
    a=0
    for i in data_left:
        frame_number.append(a)
        a=a+1
        left_x.append(i[1])
        left_y.append(i[2])
        if i[3]<270:
            left_angle.append(90-i[3])
        else:
            left_angle.append(450-i[3])
    right_x=[]
    right_y=[]
    right_angle=[]
    for i in data_right:
        right_x.append(i[1])
        right_y.append(i[2])
        if i[3]<270:
            right_angle.append(90-i[3])
        else:
            right_angle.append(450-i[3])

    left_fish_df=pd.DataFrame({'frame':frame_number,'left_x':left_x,'left_y':left_y,'left_angle':left_angle})
    right_fish_df=pd.DataFrame({'frame':frame_number,'right_x':right_x,'right_y':right_y,'right_angle':right_angle})
    circled_right_df=right_fish_df.copy()
    circled_right_df['right_x'] = np.roll(circled_right_df['right_x'],time)
    circled_right_df['right_y'] = np.roll(circled_right_df['right_y'],time)
    circled_right_df['right_angle'] = np.roll(circled_right_df['right_angle'],time)
    circled_right_df=circled_right_df.drop(columns=['frame'])
    left_x2=60
    distance=[]
    x_distance=[]
    y_distance=[]
    circled_shuffle_df=pd.concat([left_fish_df,circled_right_df], axis=1)
    for i in range(len(circled_shuffle_df)):
        dis_x=circled_shuffle_df.loc[i]['right_x']+left_x2-circled_shuffle_df.loc[i]['left_x']
        dis_y=circled_shuffle_df.loc[i]['right_y']-circled_shuffle_df.loc[i]['left_y']
        dis = ((dis_x)**2+(dis_y)**2)**0.5
        x_distance.append(dis_x)
        y_distance.append(dis_y)
        distance.append(dis)
    circled_shuffle_df['x_distance']=pd.Series(x_distance)
    circled_shuffle_df['y_distance']=pd.Series(y_distance)
    circled_shuffle_df['distance']=pd.Series(distance)
    mean_x_shift=np.mean(circled_shuffle_df['x_distance'])
    mean_y_shift=np.mean(circled_shuffle_df['y_distance'])
    mean_mutual_shift=np.mean(circled_shuffle_df['distance'])
    return mean_x_shift,mean_y_shift,mean_mutual_shift
def shift_for_all_once(left_dataset,right_dataset,time):
    mean_x_shift=[]
    mean_y_shift=[]
    mean_mutual_shift=[]
    for i in range(len(left_dataset)):
        mean_x_shift_1,mean_y_shift_1,mean_mutual_shift_1=shift_data(left_dataset[i],right_dataset[i],time)
        mean_x_shift.append(mean_x_shift_1)
        mean_y_shift.append(mean_y_shift_1)
        mean_mutual_shift.append(mean_mutual_shift_1)

    mean_for_mean_x_shift=np.mean(mean_x_shift)
    mean_for_mean_y_shift=np.mean(mean_y_shift)
    mean_for_mean_mutual_shift=np.mean(mean_mutual_shift)
    return mean_for_mean_x_shift,mean_for_mean_y_shift,mean_for_mean_mutual_shift

x_mean_distribution=[]
y_mean_distribution=[]
mutual_mean_distribution=[]

for i in range(0,500):
    print(i)
    time=random.randint(1,6400)
    mean_for_mean_x_shift,mean_for_mean_y_shift,mean_for_mean_mutual_shift=shift_for_all_once(left_dataset,right_dataset,time)
    x_mean_distribution.append(mean_for_mean_x_shift)
    y_mean_distribution.append(mean_for_mean_y_shift)
    mutual_mean_distribution.append(mean_for_mean_mutual_shift)



x_mean_distribution=np.asarray(x_mean_distribution)
y_mean_distribution=np.asarray(y_mean_distribution)
mutual_mean_distribution=np.asarray(mutual_mean_distribution)

range_plot = np.arange(56, 65, 0.001)
mutual_mean_distribution_s=pd.Series(mutual_mean_distribution)
mutual_mean_distribution_s.hist(bins=50,color='orange',alpha=0.5)
plt.axvline(mutual_mean_all_real, color='red', linestyle='dashed', linewidth=1,label='exp mean')
plt.axvline(np.percentile(mutual_mean_distribution, 2.5), color='k', linestyle='dashed', linewidth=1,label='95% interval')
plt.axvline(np.percentile(mutual_mean_distribution, 97.5), color='k', linestyle='dashed', linewidth=1)
plt.title('mutual distance shuffle 1000 times data ')
plt.ylabel('counts')
plt.xlabel('mean of mean(distance)')
plt.legend()

plt.show()
