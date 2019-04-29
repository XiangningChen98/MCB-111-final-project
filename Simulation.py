#!/usr/bin/env python
# coding: utf-8

# # randomly swimming fish


import numpy as np
import matplotlib.pyplot as plt
import random
unit_distance=2
x_pos_left_lim=[2,58]
x_pos_right_lim=[62,118]
y_pos_left_lim=[2,58]
y_pos_right_lim=[2,58]


fig = plt.figure(figsize=(8, 3))


theta_distribution=np.concatenate((np.random.normal(0, 5, 1000),np.random.normal(20, 15, 300),np.random.normal(-20, 15, 300)))
theta_meet_wall=np.concatenate((np.random.normal(0, 3, 500),np.random.normal(40, 15, 300),np.random.normal(-40, 15, 300)))
theta_have_fish_distance15=np.concatenate((np.random.normal(0, 5, 1000),np.random.normal(30, 10, int(1/15*10000))))
theta_have_fish_distance30=np.concatenate((np.random.normal(0, 5, 1000),np.random.normal(-50, 10, int(1/30*10000))))
plt.hist(theta_meet_wall, 200, density=True,alpha=0.3)
plt.title('Theta meet wall')
plt.ylim(0,0.06)
plt.xlim(-180,180)
plt.show()
fig = plt.figure(figsize=(8, 3))


plt.hist(theta_distribution,200,density=True,alpha=0.3)
plt.title('Theta freely moving')
plt.ylim(0,0.06)


plt.xlim(-180,180)
plt.show()
fig = plt.figure(figsize=(8, 3))




plt.hist(theta_have_fish_distance15,200,density=True,alpha=0.3)
plt.title('theta_have_fish_distance 15')
plt.ylim(0,0.06)


plt.xlim(-180,180)
plt.show()
fig = plt.figure(figsize=(8, 3))


plt.hist(theta_have_fish_distance30,200,density=True,alpha=0.3)
plt.title('theta_have_fish_distance 30')
plt.ylim(0,0.06)


plt.xlim(-180,180)
plt.show()





def meetwall(theta_meet_wall,unit_distance,x_pos,y_pos,theta):
    angle_change=np.random.choice(theta_meet_wall,[1])[0]
    theta=(theta+angle_change)%360
    x_pos=x_pos
#     print(x_pos)
    y_pos=y_pos
    dx=unit_distance*np.cos(theta*np.pi/180) 
    dy=unit_distance*np.sin(theta*np.pi/180)
#     print(type(x_pos),y_pos,dx,dy,theta,angle_change)
    return x_pos,y_pos,dx,dy,theta,angle_change
def normalwalk(theta_distribution,unit_distance,x_pos,y_pos,theta):
    angle_change=np.random.choice(theta_distribution,[1])[0]
    theta=(theta+angle_change)%360
    dx=unit_distance*np.cos(theta*np.pi/180) 
    dy=unit_distance*np.sin(theta*np.pi/180)

    x_pos=dx+x_pos
    y_pos=dy+y_pos
    return x_pos,y_pos,dx,dy,theta,angle_change

def havefish(relative_angle,theta_angle,value,center_mean,center_std,center_number,side_approach_mean,side_appraoch_std,side_avoid_mean,side_avoid_std):
        if value<1/20:
            if relative_angle < theta_angle:
                theta_have_fish=np.concatenate((np.random.normal(center_mean, center_std, center_number),np.random.normal(-side_approach_mean,side_appraoch_std, int(value*30000))))
            if relative_angle >= theta_angle:
                theta_have_fish=np.concatenate((np.random.normal(center_mean, center_std, center_number),np.random.normal(side_approach_mean, side_appraoch_std, int(value*30000))))
        else:
            if relative_angle < theta_angle:
                theta_have_fish=np.concatenate((np.random.normal(center_mean, center_std, center_number),np.random.normal(side_avoid_mean, side_avoid_std, int(value*30000))))
            if relative_angle >= theta_angle:
                theta_have_fish=np.concatenate((np.random.normal(center_mean, center_std, center_number),np.random.normal(-side_avoid_mean, side_avoid_std, int(value*30000))))
        return theta_have_fish



def simulate_once(theta_distribution,theta_meet_wall,x_pos_left_lim,x_pos_right_lim,y_pos_left_lim,unit_distance):
    position_left=np.array((random.randint(x_pos_left_lim[0],x_pos_left_lim[1]),random.randint(y_pos_left_lim[0],y_pos_left_lim[1]),random.randint(0,360)))
    position_right=np.array((random.randint(x_pos_right_lim[0],x_pos_right_lim[1]),random.randint(y_pos_right_lim[0],y_pos_right_lim[1]),random.randint(0,360)))
    fish_state_left=[]
    fish_state_right=[]
#     change_angle=[]
    dx_left=1
    dx_right=1
    dy_left=0
    dy_right=0
    for i in range(400):       
        x_pos_left=position_left[0]   
        y_pos_left=position_left[1]
        theta_left=position_left[2]
        x_pos_right=position_right[0]
        y_pos_right=position_right[1]
        theta_right=position_right[2]
#         print('2',x_pos_left)
        relative_angle=np.arctan((y_pos_right-y_pos_left)/(x_pos_right-x_pos_left))*180/np.pi
        if relative_angle<0:
            relative_angle=relative_angle+360
        if (theta_left-90)<relative_angle<(theta_left+90):
            if x_pos_left+dx_left >x_pos_left_lim[1] or x_pos_left+dx_left<x_pos_left_lim[0] or y_pos_left+dy_left<y_pos_left_lim[0] or y_pos_left+dy_left>y_pos_left_lim[1]:
    #         if x_pos_left+dx_left >58 or x_pos_left+dx_left<2 or y_pos_left+dy_left<28 or y_pos_left+dy_left>32:
#                 print('3',x_pos_left)
                x_pos_left,y_pos_left,dx_left,dy_left,theta_left,angle_change_left=meetwall(theta_meet_wall,unit_distance,x_pos_left,y_pos_left,theta_left)
            else:
                distance=((x_pos_left-x_pos_right)**2+(y_pos_left-y_pos_right)**2)**0.5
                value=1/distance
                theta_have_fish=havefish(relative_angle=relative_angle,theta_angle=theta_left,value=value,center_mean=0,center_std=5,center_number=1000,side_approach_mean=30,side_appraoch_std=10,side_avoid_mean=70,side_avoid_std=10)
                x_pos_left,y_pos_left,dx_left,dy_left,theta_left,angle_change_left=normalwalk(theta_have_fish,unit_distance,x_pos_left,y_pos_left,theta_left)
                if x_pos_left<x_pos_left_lim[0]:
                    x_pos_left=x_pos_left_lim[0]
                if x_pos_left>x_pos_left_lim[1]:
                    x_pos_left=x_pos_left_lim[1]
                if y_pos_left<y_pos_left_lim[0]:
                    y_pos_left=y_pos_left_lim[0]
                if y_pos_left>y_pos_left_lim[1]:
                    y_pos_left=y_pos_left_lim[1]
        else:        
            if x_pos_left+dx_left >x_pos_left_lim[1] or x_pos_left+dx_left<x_pos_left_lim[0] or y_pos_left+dy_left<y_pos_left_lim[0] or y_pos_left+dy_left>y_pos_left_lim[1]:
    #         if x_pos_left+dx_left >58 or x_pos_left+dx_left<2 or y_pos_left+dy_left<28 or y_pos_left+dy_left>32:


                x_pos_left,y_pos_left,dx_left,dy_left,theta_left,angle_change_left=meetwall(theta_meet_wall,unit_distance,x_pos_left,y_pos_left,theta_left)
            else:
                x_pos_left,y_pos_left,dx_left,dy_left,theta_left,angle_change_left=normalwalk(theta_distribution,unit_distance,x_pos_left,y_pos_left,theta_left)
                if x_pos_left<x_pos_left_lim[0]:
                    x_pos_left=x_pos_left_lim[0]
                if x_pos_left>x_pos_left_lim[1]:
                    x_pos_left=x_pos_left_lim[1]
                if y_pos_left<y_pos_left_lim[0]:
                    y_pos_left=y_pos_left_lim[0]
                if y_pos_left>y_pos_left_lim[1]:
                    y_pos_left=y_pos_left_lim[1]
        x_pos_right=position_right[0]
        y_pos_right=position_right[1]
        relative_angle=np.arctan((y_pos_right-y_pos_left)/(x_pos_right-x_pos_left))*180/np.pi+180
        ##################################################################################################
        if (theta_right-90)<relative_angle<(theta_right+90):
            if x_pos_right+dx_right >x_pos_right_lim[1] or x_pos_right+dx_right<x_pos_right_lim[0] or y_pos_right+dy_right<y_pos_right_lim[0] or y_pos_right+dy_right>y_pos_right_lim[1]:
    #         if x_pos_right+dx_right >70 or x_pos_right+dx_right<62 or y_pos_right+dy_right<25 or y_pos_right+dy_right>35:
                x_pos_right,y_pos_right,dx_right,dy_right,theta_right,angle_change_right=meetwall(theta_meet_wall,unit_distance,x_pos_right,y_pos_right,theta_right)
#                 change_angle.append(angle_change_right[0])
            else:
                distance=((x_pos_left-x_pos_right)**2+(y_pos_left-y_pos_right)**2)**0.5
                value=1/distance
                theta_have_fish=havefish(relative_angle=relative_angle,theta_angle=theta_right,value=value,center_mean=0,center_std=5,center_number=1000,side_approach_mean=30,side_appraoch_std=10,side_avoid_mean=70,side_avoid_std=10)            
                x_pos_right,y_pos_right,dx_right,dy_right,theta_right,angle_change_right=normalwalk(theta_have_fish,unit_distance,x_pos_right,y_pos_right,theta_right)
#                 change_angle.append(angle_change_right[0])
                if x_pos_right<x_pos_right_lim[0]:
                    x_pos_right=x_pos_right_lim[0]
                if x_pos_right>x_pos_right_lim[1]:
                    x_pos_right=x_pos_right_lim[1]
                if y_pos_left<y_pos_right_lim[0]:
                    y_pos_left=y_pos_right_lim[0]
                if y_pos_left>y_pos_right_lim[1]:
                    y_pos_left=y_pos_right_lim[1]
        else:        
            if x_pos_right+dx_right >x_pos_right_lim[1] or x_pos_right+dx_right<x_pos_right_lim[0] or y_pos_right+dy_right<y_pos_right_lim[0] or y_pos_right+dy_right>y_pos_right_lim[1]:
    #         if x_pos_right+dx_right >70 or x_pos_right+dx_right<62 or y_pos_right+dy_right<25 or y_pos_right+dy_right>35:


                x_pos_right,y_pos_right,dx_right,dy_right,theta_right,angle_change_right=meetwall(theta_meet_wall,unit_distance,x_pos_right,y_pos_right,theta_right)
#                 change_angle.append(angle_change_right[0])
            else:
                x_pos_right,y_pos_right,dx_right,dy_right,theta_right,angle_change_right=normalwalk(theta_distribution,unit_distance,x_pos_right,y_pos_right,theta_right)
#                 change_angle.append(angle_change_right[0])
                if x_pos_right<x_pos_right_lim[0]:
                    x_pos_right=x_pos_right_lim[0]
                if x_pos_right>x_pos_right_lim[1]:
                    x_pos_right=x_pos_right_lim[1]
                if y_pos_left<y_pos_right_lim[0]:
                    y_pos_left=y_pos_right_lim[0]
                if y_pos_left>y_pos_right_lim[1]:
                    y_pos_left=y_pos_right_lim[1]
        position_left=[x_pos_left,y_pos_left,theta_left]
        position_right=[x_pos_right,y_pos_right,theta_right]
        fish_state_left.append(position_left)
        fish_state_right.append(position_right)
    fish_state_left=np.asarray(fish_state_left)
    fish_state_right=np.asarray(fish_state_right)
#     print(fish_state_left)
    return fish_state_left,fish_state_right
left=[]
right=[]
for i in range(4):
    fish_left,fish_right=simulate_once(theta_distribution,theta_meet_wall,x_pos_left_lim,x_pos_right_lim,y_pos_left_lim,unit_distance)
#     print(fish_left)
    left.extend(fish_left)
    right.extend(fish_right)


left=np.asarray(left)
left_new=[]
right_new=[]



for j in range(len(left)):   
    a=np.insert(left[j],0,j)
    b=np.insert(right[j],0,j)
    b[1]=b[1]-60
    left_new.append(a)
    right_new.append(b)
left_new=np.asarray(left_new)
right_new=np.asarray(right_new)


# print(right_new)
# np.save("/Users/xiangning/Desktop/MCB 111 project/simulation/"+"left_simulated_extracted_x_y_ang_100pairs_60mm.npy", left_new)   

# np.save("/Users/xiangning/Desktop/MCB 111 project/simulation/"+"righ_simulated_extracted_x_y_ang_100pairs_60mm.npy", right_new)   



plt.plot(left_new[:,1],left_new[:,2])
# plt.plot(right_new[:,0],right_new[:,1])
# plt.xlim(0,120)
# plt.ylim(0,60)

plt.show()
# change_angle=pd.Series(change_angle)
# # print(change_angle)
# change_angle.hist(bins=40)
# plt.show()



from matplotlib import gridspec
import matplotlib.pyplot as plt
import math

from scipy.stats import gaussian_kde
left_x2=60
data1=left_new
data2=right_new
leftdata = data1.copy()
for j in range(4):  # index out change 5 to 4
    for i in range(len(leftdata)):
        # for i in range(2):

        if data1[i, j] == np.nan:
            leftdata[i, j] = data1[i - 1, j]
rightdata = data2.copy()
for j in range(4):  # index out change 5 to 4
    for i in range(len(leftdata)):
        # for i in range(2):

        if data2[i, j] == np.nan:
            rightdata[i, j] = data2[i - 1, j]

fig = plt.figure(figsize=(8, 16))
# fig = plt.figure(figsize=(3, 5))

gs = gridspec.GridSpec(5, 2, height_ratios=[2,2,2,2,2])

ax1 = plt.subplot(gs[0, :])
distance=[]
leftdata = leftdata.tolist()
rightdata = rightdata.tolist()
for i in range(len(leftdata)):
    dis=rightdata[i][1]+(left_x2-leftdata[i][1])
    distance.append(dis)
distance=np.asarray(distance)
ax1 = plt.plot(distance, color="hotpink")
ax1 = plt.xlabel('frame', fontsize=10)
ax1 = plt.ylabel('x distance')
print('fig1')
ax2 = plt.subplot(gs[1, :])
distance=[]
for i in range(len(leftdata)):
    dis=rightdata[i][2]-leftdata[i][2]
    distance.append(dis)
distance=np.asarray(distance)
ax2 = plt.plot(distance, color="hotpink")
ax2 = plt.xlabel('frame', fontsize=10)
ax2 = plt.ylabel('y distance')
distance = []
print('fig2')
ax3 = plt.subplot(gs[2, :])
for i in range(len(leftdata)):
    dis = ((rightdata[i][2] - leftdata[i][2])**2+(rightdata[i][1]+(left_x2-leftdata[i][1]))**2)**0.5
    distance.append(dis)
distance = np.asarray(distance)
ax3 = plt.plot(distance, color="orange")
ax3 = plt.xlabel('frame', fontsize=10)
ax3 = plt.ylabel('distance')


leftdata=np.asarray(leftdata)
rightdata=np.asarray(rightdata)
print('fig3')
ax4 = plt.subplot(gs[3, :])
ax4 = plt.plot(leftdata[:, 1], leftdata[:, 2], color="red")
ax4 = plt.plot(rightdata[:, 1]+left_x2, rightdata[:, 2], color="blue")
ax4 = plt.xlim(-5, 125)
ax4 = plt.ylim(-5, 65)
ax4 = plt.xlabel("x position (mm)")
ax4 = plt.ylabel("y position (mm)")
print('fig4')
ax4 = plt.subplot(gs[4, :])
ax4 = plt.xlim(-5, 125)
ax4 = plt.ylim(-5, 65)
ax4 = plt.xlabel("x position (mm)")
ax4 = plt.ylabel("y position (mm)")
x = np.nan_to_num(leftdata[:, 1])
y = np.nan_to_num(leftdata[:, 2])
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=50, edgecolor='')
x = np.nan_to_num(rightdata[:, 1]+left_x2)
y = np.nan_to_num(rightdata[:, 2])
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=50, edgecolor='')






