"""
@author: The students of spring semester 2018 LS100

"""

import imageio                  # this library allows us to load movies of various compression formats
import numpy as np              # this is the standard python library for doing data data analysis of matrices
import os                       # this is a helper library for the operating system, which allows to concatenate a path, for example

import cv2
import sys
import matplotlib


def combine_dataset(root_path, fish_names, half):
    # loop through all those fish names, and calculate their backgroud images
    dataset_all=[]
    for fish_name in fish_names:


        print("Combine dataset", fish_name)


        # concatenate the root path, and the fish name
        # path=os.path.join(root_path, fish_name)
        path = os.path.join(root_path, fish_name+half+'_180422_extracted_x_y_ang.npy')
        # path = os.path.join(root_path,half+ '_201903_extracted_x_y_ang_2_fish_'+fish_name+'.npy')

        data=np.load(path)
        # for i in range(1680):
        #     data=np.delete(data,[int(len(data)-1)],0)
        data_list=data.tolist()
        dataset_all.extend(data_list[0:9600])
        # np.save(path[:-4]+'_180422_extracted_x_y_ang_2min.npy',np.asarray(data_list[0:6400]))
    dataset_array=np.asarray(dataset_all)
    # print(len(dataset_array),6800*13)
    np.save(path[:-4] + half + "_extracted_x_y_ang_fixed_fish_population2.npy", dataset_array)
    # np.save(root_path[:-4]+ half+"_180422_extracted_x_y_ang_2min.npy", dataset_array)

    return dataset_array



def get_fish_position_and_angle(frame_counter,frame, background, threshold, filter_width, display,x1,x2,y1,y2):

    # we subtract the background and frame,
    # the fish is normally darker than the background, so we take the absolute value to
    # make the fish the brightest area in the image
    # because movie and background are likely of type unsigned int, for the sake
    # of the subtraction, make them signed integers
    background_substracted_image = np.abs(background.astype(np.int) - frame.astype(np.int)).astype(np.uint8)

    # threshold, play around with the thresholding parameter for optimal results

    background_substracted_image_left = background_substracted_image[y1:y2, x1:x2]
    # background_substracted_image_left=background_substracted_image(Rect(x1,y1,(x2-x1),(y2-y1)))
    # print('saving',type(background_substracted_image_left ))
    cv2.imwrite('/Users/xiangning/Desktop/Harvard/iamging/cut/background_cut_2fish.png', background_substracted_image_left)
    ret, fish_image_thresholded = cv2.threshold(background_substracted_image_left, threshold, 255, cv2.THRESH_BINARY)

    # apply a weak gaussian blur to the thresholded image to get rid of noisy pixels
    # play around with the standard deviation for optimal results, normally, if little noise
    # use small values
    fish_image_blurred = cv2.GaussianBlur(fish_image_thresholded, (filter_width, filter_width), 0)
    # imageio.imwrite(str(frame_counter) + "saperated.png", fish_image_blurred )

    # find the position of the maximum with
    x, y = np.unravel_index(np.argmax(fish_image_blurred), fish_image_blurred.shape)

    # cut out a region of interest square 100 pixels around the fish
    x_left = x - 50
    x_right = x + 50
    y_up = y - 50
    y_down = y + 50

    # we have to take same care of rare incidences where the region of interest would fall outside the movie
    if x_left < 0:
        x_left = 0
        x_right = x_left + 100

    if x_right >= background_substracted_image.shape[0]:
        x_right = background_substracted_image.shape[0] - 1
        x_left = x_right - 100

    if y_up < 0:
        y_up = 0
        y_down = y_up + 100

    if y_down >= background_substracted_image.shape[1]:
        y_down = background_substracted_image.shape[1] - 1
        y_up = y_down - 100

    # copy that region from the thresholded image and resize a little
    fish_roi_cutout = cv2.resize(fish_image_thresholded[x_left:x_right, y_up:y_down], (200, 200))
    contours,hierarchy= cv2.findContours(fish_roi_cutout, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # find the biggest contour
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    cnt = contours[np.argmax(contour_areas)]

    # determine a convex hull around that contour
    hull = cv2.convexHull(cnt)

    # if we found not enough fish points
    if hull.shape[0] < 3:
        print("No fish found. Please check thresholding parameters, etc.")
        return np.nan, np.nan, np.nan

    # move the fish coordinates to the center
    hull_moved = hull.copy()
    hull_moved[:, :, 0] = hull_moved[:, :, 0] - hull[:, 0, 0].mean()
    hull_moved[:, :, 1] = hull_moved[:, :, 1] - hull[:, 0, 1].mean()

    # see https://en.wikipedia.org/wiki/Image_moment
    moments = cv2.moments(hull_moved)
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]

    fish_orientation = 0.5*np.arctan2(2 * mu11, mu20 - mu02)

    M = cv2.getRotationMatrix2D((100, 100), fish_orientation * 180/np.pi, 1)

    dummy_image_original = np.zeros((200, 200), dtype=np.uint8)

    cv2.drawContours(dummy_image_original, [hull_moved+100], 0, 255, cv2.FILLED)
    dummy_image_rotated = cv2.warpAffine(dummy_image_original, M, (200, 200))

    fish_width = np.sum(dummy_image_rotated.copy().astype(np.int), axis=0)

    if np.argmax(fish_width) < 100:
        fish_orientation += np.pi

    if display:
        img = np.zeros((200, 200)).astype(np.uint8)
        img = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

        cv2.drawContours(img, [hull_moved + 100], 0, (255, 255, 255))

        dy = int(np.cos(fish_orientation) * 20)
        dx = int(np.sin(fish_orientation) * 20)
        cv2.line(img, (100, 100), (100+dy, 100+dx), (0, 255, 0), thickness=2)

        img2 = np.concatenate((frame[y1:y2, x1:x2], fish_image_thresholded, fish_image_blurred), axis=1)
        img2 = cv2.cvtColor(img2.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # draw into the frame for displaying
        dy = int(np.cos(fish_orientation) * 20)
        dx = int(np.sin(fish_orientation) * 20)

        cv2.line(img2, (y, x), (y + dy, x + dx), (0, 255, 0), thickness=2)
        cv2.circle(img2, (y + dy, x + dx), 3, (0, 0, 255), thickness=-1)

        img2 = cv2.resize(img2, (600, 200))

        image_for_display = np.concatenate((img2, img), axis=1)
        cv2.imshow("fish image", image_for_display)

        if cv2.waitKey(1) == 27:
            sys.exit()

    # swap y, and x,
    return int(y), int(y2-x-y1), (-fish_orientation*180/np.pi + 360) % 360

def extract_position_orientation(root_path, fish_names, threshold, filter_width, display, x1, x2, y1, y2,half):

    # loop through all those fish names, and calculate their backgroud images
    for fish_name in fish_names:

        print("Extracting position and orientation information for fish", fish_name)

        # concatenate the root path, and the fish name
        path = os.path.join(root_path, fish_name)

        # load the background
        background = np.load(path[:-4] + "_background.npy")

        # load the fish movie
        movie = imageio.get_reader(path)
        dt = 1 / movie.get_meta_data()['fps']

        ts = []
        xs = []
        ys = []
        fish_orientations = []

        pixel_to_mm = 60/ (((x2 - x1 )+ (y2- y1)) / 2)

        for frame_counter, frame in enumerate(movie):
            if frame_counter % 2 == 0:

                print("Analyzing", frame_counter)

                image = frame[:, :, 0]
                x, y, fish_orientation = get_fish_position_and_angle(frame_counter,image,
                                                                     background,
                                                                     threshold=threshold,
                                                                     filter_width=filter_width,
                                                                     display=display,x1=x1,x2=x2,y1=y1,y2=y2)

                x = (x ) * pixel_to_mm
                y = (y) * pixel_to_mm

                ts.append(frame_counter * dt)
                xs.append(x)
                ys.append(y)
                fish_orientations.append(fish_orientation)

            frame_counter += 1

            if frame_counter > 12000:
                break

        # determine the accumulated orientation
        delta_orientations = np.diff(fish_orientations)
        delta_orientations = np.nan_to_num(delta_orientations)

        ind1 = np.where(delta_orientations > 250)
        ind2 = np.where(delta_orientations < -250)

        delta_orientations[ind1] = delta_orientations[ind1] - 360
        delta_orientations[ind2] = delta_orientations[ind2] + 360

        fish_accumulated_orientation = np.cumsum(np.r_[fish_orientations[0], delta_orientations])

        # Saving the data in the same folder and the same base file name as the fish movie
        fish_data = np.c_[ts, xs, ys, fish_orientations]
        np.save(path[:-4] + half+"_180422_extracted_x_y_ang.npy", fish_data)
    return fish_data,dt
        # save a plot to easily check the quality of the tracking
def makefigure(root_path,fish_names,fish_data,half,dt):
    for i in range(1):
        path = root_path
        from matplotlib import gridspec
        import matplotlib.pyplot as plt
        import math

        from scipy.stats import gaussian_kde


        data2 = fish_data.copy()
        # print(data2)
        for j in range(4):# index out change 5 to 4
            for i in range(len(fish_data)):
            # for i in range(2):

                if fish_data[i,j] == np.nan:
                    data2[i,j] = fish_data[i-1,j]

        fig = plt.figure(figsize=(12, 32))
        # fig = plt.figure(figsize=(3, 5))

        gs = gridspec.GridSpec(8, 6, height_ratios=[3, 1, 3, 1, 2, 2, 2, 2])

        #plt.rc('xtick',labelsize=16)
        #plt.rc('ytick',labelsize=16)

        ax1 = plt.subplot(gs[0,:3])
        ax1=plt.plot(data2[:,1], data2[:,2], color="black")
        ax1=plt.xlim(-5, 65)
        ax1=plt.ylim(-5, 65)
        ax1=plt.xlabel("x position (mm)")
        ax1=plt.ylabel("y position (mm)")
        print('ax1')
        ax2 = plt.subplot(gs[0, 3:], polar=True)
        N = 180
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        abc = np.nan_to_num(data2[:,3])
        radii, tick = np.histogram(abc, bins = 180, range=(0,360), density=True)# density repalce normed
        width = (2*np.pi) / N
        ax2 = plt.bar(theta, radii, width=width, color="hotpink")
        print('ax2')
        # ax3 = plt.subplot(gs[1, 0])
        # ax3=plt.plot(data2[:,3], color="hotpink")
        # ax3=plt.xlabel( 'frame', fontsize=10 )
        # ax3=plt.ylabel( 'angle' )
        #
        # ax4 = plt.subplot(gs[1, -1])
        # N = 180
        # ax4= plt.hist( data2[:,3], N, facecolor="hotpink",
        #                               range=[-5,365], density=True)
        # ax4=plt.xlabel( 'angle' )
        # ax4=plt.ylabel( 'Probability' )

        ax5 = plt.subplot(gs[1, :])
        ax5=plt.plot(data2[:,1], color="orange")
        ax5=plt.plot(data2[:,2], color="blue")
        ax5=plt.xlabel( 'frame', fontsize=10 )
        ax5=plt.ylabel( 'distance' )
        print('ax3')
        plt.subplot(gs[2,-3:])
        x = np.nan_to_num(data2[:,1])
        y = np.nan_to_num(data2[:,2])
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        plt.scatter(x, y, c=z, s=50, edgecolor='')
        plt.colorbar()
        plt.xlim(-5, 65)
        plt.ylim(-5, 65)
        plt.ylabel( 'y position' )
        plt.xlabel( 'x position' )

        print('ax4')

        ax6 = plt.subplot(gs[3, -3:])
        N = 200
        ax6= plt.hist( data2[:,1], N, facecolor="orange",
                                          range=[-5,60], density=True)
        ax6=plt.xlabel( 'x position' )
        ax6=plt.ylabel( 'Probability' )
        print('ax5')
        ax7 = plt.subplot(gs[2,2])
        N = 200
        base = plt.gca().transData
        rot =  matplotlib.transforms.Affine2D().rotate_deg(90)

        ax7= plt.hist( data2[:,2], N, facecolor="blue", orientation='horizontal',
                                          range=[-5,60], density=True)
        ax7=plt.xlabel( 'Probability' )
        ax7=plt.ylabel( 'y position' )

        plt.savefig(path+half+"population2_fixed_fish_left.png")

def interact_figure(root_path,fish_names,data1,data2,left_x2):
    for i in range(1):
        path = root_path
        from matplotlib import gridspec
        import matplotlib.pyplot as plt
        import math

        from scipy.stats import gaussian_kde

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

        plt.colorbar()
        plt.savefig(path+ "67811_2min_combined.png")




root_path = r"/Users/xiangning/Desktop/MCB 111 project/201904_fixed_fish/201904_with_right_fish/"
population1=['20190402_05','20190405_04','20190405_07']
population2=['20190402_01','20190402_02','20190402_03','20190402_07','20190402_08','20190402_13','20190405_01','20190405_02','20190405_03','20190405_05','20190405_06','20190405_08','20190405_09','20190405_10']
right_fish_data=combine_dataset(root_path, population2, half='right')
left_fish_data=combine_dataset(root_path, population2, half='left')
makefigure(root_path, fish_names=poulation2,fish_data=right_fish_data,half='right',dt=dt)
makefigure(root_path, fish_names=population2,fish_data=left_fish_data,half='left',dt=dt)
interact_figure(root_path, fish_names=fish_names,data1=left_fish_data,data2=right_fish_data,left_x2=62)
