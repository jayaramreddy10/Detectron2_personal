    
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
np.set_printoptions(suppress=True)
import copy
import random

# util functions 
def plot_images(img_set, n_r, n_c, img_titles):
    fig = plt.figure(figsize = (19, 19))
    cnt = 0
    for i in range(n_r):
        for j in range(n_c):
            if cnt == len(img_set):
                break
            ax1 = fig.add_subplot(n_r, n_c, cnt + 1)
            ax1.imshow(img_set[cnt], cmap = 'gray')
            ax1.set_title(img_titles[cnt], fontsize = 15)
            cnt = cnt + 1
    plt.show() 

# step2: read pair of images given
img_1 = cv2.imread('0000.jpg')    #first image
img_2 = cv2.imread('0010.jpg')    #second image

# step3: get corresponding points using SIFT
minHessian = 400
sift = cv2.ORB_create()
kps_1, descriptors_1 = sift.detectAndCompute(img_1, None)
kps_2, descriptors_2 = sift.detectAndCompute(img_2, None)

#FLANN matcher
# -- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
# matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
# knn_matches = matcher.knnMatch(descriptors_1, descriptors_2, 2)
# #-- Filter matches using the Lowe's ratio test
# ratio_thresh = 0.7
# good_matches = []
# for m,n in knn_matches:
#     if m.distance < ratio_thresh * n.distance:
#         good_matches.append(m)
# n_good_matches = len(good_matches)
# good_matches = random.sample(good_matches, n_good_matches//4)

# # -- Draw matches
# img_matches = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1]+img_2.shape[1], 3), dtype=np.uint8)
# img_3 = cv2.drawMatches(img_1, kps_1, img_2, kps_2, good_matches[:], img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#FLANN matcher -- version 2 for ORB
# -- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)
knn_matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.7
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
n_good_matches = len(good_matches)
good_matches = random.sample(good_matches, n_good_matches//4)

# -- Draw matches
img_matches = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1]+img_2.shape[1], 3), dtype=np.uint8)
img_3 = cv2.drawMatches(img_1, kps_1, img_2, kps_2, good_matches[:], img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#bf MATCHER

# fm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)   #feature matching in cv2
# matches = fm.match(descriptors_1,descriptors_2)
# matches = sorted(matches, key = lambda x:x.distance)
# img_3 = cv2.drawMatches(img_1, kps_1, img_2, kps_2, matches, img_2, flags=2)
# #matches[:300]

# Save the image to a file
output_file = 'output_image.jpg'  # Replace 'output_image.jpg' with your desired file name
cv2.imwrite(output_file, img_3)

plt.imshow(img_3)
img_set = []
img_set.append(img_1)
img_set.append(img_2)
img_set.append(img_3)
plot_images(img_set, 3, 1, ['image 1','image 2', 'image with feature matching'])

#select best n feature matches
n_matches = 50
selected_matches = good_matches[:n_matches]

# every match is cv2.DMatch object and has attributes like distance, queryIdx, trainIdx etc
X1_list = [kps_1[match.queryIdx].pt for match in selected_matches] 
X2_list = [kps_2[match.trainIdx].pt for match in selected_matches]
# Save the matches to a file
matches_file = 'matches.txt'  # Replace 'matches.txt' with your desired file name

with open(matches_file, 'w') as f:
    for match in selected_matches:
        f.write(f"Distance: {match.distance}, Keypoint 1 idx: {match.queryIdx}, Keypoint 2 idx: {match.trainIdx}\n")

X1_list_all = [kps_1[match.queryIdx].pt for match in good_matches] 
X2_list_all = [kps_2[match.trainIdx].pt for match in good_matches]
print('no of correspondences used for estimating F matrix: {}'.format(len(X1_list)))
print('no of correspondences detected by SIFT in total: {}'.format(len(X1_list_all)))

assert len(X1_list) == len(X2_list), "no of features are not matching"
