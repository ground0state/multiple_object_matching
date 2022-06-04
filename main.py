from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


img1 = cv2.imread('query.png', 0)  # queryImage
img2 = cv2.imread('test.png', 0)  # trainImage

orb = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold=5)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
des1 = np.float32(des1)
des2 = np.float32(des2)

x = []
for i_kp in kp2:
    x.append(np.array(i_kp.pt))
x = np.array(x)

# MeanShiftのパラメータであるbandwidthを推定する
bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)
# 検出対象データのkeypointをMeanShiftでクラスタリングする
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)


for i in range(n_clusters_):
    # iに一致するkeypointのインデックス
    d, = np.where(ms.labels_ == i)
    kp = list(kp2[j] for j in d)
    des = des2[d]

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des, 2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 3:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

        if M is None:
            print("No Homography")
        else:
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            img3 = cv2.drawMatches(img1, kp1, img2, kp, good, None, **draw_params)

            plt.imshow(img3, 'gray'), plt.show()

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
