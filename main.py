# %matplotlib inline
#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook

import cv2
import numpy as np
from multiprocessing.pool import ThreadPool as Pool

source_image = cv2.imread("base.jpg")
h, w = source_image.shape[:2]
# templates = ["1_1.jpg"]
templates = ["1_1.jpg", "1_2.jpg", "1_3.jpg", "1_4.jpg", "1_5.jpg", "1_6.jpg", "1_7.jpg", "1_8.jpg", "1_9.jpg", "1_10.jpg", "1_11.jpg", "1_12.jpg", "1_13.jpg", "1_14.jpg", "1_15.jpg",]
def matching(img):
    template = cv2.imread(img)
    found = None
    (tH, tW) = template.shape[:2]
    # cv2_imshow(template)

    tEdged = cv2.Canny(template, 50, 200)

    for scale in range(10, 21):
        resized = cv2.resize(source_image, dsize = (0,0), fx = scale/10, fy = scale/10)
        # print(scale)
        r = source_image.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, tEdged, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    cv2.rectangle(source_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # cv2_imshow(image)
    # cv2.imwrite('output.jpg', image)
    # ~ cv2.waitKey(0)

pool_size = 15 

pool = Pool(pool_size)

for img in templates:
    pool.apply_async(matching, (img,))

pool.close()
pool.join()
cv2.imshow("frame",source_image)
cv2.waitKey(0)

