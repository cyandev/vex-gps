import cv2
import sys
import math
import numpy as np


# constants
offset = 5 #the vertical pixel offset when looking at colors


np.set_printoptions(threshold=sys.maxsize)
image = cv2.imread("images/moa5.png",0)
image = cv2.blur(image,(3,3))
ret, binImg = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

debugImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

kernel = np.array([[0,-1,0],[0,0,0],[0,1,0]])

sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
absSobel = cv2.convertScaleAbs(sobel)
# absSobel = cv2.blur(absSobel,(5,5))
ret, absSobel = cv2.threshold(absSobel,200,255,cv2.THRESH_BINARY)

lines = cv2.HoughLinesP(absSobel, 1, np.pi / 180, 150, None, 256, 32)

#data extraction
for i in range(0, len(lines)):

        l = lines[i][0]

        print(l)
        size = int(math.sqrt((l[0]-l[2]) ** 2 + (l[1]-l[3]) ** 2)) # make a point per pixel length of the line
        xs = tuple(np.linspace(max(l[0], 0), min(l[2], absSobel.shape[1]-1), size).astype(int))
        ys = tuple(np.linspace(max(l[1] - offset, 0), min(l[3] - offset, absSobel.shape[0]-1), size).astype(int))
        
        vals = binImg[(ys,xs)]

        #algo time
        thresh = 50
        dir = 0 #direction of the last detection
        prev = False
        dists = []
        points = []
        bits = []
        for j in range(size):
            # cv2.circle(debugImage, (xs[j], ys[j]+offset), 4, (0,0,0), -1)
            if dir != -1 and vals[j] < thresh:
                # cv2.circle(debugImage, (xs[j], ys[j]+offset), 4, (0,255,0), -1)
                points.append(j)
                bits.append(0)
                dir = -1
                if (prev):
                     dists.append(math.sqrt((xs[j]-prev[0])**2 + (ys[j]-prev[1])**2))
                prev = (xs[j],ys[j])

            elif dir != 1 and vals[j] > 255-thresh:
                # cv2.circle(debugImage, (xs[j], ys[j]+offset), 4, (255,0,0), -1)
                points.append(j)
                bits.append(1)
                dir = 1
                if (prev):
                     dists.append(math.sqrt((xs[j]-prev[0])**2 + (ys[j]-prev[1])**2))
                prev = (xs[j],ys[j])

        #after this loop dists doesn't mean anything
        if (dists):
            distThreshold = np.sort(dists)[int(len(dists) / 4)] #magic number 25%
            for j in reversed(range(len(points) - 1)):
                if (dists[j] > distThreshold * 1.5): #magic number 1.5x
                    points.insert(j+1, int((points[j] + points[j+1]) / 2))
                    bits.insert(j+1, bits[j])
        
        print(bits)

        if (True):
            if (len(bits) >= 16):
                cv2.line(debugImage, (l[0], l[1]), (l[2], l[3]), (0,255,0), 1, cv2.LINE_AA)
            # else:
            #     cv2.line(debugImage, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

            for j in range(len(points)):
                if (bits[j] == 0):
                    cv2.circle(debugImage, (xs[points[j]], ys[points[j]] + offset), 4, (0,255,0), -1)
                else:
                    cv2.circle(debugImage, (xs[points[j]], ys[points[j]] + offset), 4, (255,0,0), -1)

            
cv2.imshow("image", binImg)
cv2.imshow("debug", debugImage)
cv2.imshow("absSobel", absSobel)
        

cv2.waitKey(0)