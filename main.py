import cv2
import sys
import math
import numpy as np

# constants
offset = 5 #the vertical pixel offset when looking at colors


np.set_printoptions(threshold=sys.maxsize)
image = cv2.resize(cv2.imread("images/moa6.png",0), (1000,625))
#make binary (black/white) image
ret, binImg = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

debugImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

vertical_edge_kernel = np.array(
    [[-.5,-1,-1,-1,-.5],
     [-.5,-1,-1,-1,-.5],
     [  0, 0, 0, 0,  0],
     [ .5, 1, 1, 1, .5],
     [ .5, 1, 1, 1, .5]]) * 1/8

edges = cv2.convertScaleAbs(cv2.filter2D(image, cv2.CV_64F, vertical_edge_kernel))

ret, filteredEdges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)


lines = cv2.HoughLinesP(filteredEdges, 1, 1 * np.pi / 180, 150, None, 32, 32)

if (not lines is None):
    line_lengths = np.zeros(len(lines))
    for i in range(0, len(lines)):
        l = lines[i][0]
        line_lengths[i] = (l[0]-l[2]) ** 2 + (l[1]-l[3]) ** 2

    print(line_lengths)

    line_ranks = np.flip(np.argsort(line_lengths))


    #data extraction
    for i in range(0, len(lines)):

            l = lines[line_ranks[i]][0]

            size = int(math.sqrt((l[0]-l[2]) ** 2 + (l[1]-l[3]) ** 2) / 2) # make a point per 2 pixel length of the line
            xs = tuple(np.linspace(max(l[0], 0), min(l[2], edges.shape[1]-1), size).astype(int))
            ys = tuple(np.linspace(max(l[1] - offset, 0), min(l[3] - offset, edges.shape[0]-1), size).astype(int))
            
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
                distThreshold = np.sort(dists)[int(len(dists) / 4)] #the median size of a small block is the 25% percentile of block size (maybe should be 33% as small blocks are in theory 2/3rds of blocks?)
                for j in reversed(range(len(points) - 1)):
                    if (dists[j] > distThreshold * 1.5): #if larger than median small block * 1.5, assume a large block (add a bit)
                        points.insert(j+1, int((points[j] + points[j+1]) / 2))
                        bits.insert(j+1, bits[j])
                    elif (dists[j] < distThreshold * 0.5): #if smaller than median small block * 0.5, assume noise (remove a bit)
                        points.pop(j)
                        bits.pop(j)
            
            #print(bits)

            if (len(bits) >= 16):
                cv2.line(debugImage, (l[0], l[1]), (l[2], l[3]), (0,215,255), 1, cv2.LINE_AA)
            else:
                cv2.line(debugImage, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

            for j in range(len(points)):
                if (bits[j] == 0):
                    cv2.circle(debugImage, (xs[points[j]], ys[points[j]] + offset), 4, (0,255,0), -1)
                else:
                    cv2.circle(debugImage, (xs[points[j]], ys[points[j]] + offset), 4, (255,0,0), -1)
            

            if (len(bits) >= 16): # we found a match! should do bit checking in this condition
                # do solve PNP wow
                break

#cv2.imshow("binimg", binImg)
cv2.imshow("debug", debugImage)
cv2.imshow("edges", edges)
cv2.imshow("filtered", filteredEdges)
cv2.waitKey(0)