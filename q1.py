#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:22:44 2022

@author: alirezakheirandish
"""
import cv2
import numpy as np

# Check if a point is inside a rectangle
def rect_contains(rect, point):

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Write the delaunay triangles into a file
def draw_delaunay(f_w, f_h, subdiv, dictionary1):

    list4 = []

    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            list4.append((dictionary1[pt1],dictionary1[pt2],dictionary1[pt3]))

    dictionary1 = {}
    return list4

def make_delaunay(f_w, f_h, theList, img1, img2):
    

    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary. 
    theList = theList.tolist()
    points = [(int(x[0]),int(x[1])) for x in theList]
    dictionary = {x[0]:x[1] for x in list(zip(points, range(76)))}
    
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    list4 = draw_delaunay(f_w, f_h, subdiv, dictionary)
   
    # Return the list.
    return list4


def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


def bounds(t):
    t_1 = (t[0])
    t_2 = (t[1])
    t_3 = (t[2])
    min_x = np.int(np.min([t_1[0],t_3[0],t_2[0]]))
    min_y = np.int(np.min([t_1[1],t_3[1],t_2[1]]))
    max_x = np.int(np.max([t_1[0],t_3[0],t_2[0]]))
    max_y = np.int(np.max([t_1[1],t_3[1],t_2[1]]))
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    return [min_x,min_y,w,h]


original_array = np.loadtxt("points1.txt").reshape(76, 2)




original_array = np.int64(original_array)
points1 = original_array.tolist()

original_array = np.loadtxt("points2.txt").reshape(76, 2)
original_array = np.int64(original_array)
points2 = original_array.tolist()


list3 = points1.copy()
np.array(points1[0])+np.array(points1[1])
for i in range(len(points1)):
    temp = (np.array(points1[i])+np.array(points2[i]))/2
    
    list3[i] = temp.tolist()

list3 = np.array(list3)

img1 = cv2.imread('res01.jpeg')
img2 = cv2.imread('res02.jpeg')
image1 = cv2.imread('res01.jpeg')
image2 = cv2.imread('res02.jpeg')

size = (img1.shape[0],img1.shape[1])

tri = make_delaunay(size[1], size[0], list3, img1, img2)

num_images = int(45)
img1 = np.float32(image1)
img2 = np.float32(image2)
import imageio
import matplotlib.pyplot as plt
import os

filenames = []
for frame in range(0, num_images):
    filename = f'frame_{1}_{frame}.png'
    filenames.append(filename)

#         if (frame == n_frames):
#             for i in range(5):
#                 filenames.append(filename)
# j = 0


    points = []
    alpha = frame/(num_images-1)

    for i in range(len(points1)):
        points.append(((1 - alpha) * points1[i][0] + alpha * points2[i][0],(1 - alpha) * points1[i][1] + alpha * points2[i][1]))



        
    frame_img = np.zeros(img1.shape, dtype = img1.dtype)

    for i in range(len(tri)):    
        t1 = [points1[int(tri[i][0])], points1[int(tri[i][1])], points1[int(tri[i][2])]]
        t2 = [points2[int(tri[i][0])], points2[int(tri[i][1])], points2[int(tri[i][2])]]
        t = [points[int(tri[i][0])], points[int(tri[i][1])], points[int(tri[i][2])]]

        r1 = bounds(t1)
        r2 = bounds(t2)
        r = bounds(t)

        t1_0 = []
        t2_0 = []
        t_0 = []

        for k in range(3):
            
            t1_0.append(((t1[k][0] - r1[0]),(t1[k][1] - r1[1])))
            t2_0.append(((t2[k][0] - r2[0]),(t2[k][1] - r2[1])))
            t_0.append(((t[k][0] - r[0]),(t[k][1] - r[1])))


        mask = np.zeros((r[3], r[2], 3), dtype = np.float64)
        cv2.fillConvexPoly(mask, np.int32(t_0), (1.0, 1.0, 1.0), 16, 0)
        

        img1_0 = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2_0 = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        warpImage1 = apply_affine_transform(img1_0, t1_0, t_0, size)
        warpImage2 = apply_affine_transform(img2_0, t2_0, t_0, size)


        img_frame = (1.0 - alpha) * warpImage1 + alpha * warpImage2


        frame_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = frame_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + img_frame * mask
    
    res = (cv2.cvtColor(np.uint8(frame_img), cv2.COLOR_BGR2RGB))
    
    plt.imsave(filename,res)
    if frame == 14:
        plt.imsave('res03.jpg',res)
    if frame == 29:
        plt.imsave('res04.jpg',res)
        


img1 = np.float32(image2)
img2 = np.float32(image1)
points1,points2 = points2,points1

# filenames = []
for frame in range(num_images, num_images + num_images):
    filename = f'frame_{1}_{frame}.png'
    filenames.append(filename)

#         if (frame == n_frames):
#             for i in range(5):
#                 filenames.append(filename)
# j = 0



    points = []
    alpha = (frame-num_images)/(num_images-1)


    for i in range(len(points1)):
        points.append(((1 - alpha) * points1[i][0] + alpha * points2[i][0],(1 - alpha) * points1[i][1] + alpha * points2[i][1]))




    frame_img = np.zeros(img1.shape, dtype = img1.dtype)

    for i in range(len(tri)):    
        t1 = [points1[int(tri[i][0])], points1[int(tri[i][1])], points1[int(tri[i][2])]]
        t2 = [points2[int(tri[i][0])], points2[int(tri[i][1])], points2[int(tri[i][2])]]
        t = [points[int(tri[i][0])], points[int(tri[i][1])], points[int(tri[i][2])]]

        r1 = bounds(t1)
        r2 = bounds(t2)
        r = bounds(t)


        t1_0 = []
        t2_0 = []
        t_0 = []

        for k in range(0, 3):
            
            t1_0.append(((t1[k][0] - r1[0]),(t1[k][1] - r1[1])))
            t2_0.append(((t2[k][0] - r2[0]),(t2[k][1] - r2[1])))
            t_0.append(((t[k][0] - r[0]),(t[k][1] - r[1])))


        mask = np.zeros((r[3], r[2], 3), dtype = np.float64)
        cv2.fillConvexPoly(mask, np.int32(t_0), (1.0, 1.0, 1.0), 16, 0)
        

        img1_0 = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2_0 = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        warpImage1 = apply_affine_transform(img1_0, t1_0, t_0, size)
        warpImage2 = apply_affine_transform(img2_0, t2_0, t_0, size)


        img_frame = (1.0 - alpha) * warpImage1 + alpha * warpImage2


        frame_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = frame_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + img_frame * mask
    
    res = (cv2.cvtColor(np.uint8(frame_img), cv2.COLOR_BGR2RGB))
    
    plt.imsave(filename,res)
#     if frame == 14:
#         plt.imsave('res03.jpg',res)
#     if frame == 29:
#         plt.imsave('res04.jpg',res)

with imageio.get_writer('morph1.gif', mode='I') as writer:
    for filename in filenames:
        temp = imageio.imread(filename)
        writer.append_data(temp)
for filename in set(filenames):
    os.remove(filename)

points1,points2 = points2,points1
