# Copyright (c) 2012, Jan Erik Solem
# All rights reserved.
#
# Copyright (c) 2019, Anette Eltner
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os, csv
import numpy as np
import pylab as plt
import matplotlib

import cv2

import draw_functions as drawF


'''perform coregistration'''
def coregistration(image_list, directory_out, kp_nbr=None, sift_vers=False, 
                   feature_match_twosided=False, nbr_good_matches=10,
                   master_0 = True):
    
    if not os.path.exists(directory_out):
        os.system('mkdir ' + directory_out)
        '''执行cmd命令，创建directory——out文件'''
    
    master_img_name = image_list[1]
    master_img_dirs = master_img_name.split("/")    
    img_master = cv2.imread(master_img_name)
    #cv2.imwrite(os.path.join(directory_out, master_img_dirs[-1])[:-4] + '_coreg.jpg', img_master)
    '''读取master-img-name路径图片文件'''
    
    if master_0 == True:    #matchin to master
        '''detect Harris keypoints in master image主图像中Harris关键点的检测 '''
        keypoints_master, _ = HarrisCorners(master_img_name, kp_nbr, False)
        
        '''calculate ORB or SIFT descriptors in master image计算主图像中的ORB或SIFT描述符
         ORB（Oriented FAST and Rotated BRIEF）特征也是由关键点和描述子组成。
         正如其英文全名一样，这种特征使用的特征点是”Oriented FAST“，描述子是”Rotated BRIEF“。
         其实这两种关键点与描述子都是在ORB特征出现之前就已经存在了，ORB特征的作者将二者进行了一定程度的改进，
         并将这两者巧妙地结合在一起，得出一种可以快速提取的特征－－ORB特征。ORB特征在速度方面相较于SIFT、SURF已经有明显的提升的同时，
         保持了特征子具有旋转与尺度不变性。'''
        if not sift_vers:
            keypoints_master, descriptor_master = OrbDescriptors(master_img_name, keypoints_master)
            print('ORB descriptors calculated for master ' + master_img_dirs[-1])
            '''加速：通常在FAST-12算法中，为了更加高效，可以添加一项预测试操作，来快速排除图像中海量的不是角点的像素。
            具体操作为：对于每个像素，直接检测领域圆上的第1、5、9、13个像素的亮度。只有当这四个像素中有三个同时大于Ip+T或者小于Ip-T时，
            当前像素才有可能是一个角点，继续进行更加严谨的判断，否则直接排除。
            优化：通常，原始的FAST角点经常出现“扎堆”的现象。
            所以在第一遍检测之后，还需要用非极大值抑制，在一定范围内仅仅保留响应极大值的角点。这样可以有效缓解角点集中的问题。'''
        else: 
            keypoints_master, descriptor_master = SiftDescriptors(master_img_name, keypoints_master)    
            print('SIFT descriptors calculated for master ' + master_img_dirs[-1])
    '''  局部影像特征的描述与侦测可以帮助辨识物体，SIFT特征是基于物体上的一些局部外观的兴趣点而与影像的大小和旋转无关。
    对于光线、噪声、些微视角改变的容忍度也相当高。基于这些特性，它们是高度显著而且相对容易撷取，在母数庞大的特征数据库中，
    很容易辨识物体而且鲜有误认。使用 SIFT特征描述对于部分物体遮蔽的侦测率也相当高，
    甚至只需要3个以上的SIFT物体特征就足以计算出位置与方位。在现今的电脑硬件速度下和小型的特征数据库条件下，
    辨识速度可接近即时运算。SIFT特征的信息量大，适合在海量数据库中快速准确匹配。
    
      SIFT算法的实质是在不同的尺度空间上查找关键点(特征点)，并计算出关键点的方向。
      SIFT所查找到的关键点是一些十分突出，不会因光照，仿射变换和噪音等因素而变化的点，
      如角点、边缘点、暗区的亮点及亮区的暗点等。'''
    
    '''border mask preparation (for temp texture)'''
    maskForBorderRegion_16UC1 = np.ones((img_master.shape[0], img_master.shape[1]))
    '''ones（）函数返回给定形状和数据类型的新数组，其中元素的值设置为1'''
    maskForBorderRegion_16UC1 = maskForBorderRegion_16UC1.astype(np.uint16)
    '''astype 修改数据类型'''
    
    
    '''perform co-registration for each image每个图片执行此函数'''
    i = 0
    while i < len(image_list):

        slave_img_name = image_list[i-1]
        slave_img_dirs = slave_img_name.split("/")
        '''split()：拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）'''
                
        if master_0 == False:   #matching always to subsequent frame (no master)
            '''skip first image (because usage of subsequent images)跳过第一个图像（因为使用了后续图像） '''
            if i == 0:
                i = i + 1
                continue   
            
            '''detect Harris keypoints in master image'''
            keypoints_master, _ = HarrisCorners(slave_img_name, kp_nbr, False)           
            
            '''calculate ORB or SIFT descriptors in master image主图像中Harris关键点的检测 '''
            if not sift_vers:
                keypoints_master, descriptor_master = OrbDescriptors(slave_img_name, keypoints_master)
                print('ORB descriptors calculated for master ' + slave_img_dirs[-1])
            else: 
                keypoints_master, descriptor_master = SiftDescriptors(slave_img_name, keypoints_master)    
                print('SIFT descriptors calculated for master ' + slave_img_dirs[-1])
        
         
        '''skip first image (because already read as master)'''
        if slave_img_dirs[-1] == master_img_dirs[-1]:
            i = i + 1
            continue
    
        slave_img_name_1 = image_list[i]
        slave_img_dirs_1 = slave_img_name_1.split("/") 

        '''detect Harris keypoints in image to register'''
        keypoints_image, _ = HarrisCorners(slave_img_name_1, kp_nbr, False)
    
        '''calculate ORB or SIFT descriptors in image to register'''
        if not sift_vers:
            keypoints_image, descriptor_image = OrbDescriptors(slave_img_name_1, keypoints_image)
            print('ORB descriptors calculated for image ' + slave_img_dirs_1[-1])
        else:
            keypoints_image, descriptor_image = SiftDescriptors(slave_img_name_1, keypoints_image)
            print('SIFT descriptors calculated for image ' + slave_img_dirs_1[-1])
        
        
        '''match images to master using feature descriptors (SIFT)使用特征描述符（SIFT）将图像与主图像匹配 '''
        if not sift_vers:
            matched_pts_master, matched_pts_img = match_DescriptorsBF(descriptor_master, descriptor_image, keypoints_master, keypoints_image,
                                                                      True,feature_match_twosided)
            matched_pts_master = np.asarray(matched_pts_master, dtype=np.float32)
            matched_pts_img = np.asarray(matched_pts_img, dtype=np.float32)
        else:
            if feature_match_twosided:        
                matched_pts_master, matched_pts_img = match_twosidedSift(descriptor_master, descriptor_image, keypoints_master, keypoints_image, "FLANN")    
            else:
                matchscores = SiftMatchFLANN(descriptor_master, descriptor_image)
                '''使用FLANN方法来适配'''
                matched_pts_master = np.float32([keypoints_master[m[0].queryIdx].pt for m in matchscores]).reshape(-1,2)
                matched_pts_img = np.float32([keypoints_image[m[0].trainIdx].pt for m in matchscores]).reshape(-1,2)
                '''常用于矩阵规格变换，将矩阵转换为特定的行和列的矩阵'''
        
        print('number of matches: ' + str(matched_pts_master.shape[0]))
        
        
        '''calculate homography from matched image points and co-register images with estimated 3x3 transformation
        从匹配的图像点计算单应性，并使用估计的3x3变换对图像进行共配准' '''
        if matched_pts_master.shape[0] > nbr_good_matches:
            '''shape它的功能是读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度。'''
            # Calculate Homography
            H_matrix, _ = cv2.findHomography(matched_pts_img, matched_pts_master, cv2.RANSAC, 3)
            '''cv2.findHomography().如果我们传了两个图像里的点集合，它会找到那个目标的透视转换'''
            
            # Warp source image to destination based on homography
            img_src = cv2.imread(slave_img_name_1)
            img_coregistered = cv2.warpPerspective(img_src, H_matrix, (img_master.shape[1],img_master.shape[0]))
            '''warpPerspective透视变换函数，可保持直线不变形，但是平行线可能不再平行'''
            #cv2.PerspectiveTransform() for points only
            
            #save co-registered image
            cv2.imwrite(os.path.join(directory_out, slave_img_dirs_1[-1])[:-4] + '_coreg.jpg', img_coregistered)
            
            
            '''Mask for border region'''
            currentMask = np.ones((img_master.shape[0], img_master.shape[1]))
            currentMask = currentMask.astype(np.uint16)
            currentMask = cv2.warpPerspective(currentMask, H_matrix, (img_master.shape[1],img_master.shape[0]))
            maskForBorderRegion_16UC1 = maskForBorderRegion_16UC1 * currentMask
            
        i = i + 1   
    
    
    write_file = open(directory_out + 'mask_border.txt', 'wb')        
    writer = csv.writer(write_file, delimiter=",")
    writer.writerows(maskForBorderRegion_16UC1)
    write_file.close()
    

#detect Harris corner features
def HarrisCorners(image_file, kp_nbr=None, visualize=False, img_import=False):
    
    if img_import:
        image_gray = image_file
        '''image_gray string格式'''
    else:
        image = cv2.imread(image_file)
        '''image mat格式'''
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        '''转为gif'''
        '''image_gray mat格式'''
                                                         
    image_gray = np.uint8(image_gray)
    '''将imagegray转为unit8格式
    绘制hsv空间中的2d直方图，必须要将生成的hist数组的格式转换为uint8格式，否则应用cv2.imshow时图像不能显示！---csdn'''
    
    '''detect Harris corners'''
    keypoints = cv2.cornerHarris(image_gray,2,3,0.04)
    '''角点是一类具有特定特征的点，角点也是处在一个无论框框往哪边移动　框框内像素值都会变化很大的情况而定下来的点　可以这么去理解。。。。
    # 输入图像必须是 float32(这里不知道为啥他要变成uint8的数据类型) ,最后一个参数在 0.04 到 0.05 之间
    block_size=2邻域大小（见关于cvCornerEigenValsAndVecs的讨论）。 
    aperture_size=3扩展 Sobel 核的大小（见 cvSobel），该参数表示用来计算差分固定的浮点滤波器的个数  '''
    keypoints = cv2.dilate(keypoints,None)
    '''opencv可以利用cv2.dilate()函数对图片进行膨胀处处理。
    常用的格式如下：
    cv2.dilate(img, kernel, iteration)
    该函数的参数含义：
    img – 目标图片
    kernel – 进行操作的内核，默认为3×3的矩阵
    iterations – 腐蚀次数，默认为1'''
    #reduce keypoints to specific number将关键点减少到特定数量
    thresh_kp_reduce = 0.01
    keypoints_prefilt = keypoints
    keypoints = np.argwhere(keypoints > thresh_kp_reduce * keypoints.max())
    '''返回keypoints中大于0.01*keypoints的最大值的数列索引值'''

    if not kp_nbr == None:
        keypoints_reduced = keypoints
        while len(keypoints_reduced) >= kp_nbr:
            thresh_kp_reduce = thresh_kp_reduce + 0.01
            keypoints_reduced = np.argwhere(keypoints_prefilt > thresh_kp_reduce * keypoints_prefilt.max())
    else:
        keypoints_reduced = keypoints       
        
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints_reduced]
    
    return keypoints, keypoints_reduced #keypoints_reduced for drawing
    '''计算检测特征处的ORB描述符（使用各种特征检测器）'''

#calculate ORB descriptors at detected features (using various feature detectors)
def OrbDescriptors(image_file, keypoints):
    image = cv2.imread(image_file)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    '''【将图像灰度化】'''
    image_gray = np.uint8(image_gray) 
    
    '''perform ORB'''
    if "3." in cv2.__version__:
        orb = cv2.ORB_create()
    else:
        orb = cv2.ORB()
    keypoints, descriptors = orb.compute(image_gray, keypoints)
    
    return keypoints, descriptors


#calculate SIFT descriptors at detected features (using various feature detectors)
def SiftDescriptors(image_file, keypoints):
    image = cv2.imread(image_file)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                                             
    image_gray = np.uint8(image_gray) 
        
    '''perform SIFT'''
    if "3." in cv2.__version__:
        siftCV2 = cv2.xfeatures2d.SIFT_create()
        #siftCV2 = cv2.SIFT_create()
    else:
        siftCV2 = cv2.SIFT()
    keypoints, descriptors = siftCV2.compute(image_gray, keypoints)
    descriptors = descriptors.astype(np.uint8)
    
    return keypoints, descriptors


#match SIFT features using SIFT matching
#source code from Jan Erik Solem
def match_SIFT(desc1, desc2):
    '''For each descriptor in the first image, select its match in the second image.
    input: desc1 (descriptors for the first image),
    desc2 (same for the second image).'''
    '''array(object[, dtype, copy, order, subok, ndmin])	创建一个数组。'''
    '''linalg.norm(x[, ord, axis, keepdims])	矩阵或向量范数。'''
    desc1 = np.array([d/plt.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/plt.linalg.norm(d) for d in desc2])
    
    dist_ratio = 0.6
    desc1_size = desc1.shape
    
    matchscores = np.zeros((desc1_size[0],1),'int')
    desc2t = desc2.T #precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:], desc2t)   #vector of dot products
        dotprods = 0.9999*dotprods
        # inverse cosine and sort, return index for features in second Image
        indx = np.argsort(plt.arccos(dotprods))
        
        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if plt.arccos(dotprods)[indx[0]] < dist_ratio * plt.arccos(dotprods)[indx[1]]:
            matchscores[i] = np.int(indx[0])
            
    return matchscores


#match SIFT features using SIFT matching and perform two-sided
#source code from Jan Erik Solem
def match_twosided_SIFT(desc1, desc2):
    '''Two-sided symmetric version of match().'''
    
    matches_12 = match_SIFT(desc1,desc2)
    matches_21 = match_SIFT(desc2,desc1)
    
    ndx_12 = matches_12.nonzero()[0]
    
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
            
    return matches_12


#match SIFT image features using FLANN matching
#source code from Jan Erik Solem    
def SiftMatchFLANN(des1,des2):
    max_dist = 0
    min_dist = 100
    
    # FLANN parameters   
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    '''FLANN (Fast_Library_for_Approximate_Nearest_Neighbors)快速最近邻搜索包。
    它是一个对大数据集和高维特征进行最近邻搜索的算法的集合,而且这些算法都已经被优化过了。在面对大数据集时它的效果要好于 BFMatcher。'''
    
    if des1.dtype != np.float32:
        des1 = des1.astype(np.float32)
    if des2.dtype != np.float32:
        des2 = des2.astype(np.float32)
    
    matches = flann.knnMatch(des1,des2,k=2)
       
    # ratio test as per Lowe's paper
    for m,n in matches:
        if min_dist > n.distance:
            min_dist = n.distance
        if max_dist < n.distance:
            max_dist = n.distance
    
    good = []
    for m,n in matches:
        #if m.distance < 0.75*n.distance:
        if m.distance <= 3*min_dist:
            good.append([m])
            '''去除错误匹配点'''
    
    return good


#match SIFT image features using FLANN matching and perform two-sided matching
#source code from Jan Erik Solem
def match_twosidedSift(desc1, desc2, kp1, kp2, match_Variant="FLANN"):
    '''Two-sided symmetric version of match().'''    
    if match_Variant == "FLANN":
        matches_12 = SiftMatchFLANN(desc1,desc2)
        matches_21 = SiftMatchFLANN(desc2,desc1)
    elif match_Variant == "BF":
        matches_12 = SiftMatchBF(desc1,desc2)
        matches_21 = SiftMatchBF(desc2,desc1)

    pts1 = []
    pts2 = []
    for m in matches_12:
        pts1.append(kp1[m[0].queryIdx].pt)
        pts2.append(kp2[m[0].trainIdx].pt)

    pts1_b = []
    pts2_b = []    
    for m in matches_21:
        pts2_b.append(kp1[m[0].trainIdx].pt)
        pts1_b.append(kp2[m[0].queryIdx].pt)
    
    pts1_arr = np.asarray(pts1)
    pts2_arr = np.asarray(pts2)
    pts_12 = np.hstack((pts1_arr, pts2_arr))
    pts1_arr_b = np.asarray(pts1_b)
    pts2_arr_b = np.asarray(pts2_b)        
    pts_21 = np.hstack((pts1_arr_b, pts2_arr_b))
       
    pts1_ts = []
    pts2_ts = []        
    for pts in pts_12:
        pts_comp = np.asarray(pts, dtype = np.int)
        for pts_b in pts_21:
            pts_b_comp = np.asarray(pts_b, dtype = np.int)
            if ((int(pts_comp[0]) == int(pts_b_comp[2])) and (int(pts_comp[1]) == int(pts_b_comp[3]))
                and (int(pts_comp[2]) == int(pts_b_comp[0])) and (int(pts_comp[3]) == int(pts_b_comp[1]))):
                pts1_ts.append(pts[0:2].tolist())
                pts2_ts.append(pts[2:4].tolist())                
                break
    
    pts1 = np.asarray(pts1_ts, dtype=np.float32)
    pts2 = np.asarray(pts2_ts, dtype=np.float32)
    
    #print('Matches twosided calculated')
        
    return pts1, pts2


#match STAR image features using bruce force matching
#source code from Jan Erik Solem
def match_DescriptorsBF(des1,des2,kp1,kp2,ratio_test=True,twosided=True):
    '''Match STAR descriptors between two images'''
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    '''首先在第一幅图像中选取一个关键点然后依次与第二幅图像的每个关键点进行（描述符）距离测试，最后返回距离最近的关键点
    BFMatcher对象具有两个方法，BFMatcher.match()和BFMatcher.knnMatch()。第一个方法会返回最佳匹配'''
    
    # Match descriptors.
    matches = bf.match(des1,des2)                    
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)     
    '''根据距离排序；'''
    pts1 = []
    pts2 = []        
    
    if ratio_test: 
        # ratio test as per Lowe's paper
        good = []
        for m in matches:
            '''从matches里一次选出数据'''
            if m.distance < 100:
                good.append(m)
                '''相当于good继承了来自于matches-distance小于100的值'''
                pts2.append(kp2[m.trainIdx].pt)
                '''append函数是在列表末尾添加新的对象'''
                pts1.append(kp1[m.queryIdx].pt)
    else:
        for m in matches:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            
    if twosided:
        pts1_b = []
        pts2_b = []
        
        matches_back = bf.match(des2,des1)
        for m in matches_back:
            pts2_b.append(kp1[m.trainIdx].pt)
            pts1_b.append(kp2[m.queryIdx].pt)
        
        pts1_arr = np.asarray(pts1)
        '''numpy中的array()和asarray()方法非常类似，他们都可以接受列表或数组类型的数据作为参数。当他们的参数是列表型数据时，二者没有区别；
        当他们的参数是数组类型时，np.array()会返回参数数组的一个副本(copy,2者值一样但指向不同的内存),np.asarray()会返回参数数组的一个视图(2者指向同一块内存).
        
        副本会新开辟一块内存，对于大数组来说，会存在大量的复制操作，速度更慢且不节约内存；视图相当于新增加了一个指向当前内存的引用，不存在复制操作，
        速度更快且节约内存，但是注意通过其中的一个引用修改数据，其他引用的数据也会跟着变，因为他们指向同一块内存区域。'''
        pts2_arr = np.asarray(pts2)
        pts_12 = np.hstack((pts1_arr, pts2_arr))
        pts1_arr_b = np.asarray(pts1_b)
        pts2_arr_b = np.asarray(pts2_b)        
        pts_21 = np.hstack((pts1_arr_b, pts2_arr_b))
        '''hstack（）：堆栈数组水平顺序（列）。'''
       
        
        pts1_ts = []
        pts2_ts = []        
        for pts in pts_12:
            pts_comp = np.asarray(pts, dtype = np.int)
            for pts_b in pts_21:
                pts_b_comp = np.asarray(pts_b, dtype = np.int)
                if ((int(pts_comp[0]) == int(pts_b_comp[2])) and (int(pts_comp[1]) == int(pts_b_comp[3]))
                    and (int(pts_comp[2]) == int(pts_b_comp[0])) and (int(pts_comp[3]) == int(pts_b_comp[1]))):
                    pts1_ts.append(pts[0:2].tolist())
                    pts2_ts.append(pts[2:4].tolist())
                    '''tolist把numpy数组转化为列表
                    1、一个参数：a[i]

                    如 [2]，将返回与该索引相对应的单个元素。
                    2、两个参数：b=a[i:j]

                    b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的list对象
                    i缺省时默认为0,即 a[:n] 代表列表中的第一项到第n项，相当于 a[0:n]
                    j缺省时默认为len(alist),即a[m:] 代表列表中的第m+1项到最后一项，相当于a[m:5]
                    当i,j都缺省时，a[:]就相当于完整复制a
                    '''
                    
                    break
        
        pts1 = pts1_ts
        pts2 = pts2_ts      
        
        #print('Matches calculated')
            
    return pts1, pts2


#match SIFT image features using bruce force matching    
#source code from Jan Erik Solem
def SiftMatchBF(des1, des2):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    return good
    
    
def accuracy_coregistration(image_list_coreg, check_pts_img, template_size, output_dir):
    
    write_all = open(os.path.join(output_dir, 'stat_img_coreg_all.txt'), 'wb')
    writer_all = csv.writer(write_all, delimiter="\t")
    write_stat = open(os.path.join(output_dir, 'stat_img_coreg.txt'), 'wb')
    writer_stat = csv.writer(write_stat, delimiter="\t")
    
    first_image = True

    distance_matched_points_for_stat = np.ones((check_pts_img.shape[0],1))
    image_list_coreg_dirs = image_list_coreg[0].split("/")   
    frame_ids = image_list_coreg_dirs[-1]
    frame_ids = np.asarray(frame_ids)
    frame_ids = frame_ids.reshape(1,1)
    
    for image in image_list_coreg:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        if first_image:
            '''template matching --> create templates from first image pairs with approximate position due to img point information'''
            #include template size to approximate template position table for subsequent template extraction
            template_approx_pos = check_pts_img
            
            #calculate template with corresponding template size
            template_img, _ = getTemplateAtImgpoint(img, template_approx_pos, template_size, template_size)
            
            first_image = False 
    
        else:  
            '''approximation of template position for subsequent images'''
            master_pts = check_pts_img
            
            # Apply template Matching --> print with img with new pts
            template_approx_size = check_pts_img
            approx_pos_template, anchor_pts = getTemplateAtImgpoint(img, template_approx_size, template_size*3, template_size*3)
            check_pts_img = performTemplateMatch(approx_pos_template, template_img, anchor_pts)
            
            dist_check_master = np.sqrt(np.square(master_pts[:,0] - check_pts_img[:,0]) + np.square(master_pts[:,1] - check_pts_img[:,1]))
            frame_id_dirs = image.split("/")   
            frame_id = frame_id_dirs[-1]

            if frame_id == 'output_033.jpg':
                print()
            frame_id = np.asarray(frame_id)
            frame_id = frame_id.reshape(1,1)

            dist_check_master = dist_check_master.reshape(dist_check_master.shape[0],1)
            
            distance_matched_points_for_stat = np.hstack((distance_matched_points_for_stat, dist_check_master))
            frame_ids = np.hstack((frame_ids, frame_id))   
    
    distance_matched_points = distance_matched_points_for_stat[:,1:distance_matched_points_for_stat.shape[1]]
    
    distance_matched_points_for_stat = np.vstack((frame_ids, distance_matched_points_for_stat))
    distance_matched_points_for_stat = distance_matched_points_for_stat[:,1:distance_matched_points_for_stat.shape[1]]
    
    
    '''calculate statistics'''
    point_nbr = []
    for i in range(distance_matched_points.shape[0]):
        point_nbr.append(i+1)
    writer_stat.writerow(['id', point_nbr])    
    
    average_dist_per_point = np.mean(distance_matched_points, axis=1)
    writer_stat.writerow(['mean', average_dist_per_point])
    std_dist_per_point = np.std(distance_matched_points, axis=1)
    writer_stat.writerow(['std', std_dist_per_point])    
    max_dist_per_point = np.max(distance_matched_points, axis=1)
    writer_stat.writerow(['max', max_dist_per_point])    
    min_dist_per_point = np.min(distance_matched_points, axis=1)
    writer_stat.writerow(['min', min_dist_per_point])    
   
    writer_all.writerows(distance_matched_points_for_stat)

        
    '''draw errorbar'''
    matplotlib.rcParams.update({'font.family': 'serif',
                                'font.size' : 12,})    
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()  
    ax.set_xlabel('point ID')
    ax.set_ylabel('point deviations [pixel]')
    ax.set_xlim(0, len(point_nbr)+1)
    ax.set_xticks(np.arange(0, len(point_nbr)+1, 1))
    distance_matched_points = distance_matched_points.T    
    ax.errorbar(point_nbr, average_dist_per_point, xerr=0, yerr=std_dist_per_point, 
                 fmt='o', ecolor='g')

    plt.savefig(os.path.join(output_dir, 'errorbar_checkPts.jpg'),
                bbox_inches="tight", dpi=600)
    
    
    '''draw check point locations'''
    drawF.draw_points_onto_image(cv2.imread(image_list_coreg[0], 0), 
                                 check_pts_img, point_nbr, 5, 15)
    plt.savefig(os.path.join(output_dir, 'accuracy_coreg_checkPts_location.jpg'), dpi=600)
    
    
#define template at image point position (of corresponding GCP)
def getTemplateAtImgpoint(img, img_pts, template_width=10, template_height=10):
#consideration that row is y and column is x   
#careful that template extent even to symmetric size around point of interest 
    
    template_img = []
    anchor_pts = []
    for pt in img_pts:
        if img_pts.shape[1] > 2:
            template_width_for_cut_left = pt[2]/2
            template_width_for_cut_right = pt[2]/2 + 1
        elif template_width > 0:
            template_width_for_cut_left = template_width/2
            template_width_for_cut_right = template_width/2 + 1
        else:
            print('missing template size assignment')
        
        if img_pts.shape[1] > 2:
            template_height_for_cut_lower = pt[3]/2
            template_height_for_cut_upper = pt[3]/2 + 1
        elif template_height > 0:
            template_height_for_cut_lower = template_height/2
            template_height_for_cut_upper = template_height/2 + 1
        else:
            print('missing template size assignment')
        
        cut_anchor_x = pt[0] - template_width_for_cut_left
        cut_anchor_y = pt[1] - template_height_for_cut_lower
        
        #consideration of reaching of image boarders (cutting of templates)
        if pt[1] + template_height_for_cut_upper > img.shape[0]:
            template_height_for_cut_upper = np.int(img.shape[0] - pt[1])
        if pt[1] - template_height_for_cut_lower < 0:
            template_height_for_cut_lower = np.int(pt[1])
            cut_anchor_y = 0
        if pt[0] + template_width_for_cut_right > img.shape[1]:
            template_width_for_cut_right = np.int(img.shape[1] - pt[0])
        if pt[0] - template_width_for_cut_left < 0:
            template_width_for_cut_left = np.int(pt[0])
            cut_anchor_x = 0
        
        template = img[np.int(pt[1]-template_height_for_cut_lower) : np.int(pt[1]+template_height_for_cut_upper), 
                       np.int(pt[0]-template_width_for_cut_left) : np.int(pt[0]+template_width_for_cut_right)]
        
        #template_img = np.dstack((template_img, template))
        template_img.append(template)
        
        anchor_pts.append([cut_anchor_x, cut_anchor_y])
        
    anchor_pts = np.asarray(anchor_pts, dtype=np.float32) 
    #template_img = np.delete(template_img, 0, axis=2) 
    
    return template_img, anchor_pts #anchor_pts defines position of lower left of template in image


#template matching for automatic detection of image coordinates of GCPs
def performTemplateMatch(img_extracts, template_img, anchor_pts, plot_results=False):
    new_img_pts = []
    template_nbr = 0
    
    count_pts = 0
    while template_nbr < len(template_img):
        template_array = np.asarray(template_img[template_nbr])
        if (type(img_extracts) is list and len(img_extracts) > 1) or (type(img_extracts) is tuple and len(img_extracts.shape) > 2):      
            img_extract = img_extracts[template_nbr]
        else:
            img_extract = img_extracts
        res = cv2.matchTemplate(img_extract, template_array, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #min_loc for TM_SQDIFF
        match_position_x = max_loc[0] + template_array.shape[1]/2
        match_position_y = max_loc[1] + template_array.shape[0]/2
        del min_val, min_loc
        
        if max_val > 0.9:
            new_img_pts.append([match_position_x + anchor_pts[template_nbr,0], 
                                match_position_y + anchor_pts[template_nbr,1]])
            count_pts = count_pts + 1
             
        template_nbr = template_nbr + 1

        if plot_results:    
            plt.subplot(131),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.plot(match_position_x-template_array.shape[1]/2, match_position_y-template_array.shape[0]/2, "r.", markersize=10)
            plt.subplot(132),plt.imshow(img_extract,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])    
            plt.plot(match_position_x, match_position_y, "r.", markersize=10)
            plt.subplot(133),plt.imshow(template_array,cmap = 'gray')
            plt.title('Template'), plt.xticks([]), plt.yticks([])
            plt.show()
        
    new_img_pts = np.asarray(new_img_pts, dtype=np.float32)
    new_img_pts = new_img_pts.reshape(count_pts, 2)
         
    return new_img_pts
