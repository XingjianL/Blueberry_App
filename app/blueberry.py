
import numpy as np
import cv2

SCALING_FACTOR = 0.5

def findRealBlueBerry(circles, colors):
    # stack together coord of circle center, and the color of circle center
    embedded_circle = np.hstack((circles[:,:2], np.where(colors<90, 0, 255)))
    # calculate the mean of (x, y, blue, green, red)
    mean, std = np.mean(embedded_circle, axis = 0), np.std(embedded_circle, axis=0)
    # check the outliers of each (x, y, b, g, r), coord uses 2.5 * std, colors uses 1.5 * std
    if (any(std[2:] > 100)):
        #print(np.abs(embedded_circle - mean))
        element_is_outlier = np.abs(embedded_circle - mean) > ([3, 3, 2.5, 2.5, 2.5]*std)
    else:
        element_is_outlier = np.abs(embedded_circle - mean) > ([3, 3, 2.5, 2.5, 2.5]*std)
    # compile for each circle
    circle_is_outlier = np.any(element_is_outlier, axis = 1)
    #print(mean + 1.5*std)
    #print(element_is_outlier)
    real_circles = np.array([])
    for i,is_outlier in enumerate(circle_is_outlier):
        if not is_outlier:
            real_circles = np.concatenate((real_circles, circles[i]))
    return real_circles.reshape(-1,3)

def findCircleAverageColor(img, circles):
    colors = []
    for i in circles:
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.circle(mask,tuple(i[:2].astype(int)), i[2].astype(int), 1, thickness=-1)
        this_circle = cv2.bitwise_and(img, img, mask=mask)
        color = (np.sum(this_circle.reshape(-1,3), axis=0) / (np.pi*np.power(i[2],2)))
        colors.append(color)
    return np.array(colors).astype(int)

# otsu threshold https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
def otsuThresholding(gray):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(gaussian_filtered.shape)
    ret, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary_img
def adaptiveThreshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
## region growing - watershed https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
def watershed(original, berries, berries_center):
    #bg = cv2.dilate(berries, (3,3), iterations=2)
    #fg = cv2.erode(berries, (3,3), iterations=2)
    #dist_transform = cv2.distanceTransform(berries, cv2.DIST_L2, 5)
    #ret, fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    fg = np.uint8(berries_center)
    unknown = cv2.subtract(berries, fg)

    ret, markers = cv2.connectedComponents(fg)
    markers = markers+1
    markers[unknown==1] = 0

    markers = cv2.watershed(original, markers)
    #original[markers == -1] = [0,0,255]
    markers = np.where(markers == -1, 1, markers)
    return markers, unknown, fg, berries#, dist_transform 


def colorEmphasis(img,a1,a2,a3):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    c1 = img[:,:,0]
    c2 = img[:,:,1]
    c3 = img[:,:,2]
    output = (a1*c1 + a2*c2 + a3*c3)
    output = np.where(np.abs(output)>1, output, .1)
    #print(np.max(output), np.min(output))
    return output

# https://www-sciencedirect-com.prox.lib.ncsu.edu/science/article/pii/S0168169916301557?via%3Dihub
def HDI(img):
    nominator = colorEmphasis(img,1,-1,0)
    denominator = colorEmphasis(img,1,1,0)
    output = np.uint8(128*(nominator/denominator + 1))
    return output

def circularity(img, num_circles):
    output = np.zeros(num_circles)
    
    for i in range(num_circles):
        x_coord, y_coord = np.where(img == i+2)
        circle_area = len(x_coord)
        circle_center = [np.mean(x_coord), np.mean(y_coord)]
        circle_radius = np.sqrt(circle_area/np.pi)

        circle = [int(circle_center[1]),int(circle_center[0]),int(circle_radius)]
        
        circle_img = np.where(img == i+2, 255, 0)
        contours, _ = cv2.findContours(circle_img.astype(np.uint8),1,2)
        enclose_cirlce_center, enclose_cirlce_radius = cv2.minEnclosingCircle(contours[0])
        #enclose_circle = [enclose_cirlce_center[0], enclose_cirlce_center[1], enclose_cirlce_radius]
        #plt.imshow(blob(circle_img))
        circle_img = np.stack((circle_img,circle_img,circle_img),axis=-1)

        color_ratio = findCircleAverageColor(circle_img, np.array([circle]))[0][0] / 255
        radius_ratio = circle_radius / enclose_cirlce_radius
        #enclosed_color_ratio = findCircleAverageColor(circle_img, np.array([enclose_circle]))[0][0] / 255
        #print(i+2, color_ratio, radius_ratio, (color_ratio-radius_ratio)/(color_ratio+radius_ratio))
        #print(color_ratio < 0.85, radius_ratio < 0.7, enclosed_color_ratio < .7)
        #print('')
        
        
        #plt.show()
        
        if color_ratio < 0.85 or radius_ratio < 0.7:
            output[i] = int(i+2)
    return output

class AppProcess():
    SCALING_FACTOR = 1
    def process_image(frame, scaling = SCALING_FACTOR, min_r = 20, max_r = 55):
        # resize
        width = int(frame.shape[1] * scaling)
        height = int(frame.shape[0] * scaling)
        
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        # find general circles
        dist_between = int(2.5 * min_r)
        circles = cv2.HoughCircles(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, dist_between,
                               param1=100, param2=40, # 30, 25, 13
                               minRadius=int(min_r), maxRadius=int(max_r))
        if (circles is None):
            return None
        # get average color of each circle
        center_colors = findCircleAverageColor(frame, circles[0,:])
        print("process_image(cc): ", center_colors)
        # filter the outliers (location and color)
        real_circles = findRealBlueBerry(circles[0,:], center_colors)
        print("process_image(rc): ", real_circles)
        # create image masks for visual display and watershed (foreground and background)
        raw_circle = frame.copy()
        raw_circle_mask = np.zeros(raw_circle.shape[:2], dtype="uint8")
        blueberry_center = raw_circle_mask.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # display all hough circles (unfiltered)
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(raw_circle, center, 4, (0,255,0), 2)
                # circle outline
                radius = i[2]
                cv2.circle(raw_circle, center, radius, (0,255,0), 2)

            # all blueberry circles (filtered)
            real_circles = np.uint16(np.around(real_circles))
            for i in real_circles[:]:
                center = (i[0], i[1])
                
                # circle center
                cv2.circle(raw_circle, center, 1, (0,255,255), 2)
                # circle outline
                radius = i[2]
                cv2.circle(raw_circle, center, radius, (0,255,255), 2)

                # draw the foreground (blueberry_center)/background (raw_circle_mask)
                raw_circle_mask = cv2.circle(raw_circle_mask, center, radius+max_r, 1, thickness=-1)
                blueberry_center = cv2.circle(blueberry_center, center, min_r, 1, thickness=-1)
        
        # crop the image around the blueberries
        y_bounds = [np.min(np.nonzero(raw_circle_mask)[0]), np.max(np.nonzero(raw_circle_mask)[0])]
        x_bounds = [np.min(np.nonzero(raw_circle_mask)[1]), np.max(np.nonzero(raw_circle_mask)[1])]

        cv2.rectangle(raw_circle,(x_bounds[0],y_bounds[0]),(x_bounds[1],y_bounds[1]),(255,0,0),5)
        print("process_image(bounds): Done")
        focused_frame = frame[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1],:]
        focused_mask = raw_circle_mask[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]]
        focused_center = blueberry_center[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]]

        # watershed
        blueberry_stage_4 = watershed(focused_frame,focused_mask,focused_center)
        print("process_image(watershed): Done")
        # filter the non-circles from the watershed
        non_circles = circularity(blueberry_stage_4[0], np.max(blueberry_stage_4[0])-1)

        filtered_final_stage = blueberry_stage_4[0].copy()
        filtered_final_stage[np.isin(filtered_final_stage,non_circles)] = 1
        num_filtered = np.count_nonzero(non_circles)

        # calculate area (both unfiltered and filtered watershed)
        # average area = (total_area - background_area)/num_blueberries
        #       unfiltered
        focus_area = (y_bounds[1]-y_bounds[0]) * (x_bounds[1] - x_bounds[0])
        background_area = np.count_nonzero(blueberry_stage_4[0] == 1)
        final = np.uint8(np.where(blueberry_stage_4[0] == 1, 0, 255))
        final_inv = np.uint8(np.where(blueberry_stage_4[0] == 1, 255, 0))
        area = (focus_area - background_area)/(np.max(blueberry_stage_4[0])-1)
        #       filtered
        background_area_f = np.count_nonzero(filtered_final_stage == 1)
        final_f = np.uint8(np.where(filtered_final_stage == 1, 0, 255))
        final_inv_f = np.uint8(np.where(filtered_final_stage == 1, 255, 0))
        filtered_area = (focus_area - background_area_f)/(np.max(blueberry_stage_4[0])-1-num_filtered)
        #       how much average area has changed
        percent_area_change = 100*((filtered_area-area)/area)
        print("\tavg area: ", area, "\n\tfiltered avg: ", filtered_area, "\n\t# filtered", num_filtered, "\n\t% change: ", percent_area_change)


        final = cv2.bitwise_and(focused_frame,focused_frame,mask=final)
        final_f = cv2.bitwise_and(focused_frame,focused_frame,mask=final_f)
        final_inv = cv2.bitwise_and(focused_frame,focused_frame,mask=final_inv)
        final_inv_f = cv2.bitwise_and(focused_frame,focused_frame,mask=final_inv_f)

        cv2.putText(raw_circle,"hough: {count:d}".format(count = len(real_circles)),(0,50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), thickness=2)
        cv2.putText(raw_circle,"watershed: {count:d}".format(count = np.max(blueberry_stage_4[0])-1),(0,100), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), thickness=2)
        cv2.putText(raw_circle,"area_nofilter: {count:.2f}".format(count = area),(0,150), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), thickness=2)
        cv2.putText(raw_circle,"area_withfilter: {count:.2f}".format(count = filtered_area),(0,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), thickness=2)
        cv2.putText(raw_circle,"num_filtered: {count:d}".format(count = num_filtered),(0,250), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), thickness=2)
        cv2.putText(raw_circle,"%diff: {count:.2f}".format(count = percent_area_change),(0,300), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), thickness=2)


        output_imgs = [raw_circle, final, final_inv, final_f, final_inv_f]
        counts = [np.max(blueberry_stage_4[0])-1, np.max(blueberry_stage_4[0])-1-num_filtered]
        average_area = [area, filtered_area]
        watersheds = [blueberry_stage_4[0],filtered_final_stage,blueberry_stage_4[0]-filtered_final_stage+1]
        cv2.normalize(blueberry_stage_4[0],watersheds[0],0,255,cv2.NORM_MINMAX)
        cv2.normalize(filtered_final_stage,watersheds[1],0,255,cv2.NORM_MINMAX)
        avg_radius = np.mean(circles[0,:,2])
        misc_output = [avg_radius,percent_area_change]
        print("process_image: finished")
        return output_imgs, counts, average_area, watersheds, misc_output
    
    def checkerboard(frame, pattern = (4,4), target_radius = 50):
        output = frame.copy()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        print('checkerboard: ', ret)
        if ret:
            
            corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            output = cv2.drawChessboardCorners(output,pattern,corners2, ret)
            w = 1
            h = pattern[0]
            print(corners2[0])
            print(corners2[0], corners2[w], corners2[h])
            p1 = corners2[0][0]
            p2 = corners2[w][0]
            p3 = corners2[h][0]
            vecs = [[p1[0]-p2[0], p1[1] - p2[1]],
                    [p1[0]-p3[0], p1[1] - p3[1]],
                    [p2[0]-p3[0], p2[1] - p3[1]]]
            print(vecs)
            dists = [np.sqrt(vecs[0][0]**2 + vecs[0][1]**2),
                     np.sqrt(vecs[1][0]**2 + vecs[1][1]**2),
                     np.sqrt(vecs[2][0]**2 + vecs[2][1]**2)]
            print(dists[np.argmax(dists)])
            print(dists)
            resize_factor = target_radius / np.max(dists)
            #retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize)
            return output, resize_factor
        else:
            return output, None
