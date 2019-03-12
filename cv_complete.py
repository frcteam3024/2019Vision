# On startup, in command prompt run source .profile -> workon cv -> ./startUSBtest

from networktables import NetworkTables
import numpy as np
import cv2
import time
import math


def nothing(bep):
    bep+1


cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 180)
cap.set(5, 15)

cv2.namedWindow("Sliders")
cv2.createTrackbar("Minimum Hue", "Sliders", 40, 179, nothing)
cv2.createTrackbar("Minimum Saturation", "Sliders", 255, 255, nothing)
cv2.createTrackbar("Minimum Value", "Sliders", 48, 255, nothing)
cv2.createTrackbar("Maximum Hue", "Sliders", 90, 179, nothing)
cv2.createTrackbar("Maximum Saturation", "Sliders", 255, 255, nothing)
cv2.createTrackbar("Maximum Value", "Sliders", 145, 255, nothing)
# mask2 = np.zeros([320,160,3], dtype=np.int8, order='C')


def main_vision_loop():
    count = 0
    targets = []
    
    while len(targets) < 2:
        count = count+1
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_low = cv2.getTrackbarPos("Minimum Hue", "Sliders")
        sat_low = cv2.getTrackbarPos("Minimum Saturation", "Sliders")
        val_low = cv2.getTrackbarPos("Minimum Value", "Sliders")
        hue_high = cv2.getTrackbarPos("Maximum Hue", "Sliders")
        sat_high = cv2.getTrackbarPos("Maximum Saturation", "Sliders")
        val_high = cv2.getTrackbarPos("Maximum Value", "Sliders")    
        lower_bound = np.array([hue_low,sat_low,val_low])
        upper_bound = np.array([hue_high,sat_high,val_high])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask2 = frame - frame
        mask3 = frame - frame
        # Display the resulting frame
        cv2.imshow('frame',hsv)
        im2, contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        # for contour in contours:
        cv2.drawContours(mask2, contours, -1, (0, 0, 255), 3,)
        # print("length of contours", len(contours))
        # print("first contour", contours[0])
        area = 0
        big_contours_l = list()
        big_contours_r = list()
        bad_contours = list()

        target_left = {}
        target_right = {}
        target_OOR = {}
    
        for cnt in contours:
            center = (0,0)
            conHeight = 0
            conWidth = 0
            conAngle = 0
            conDirect = 'NULL'
            area = cv2.contourArea(cnt)
            if area > 80:
                rect_page = cv2.minAreaRect(cnt)
                if rect_page[2] < -45 and rect_page[2] > -90:
                    box_page = np.int0(cv2.boxPoints(rect_page))
                    big_contours_l.append(box_page)
                    conDirect = 'left'
                    cv2.drawContours(mask3,big_contours_l, -1, (0, 255, 0),3,)
                elif rect_page[2] < 0 and rect_page[2] > -45:
                    box_page = np.int0(cv2.boxPoints(rect_page))
                    big_contours_r.append(box_page)
                    conDirect = 'right'
                    cv2.drawContours(mask3,big_contours_r, -1, (255, 0, 0),3,)
                else:
                    box_page = np.int0(cv2.boxPoints(rect_page))
                    bad_contours.append(box_page)
                    conDirect = 'OOR'
                    cv2.drawContours(mask3,bad_contours, -1, (0, 0, 255),3,)
                if (count%20 == 0):
                    center = (round(rect_page[0][0], 2), round(rect_page[0][1], 2))
                    conHeight = rect_page[1][0]
                    conWidth = rect_page[1][1]
                    if  conDirect == 'left':
                        conAngle = rect_page[2]+90
                    else:
                        conAngle = rect_page[2]
                    targetInfo = {
                                'center': center[0],
                                'height' : round(conHeight, 2),
                                'width' : round(conWidth, 2),
                                'area' : area,
                                'angle' : round(conAngle, 2),
                                'direction' : conDirect
                                }
                    if targetInfo['direction'] == 'left':
                        target_left = targetInfo
                    elif targetInfo['direction'] == 'right':
                        target_right = targetInfo
                    else:
                        target_OOR = targetInfo
        
        #if (count%60 == 0):
            #print("length of bigContours", len(bigContours))
        #print("first bigContour", bigContours[0])
        
        if target_left:
            if target_left['height']/target_left['width'] < 3:
                targets.append(target_left)
        if target_right:
            height = target_right['width']
            target_right['width'] = target_right['height']
            target_right['height'] = height
            if target_right['height']/target_right['width'] < 3:
                targets.append(target_right)


        cv2.imshow('mask',mask)
        cv2.imshow('contours',mask2)
        cv2.imshow('filtered',mask3)
        #cv2.imshow('contours',contours)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Hue range " +str(hue_low)+" to " +str(hue_high))
            print("Saturation range " +str(sat_low)+" to " +str(sat_high))
            print("Value range " +str(val_low)+" to " +str(val_high))
            break
        
    return(targets)


def calc_distance(heights):
    distances = []
    for h in heights:
        camera_h_real = t_height / h * pixels_y
        d = camera_h_real / math.tan(y_angle)
        distances.append(d)
    return sum(distances)/len(distances)


def calc_strafe(vision_data):
    t1 = vision_data[0]
    t2 = vision_data[1]
    width = t2['center'] - t1['center']
    center = t1['center'] + .5*width
    offset = 0 #12 * (width/t_width)
    ratio = (center + offset)/pixels_x
    angle = (x_angle*ratio) - (.5*x_angle)
    return angle*-1


def calc_rotation(heights_array, d_mid, theta):
    d1 = calc_distance([heights_array[0]])
    d2 = calc_distance([heights_array[1]])
    if d1 == d2:
        return theta
    elif d1>d2:
        cos_a = (d1**2 + t_width**2 - d2**2)/(2*d1*t_width)
    else:
        cos_a = (d2**2 + t_width**2 - d1**2)/(2*d2*t_width)

    d_diff = d2 - d1
    a_sqrd = d_diff**2 + t_width**2 - 2*abs(d_diff)*t_width*cos_a
    length_a = math.sqrt(abs(a_sqrd))
    angle_b = (180/math.pi)*math.asin(abs(d_diff)*math.sin(a)/length_a)  # law of sines
    # angle_b is the strafe minus rotational angles
    z = theta - angle_b
    return z


y_angle = 35 * math.pi/180
x_angle = 70.4
t_width = 13.42
t_height = 5.82
t_angle = 14.5 * math.pi/180
pixels_x = 320
pixels_y = 180

NetworkTables.initialize(server='10.30.24.2')
table = NetworkTables.getTable('Pi')

while True:
    vision_array = main_vision_loop()

    if len(vision_array) == 2:
        heights = []
        for target in vision_array:
            a = target['width']*math.sin(t_angle)
            b = target['height']*math.cos(t_angle)
            heights.append(a+b)
        
        strafe = calc_strafe(vision_array)

        # check to see if target is too close: if so, return d=0 and use IR value in LV code
        check_1 = vision_array[0]['center'] - vision_array[0]['width'] <= 0
        check_2 = vision_array[1]['center'] + vision_array[1]['width'] >= pixels_x

        if check_1 or check_2:
            distance = 0
            z_rotation = 0
        else:
            distance = calc_distance(heights)
            z_rotation = calc_rotation(heights, distance, strafe)

    else:
        distance = 0
        strafe = 0
        z_rotation = 0
        
    #print('Distance: ', distance)
    #print('Strafe: ', strafe)
    print('Z angle: ', z_rotation)
    table.putNumberArray('VisionData', [distance, strafe, z_rotation])


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
