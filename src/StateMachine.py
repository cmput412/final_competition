#!/usr/bin/env python
import rospy, cv2, cv_bridge, numpy, smach, smach_ros, time, math, actionlib, tf, imutils
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, Joy, Image
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import Led, BumperEvent, Sound
from ar_track_alvar_msgs.msg import AlvarMarkers
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from smach_ros import SimpleActionState
from skimage import filters, morphology, measure
from shapedetector import ShapeDetector
from math import copysign
from smach import State, StateMachine
from ar_track_alvar_msgs.msg import AlvarMarkers


numpy.set_printoptions(threshold=numpy.nan)
counter = 0
gshape = "triangle"
random = 'two'

BoxGoal = None
BoxSpot = None

BoxAR = [1]
OtherAR = [3,2]

objectives = ["random", "ar_tag", "shape"]
parking_spot = ["one","two","three", "four","five"]


exit_enter_waypoints = [

['exit1', (0.886, 0.45), (0.0, 0.0, 0, 1)],
['exit2', (1.26, 0.228), (0.0, 0.0, -0.731, .682)],
['exit3', (1.39, -0.00465), (0.0, 0.0, -0.731, .682)],
['enter1',(2.12, -0.84), (0, 0,.6935,.7203)],
['enter2', (3.68, -0.836), (0, 0,.6935,.7203)],
['enter3', (3.9, 0.17), (0, 0,.6935,.7203)]
]

shape_waypoints = [
['eight', (0.94 - .05, -0.961), (0,0,-1,0.006) ],
['seven', (2.12, -0.64), (0, 0,.6935,.7203)],
['six', (2.83+0.1, -0.666), (0, 0,.6935,.7203)]
]

push_waypoints = [
['one', (3.84 +.04, -1.75 -.1), (0.0, 0.0, -0.731, .682)],
['two', (3.12 , -1.8 -.1),  (0.0, 0.0, -0.731, .682)],
['three', (2.43 +0.3, -1.76 - .1),(0.0, 0.0, -0.731, .682) ],
['four', (1.73 - 0.06, -1.76 -.1),  (0.0, 0.0, -0.731, .682)],
['five', (0.899 +.04, -1.8 - 0.1,), (0.0, 0.0, -0.731, .682)]
]

ar_waypoints = [
['checkAR5', (0.899, -1.2 - 0.1,), (0.0, 0.0, -0.731, .682)],

['checkAR4', (1.60, -0.9),  (0.0, 0.0, -0.731, .682)],
['checkAR3', (2.48  , -0.9),(0.0, 0.0, -0.731, .682) ],
['checkAR2', (3.12 , -0.9),  (0.0, 0.0, -0.731, .682)],

['checkAR1', (3.84 +.04, -1.1), (0.0, 0.0, -0.731, .682)]


]

class SleepState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Line','Done'])
        self.led = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size = 1 )
        self.rate = rospy.Rate(10)  
        self.button = rospy.Subscriber('/joy', Joy, self.button_callback)
        self.end = 0                 # used to determine if the program should exit
        self.START = 0               # used to determine if the program should begin


    def button_callback(self,msg):
        rospy.loginfo('in callback')
        if msg.buttons[0] == 1:
            self.START = 1
        if msg.buttons[1] == 1:
            self.end = 1

    def execute(self, userdata):
        rospy.loginfo('Executing sleep state')

        while not rospy.is_shutdown():
            if self.end:
                return 'Done'
            if self.START:
                return 'Line'
        return 'Done'


class LineFollow(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Scan', 'TurnCounter','TurnClock','Stop','Done'])# 'GoToParkingStart'])
        self.bridge = cv_bridge.CvBridge()
        self.led1 = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size = 1 )
        self.led2 = rospy.Publisher('/mobile_base/commands/led2', Led, queue_size = 1 )
        self.image_sub = rospy.Subscriber('usb_cam/image_raw',   
                        Image,self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                            Twist, queue_size=1)
        self.button = rospy.Subscriber('/joy', Joy, self.button_callback)
        self.twist= Twist()
        self.rate = rospy.Rate(10)
        self.end = 0 
        self.stop = 0
        self.M = None
        self.RM = None
        self.image = None
        self.noLine = 0
        self.t1 = None
        self.time = 0

    def execute(self, userdata):
        global counter, BoxSpot, BoxGoal
        self.time = 1
        rospy.loginfo('Executing Line Follow state')
        self.stop = 0 
        self.twist = Twist()
        self.noLine = 0
        self.t1 = None
        self.led1.publish(0)
        self.led2.publish(0)
        while not rospy.is_shutdown():
            if self.end:
                return 'Done'

            elif self.stop:
                if counter == 2 or counter == 4:
                    counter += 1
                    self.twist.linear.x = 0.3
                    self.cmd_vel_pub.publish(self.twist)

                    return 'TurnCounter'

                elif counter == 1 or counter == 6 or counter == 0 or counter == 7 or counter == 8 or counter == 9:
                    #just stop for a moment
                    counter += 1
                    rospy.sleep(0.3)
                    self.twist = Twist()
                    self.cmd_vel_pub.publish(self.twist)
                    return 'Stop'

                elif counter == 5:
                    counter += 1
                    rospy.sleep(0.3)
                    self.twist = Twist()
                    self.cmd_vel_pub.publish(self.twist)
                    self.time = 0
                    return 'Stop'




                elif counter == 10:
                    BoxSpot = None
                    BoxGoal = None
                    counter = 0
                    rospy.sleep(1)
                    self.twist = Twist()
                    self.cmd_vel_pub.publish(self.twist)
                    led1 = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size = 1 )
                    led2 = rospy.Publisher('/mobile_base/commands/led2', Led, queue_size = 1 )
                    sound = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size = 1)
                    led1.publish(3)
                    led2.publish(3)
                    sound.publish(0)
                    rospy.sleep(1)
                    
                    return 'Stop'


            elif self.noLine == 2 and (counter < 6):
                counter += 1
                self.twist.linear.x = -.4
                self.cmd_vel_pub.publish(self.twist)
                rospy.sleep(3.5)
                self.twist = Twist()
                self.cmd_vel_pub.publish(self.twist)

                return 'Scan'

        return 'Done'

    def threshold_hsv_360(self,s_min, v_min, h_max, s_max, v_max, h_min, hsv):
        lower_color_range_0 = numpy.array([0, s_min, v_min],dtype=float)
        upper_color_range_0 = numpy.array([h_max/2., s_max, v_max],dtype=float)
        lower_color_range_360 = numpy.array([h_min/2., s_min, v_min],dtype=float)
        upper_color_range_360 = numpy.array([360/2., s_max, v_max],dtype=float)
        mask0 = cv2.inRange(hsv, lower_color_range_0, upper_color_range_0)
        mask360 = cv2.inRange(hsv, lower_color_range_360, upper_color_range_360)
        mask = mask0 | mask360
        return mask

    def button_callback(self,msg):
        rospy.loginfo('in callback')
        if msg.buttons[1] == 1:
            self.end = 1

    def image_callback(self, msg):
        global counter
        if self.time and self.noLine != 2:
            self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

            lower_white = numpy.array([180,170,170])#[186,186,186])    [180,180,180] [220,220,220]    # set upper and lower range for white mask
            upper_white = numpy.array([255,255,255])#[255,255,255]) [255,255,255]
            whitemask = cv2.inRange(self.image,lower_white,upper_white)


            redmask = self.threshold_hsv_360(110,110,320,255,255,20,hsv)
            #lower_red = numpy.array([120,130,130]) # [120,150,150]                          # set upper and lower range for red mask
            #upper_red = numpy.array([180,255,255]) # [180,255,255]
            #redmask = cv2.inRange(hsv,lower_red,upper_red)

      
            h, w, d =self.image.shape
            search_top = 3*h/4
            search_bot = search_top + 20

            whitemask[0:search_top, 0:w] = 0                                # search for white color
            whitemask[search_bot:h, 0:w] = 0

            redmask[0:search_top, 0:w] = 0                                  # search for red color
            redmask[search_bot:h, 0:w] = 0

            self.M = cv2.moments(whitemask)
            self.RM = cv2.moments(redmask)

            if self.RM['m00'] > 0:
                cx = int(self.RM['m10']/self.RM['m00'])
                cy = int(self.RM['m01']/self.RM['m00'])
                cv2.circle(self.image, (cx, cy), 20, (0,255,0),-1)
                self.noLine = 0
                self.stop = 1
                self.twist.linear.x = 0.3
                self.cmd_vel_pub.publish(self.twist)

            elif self.M['m00'] > 0 and self.stop == 0:
                #rospy.loginfo("Line found")
                self.noLine = 0
                self.PID_Controller(w)

            else:
                #rospy.loginfo("no line")
                if self.noLine == 0:
                    self.t1 = rospy.Time.now() + rospy.Duration(2)
                    self.noLine = 1
                elif self.noLine == 1 and (self.t1 <= rospy.Time.now()):
                    self.noLine = 2

            cv2.imshow("window", self.image)
            cv2.waitKey(3)

    def PID_Controller(self,w):
        global counter
        prev_err = 0
        integral = 0
        dt = 1

        cx = int(self.M['m10']/self.M['m00'])
        cy = int(self.M['m01']/self.M['m00'])
        cv2.circle(self.image, (cx, cy), 20, (0,0,255),-1)
        err = cx - w/2
        Kp = .0035 
        Ki = 0
        Kd = 0.004
        if counter == 3 or counter == 4 or counter == 5:
            Kd = .002
        integral = integral + err * dt
        derivative = (err-prev_err) / dt
        prev_err = err
        output = (err * Kp) + (integral * Ki) + (derivative * Kd)
        if counter == 3 or counter == 4 or counter == 5:
            self.twist.linear.x = 0.3
        else:
            self.twist.linear.x = .6
        self.twist.angular.z =  -output
        self.cmd_vel_pub.publish(self.twist)


class StopState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Line','Waypoint','Done'])
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                            Twist, queue_size=1)
        self.button = rospy.Subscriber('/joy', Joy, self.button_callback)
        self.end = 0
        self.twist = Twist()

    def button_callback(self,msg):
        rospy.loginfo('in callback')
        if msg.buttons[1] == 1:
            self.end = 1

    def execute(self,userdata):
        rospy.loginfo('Executing Stop state')
        self.twist = Twist()
        while not rospy.is_shutdown():
            
            time = rospy.Time.now() + rospy.Duration(0.5)
            while rospy.Time.now() < time:
                self.twist.linear.x = 0
                self.cmd_vel_pub.publish(self.twist)
                if self.end:
                    return 'Done'
                
            

            self.twist.linear.x = 0.3
            self.cmd_vel_pub.publish(self.twist)
            rospy.sleep(.5)

            if counter == 6:
                return 'Waypoint'
                
            return 'Line'
        return 'Done'


class Turn90Clockwise(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Line','Done'])
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                            Twist, queue_size=1)
        self.button = rospy.Subscriber('/joy', Joy, self.button_callback)
        self.end = 0
        self.twist = Twist()
        self.speed = -45
        self.angle = 90
        self.angular_speed = self.speed*2*math.pi/360
        self.relative_angle = self.angle*2.3*math.pi/360
        self.mult = 1.3
        self.mult2 = 1.2

    def button_callback(self,msg):
        rospy.loginfo('in callback')
        if msg.buttons[1] == 1:
            self.end = 1

    def execute(self,userdata):
        rospy.loginfo('Executing Turn90Clockwise state')

        global counter
        self.angular_speed = self.speed*2*math.pi/360
        self.relative_angle = self.angle*2.3*math.pi/360

        if counter == 9:
            self.relative_angle = self.relative_angle * self.mult
        if counter == 1:
            self.relative_angle = self.relative_angle * self.mult2

        self.twist = Twist()

        while not rospy.is_shutdown():

            current_angle = 0
            self.twist.angular.z = self.angular_speed
            self.cmd_vel_pub.publish(self.twist)
            t0 = rospy.Time.now().to_sec()

            while(current_angle < self.relative_angle):
                self.cmd_vel_pub.publish(self.twist)
                t1 = rospy.Time.now().to_sec()
                current_angle = abs(self.angular_speed)*(t1-t0)

            return 'Line'

        return 'Done'


class Turn90CounterClockwise(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Read', 'Scan', 'Line','Done'])
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                            Twist, queue_size=1)
        self.button = rospy.Subscriber('/joy', Joy, self.button_callback)
        self.end = 0
        self.twist = Twist()
        self.speed = 45
        self.angle = 105
        self.angular_speed = self.speed*2*math.pi/360
        self.relative_angle = self.angle*2.3*math.pi/360
        self.mult = 1.4
        self.mult2 = 0.8


    def button_callback(self,msg):
        rospy.loginfo('in callback')
        if msg.buttons[1] == 1:
            self.end = 1

    def execute(self,userdata):
        global counter
        self.angular_speed = self.speed*2*math.pi/360
        self.relative_angle = self.angle*2.3*math.pi/360
        if counter == 9 or counter == 5:
            self.relative_angle = self.relative_angle * self.mult
        if counter == 8:
            self.relative_angle = self.relative_angle * self.mult2


        rospy.loginfo('Executing Turn90 state')
        self.twist = Twist()
        self.twist.linear.x =.3

        if counter == 9:
            self.twist.angular.z = 0.6
        self.cmd_vel_pub.publish(self.twist)
        t0 = rospy.Time.now() + rospy.Duration(3)

        while t0 > rospy.Time.now():
            x = 0
        self.twist = Twist()

        while not rospy.is_shutdown():
            current_angle = 0
            self.twist.angular.z = self.angular_speed
            self.cmd_vel_pub.publish(self.twist)
            t0 = rospy.Time.now().to_sec()
            while(current_angle < self.relative_angle):
                self.cmd_vel_pub.publish(self.twist)
                t1 = rospy.Time.now().to_sec()
                current_angle = abs(self.angular_speed)*(t1-t0)
            rospy.loginfo(counter)

            if counter == 1:
                return 'Scan' 
            elif counter == 8 or counter == 9 or counter == 10:
                return 'Read'

            elif counter == 3 or counter == 5:
                return 'Line'
            
        return 'Done'


class Turn180(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Line','Done'])
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                            Twist, queue_size=1)

        self.button = rospy.Subscriber('/joy', Joy, self.button_callback)
        self.end = 0
        self.twist = Twist()
        self.speed = 90
        self.angle = 180
        self.angular_speed = self.speed*2*math.pi/360
        self.relative_angle = self.angle*2.6*math.pi/360


    def button_callback(self,msg):
        rospy.loginfo('in callback')
        if msg.buttons[1] == 1:
            self.end = 1

    def execute(self,userdata):
        rospy.loginfo('Executing Turn90 state')
        self.twist = Twist()
        while not rospy.is_shutdown():

            current_angle = 0
            self.twist.angular.z = self.angular_speed
            self.cmd_vel_pub.publish(self.twist)
            t0 = rospy.Time.now().to_sec()

            while(current_angle < self.relative_angle):
                self.cmd_vel_pub.publish(self.twist)
                t1 = rospy.Time.now().to_sec()
                current_angle = self.angular_speed*(t1-t0)

            self.twist.angular.z = 0
            self.cmd_vel_pub.publish(self.twist)
            
            return 'Line'
    
        return 'Done'


class ScanObject(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Read', 'TurnClock','Done'])
        
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                            Twist, queue_size=1)
        #self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',   
         #               Image,self.image_callback)
        self.led1 = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size = 1 )
        self.led2 = rospy.Publisher('/mobile_base/commands/led2', Led, queue_size = 1 )
        self.button = rospy.Subscriber('/joy', Joy, self.button_callback)
        self.sound = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size = 1)

        self.bridge = cv_bridge.CvBridge()
        self.val = None
        self.found = 0
        self.lst = []
        self.scanTime = 0


    def button_callback(self,msg):
        rospy.loginfo('in callback')
        if msg.buttons[1] == 1:
            self.end = 1

    def execute(self,userdata):
        image_sub = rospy.Subscriber('/camera/rgb/image_raw',   
                        Image,self.image_callback)
        self.scanTime = 1
        self.lst = []
        self.found = 0
        global counter
        self.twist = Twist()
        self.cmd_vel_pub.publish(self.twist)
        while not rospy.is_shutdown():
            if self.found:
                if counter == 4:
                    self.val += 1
                if self.val == 1:
                    rospy.loginfo('here1')
                    self.led2.publish(1)
                    self.sound.publish(0)

                elif self.val == 2:
                    rospy.loginfo('here2')
                    self.led1.publish(1)
                    self.sound.publish(0)
                    rospy.sleep(0.5)
                    self.sound.publish(0)
                else:
                    rospy.loginfo('here3')
                    self.led1.publish(1)
                    self.led2.publish(1)
                    self.sound.publish(0)
                    rospy.sleep(0.5)
                    self.sound.publish(0)
                    rospy.sleep(0.5)
                    self.sound.publish(0)

                self.scanTime = 0  
                image_sub.unregister()              
                if counter == 4:
                    return 'Read'
                return 'TurnClock'

        return 'Done'

    def image_callback(self, msg):
        if self.scanTime:
            global counter
            self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            if counter == 4:
                redmask = self.threshold_hsv_360(140,100,10,255,255,120,hsv)    # ignores green, really good for red
            else:
                redmask = self.threshold_hsv_360(30,80,20,255,255,120,hsv)
            ret, thresh = cv2.threshold(redmask, 127, 255, 0)
            im2, cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(redmask, cnts, -1, (0,255,0), 3)

            num = 0
            for c in cnts:

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, .04 * peri, True)

                if cv2.contourArea(c) > 7000: 
                    num +=1


            rospy.loginfo(num)
            #img = measure.label(redmask, background=0)
            #img += 1
            #propsa = measure.regionprops(img.astype(int))
            #length = len(propsa)

            self.lst.append(num)

            if len(self.lst) > 15:
                self.val = self.lst[-1]
                self.found = 1

        #self.grouping += length - 1
        #self.i += 1
        #self.avg = self.grouping/self.i
        #cv2.imshow("window", redmask)
        #cv2.waitKey(3)

        

    def threshold_hsv_360(self,s_min, v_min, h_max, s_max, v_max, h_min, hsv):
        lower_color_range_0 = numpy.array([0, s_min, v_min],dtype=float)
        upper_color_range_0 = numpy.array([h_max/2., s_max, v_max],dtype=float)
        lower_color_range_360 = numpy.array([h_min/2., s_min, v_min],dtype=float)
        upper_color_range_360 = numpy.array([360/2., s_max, v_max],dtype=float)
        mask0 = cv2.inRange(hsv, lower_color_range_0, upper_color_range_0)
        mask360 = cv2.inRange(hsv, lower_color_range_360, upper_color_range_360)
        mask = mask0 | mask360
        return mask

class ReadShape(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Turn180', 'TurnClock','Done'])
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                            Twist, queue_size=1)
        #self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',   
        #                Image,self.image_callback)
        self.led1 = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size = 1 )
        self.led2 = rospy.Publisher('/mobile_base/commands/led2', Led, queue_size = 1 )
        self.sound = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size=1)

        self.button = rospy.Subscriber('/joy', Joy, self.button_callback)
        self.bridge = cv_bridge.CvBridge()
        self.readTime = 0
        self.found = 0
        self.shape_list = list()

    def button_callback(self,msg):
        rospy.loginfo('in callback')
        if msg.buttons[1] == 1:
            self.end = 1

    def execute(self,userdata):
        image_sub = rospy.Subscriber('/camera/rgb/image_raw',   
                        Image,self.image_callback)
        global counter
        self.shape_list = list()
        rospy.sleep(1)
        self.readTime = 1
        self.lst = []
        self.twist = Twist()
        self.cmd_vel_pub.publish(self.twist)
        self.found = 0
        while not rospy.is_shutdown():
            if self.found:
                self.readTime = 0
                image_sub.unregister()
                if counter == 4:
                    return 'Turn180'
                return 'TurnClock'

        return 'Done'

    def image_callback(self, msg):
        global counter, gshape
        if self.readTime:
            self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            if counter == 4:
                lower_red = numpy.array([40,50,50])#[100,0,0])                   
                upper_red = numpy.array([90,255,255])#[255,30,30])
                mask = cv2.inRange(hsv,lower_red,upper_red) # green masks
            else:
                mask = self.threshold_hsv_360(140,10,10,255,255,120,hsv)
            #cv2.inRange(hsv,lower_red,upper_red)
            #rospy.loginfo(redmask)
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            im2, cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, cnts, -1, (0,255,0), 3)
            #cnts = imutils.grab_contours(cnts)
            sd = ShapeDetector()
            shape = None
            for c in cnts:
                shape = sd.detect(c)
     
                if shape != None:
                    self.shape_list.append(shape)
                
                
            if len(self.shape_list) >= 20:
                if counter == 4:
                    gshape =  max(set(self.shape_list), key=self.shape_list.count)
                    rospy.loginfo(gshape)
                else:
                    shape =  max(set(self.shape_list), key=self.shape_list.count)
                    if gshape == shape:
                        self.led1.publish(3)
                        #TODO: make beep
                        self.sound.publish(6)

                    rospy.loginfo(shape)
                self.found = 1

    def threshold_hsv_360(self,s_min, v_min, h_max, s_max, v_max, h_min, hsv):
        lower_color_range_0 = numpy.array([0, s_min, v_min],dtype=float)
        upper_color_range_0 = numpy.array([h_max/2., s_max, v_max],dtype=float)
        lower_color_range_360 = numpy.array([h_min/2., s_min, v_min],dtype=float)
        upper_color_range_360 = numpy.array([360/2., s_max, v_max],dtype=float)
        mask0 = cv2.inRange(hsv, lower_color_range_0, upper_color_range_0)
        mask360 = cv2.inRange(hsv, lower_color_range_360, upper_color_range_360)
        mask = mask0 | mask360
        return mask
                        


class FindTag(smach.State):
    def __init__(self):

        smach.State.__init__(self, outcomes=['ApproachTag','Done'])
        self.cmd_vel_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=5)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odomCallback)
        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.set_cmd_vel)
        

        rospy.wait_for_message('ar_pose_marker', AlvarMarkers)
        

        self.move_cmd = Twist()  
        # Set flag to indicate when the AR marker is visible
        self.current_marker = None
        self.look_for_marker = 1
        self.now = 0
        self.pose = None


    def execute(self,userdata):
        global start
        rospy.loginfo("Executing State FindTag")

        rospy.wait_for_message('odom', Odometry)

        while not rospy.is_shutdown():
            self.now = 1
            self.move_cmd = Twist()
            self.found = 0

            while self.found == 0:
                self.cmd_vel_pub.publish(self.move_cmd)
            self.now = 0

            return 'ApproachTag'
        return 'Done'


    def set_cmd_vel(self,msg):
        # if there is a marker do try
        global found_markers
        if self.now:
            try: 
                marker = msg.markers[0]
                self.current_marker = marker.id
                if (self.current_marker not in found_markers) and (self.current_marker != 0):
                        rospy.loginfo("FOLLOWER found Target!")
                        found_markers.append(self.current_marker)
                        self.found = 1
                        rospy.loginfo(found_markers)

                else:
                    rospy.loginfo("FOLLOWER is looking for Target")
                    self.move_cmd.linear.x = 0
                    self.move_cmd.angular.z = 0.3

            except:
                self.move_cmd.linear.x = 0
                self.move_cmd.angular.z = 0.3


    def odomCallback(self, msg):
        self.pose = msg.pose.pose

class Shape_Waypoints(smach.State):
    def __init__(self, name, position, orientation):
        smach.State.__init__(self, outcomes=['success','checkAR5'])
        rospy.loginfo("Setting up client")
        self.led1 = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size = 1 )
        self.led2 = rospy.Publisher('/mobile_base/commands/led2', Led, queue_size = 1 )
        self.sound = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size=1)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("ready1")
        self.client.wait_for_server()
        rospy.loginfo("ready2")
        self.name = name
        self.bridge = cv_bridge.CvBridge()
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = 0
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]
        self.first = 1
        self.found = 0


    def execute(self, userdata):
        global counter, objectives, random, parking_spot
        image_sub = rospy.Subscriber('camera/rgb/image_raw',   
                        Image,self.image_callback)
        self.shapeLocated = 0
        self.first = 1
        self.readTime = 0
        self.found = 0
        self.led1.publish(0)
        self.shape_list = []

        while not rospy.is_shutdown():

            #go to the goal
            if self.first:
                self.client.send_goal(self.goal)
                self.client.wait_for_result()
                rospy.loginfo(self.name)
                self.first = 0
                

            
            cmd_vel_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=5)
            twist = Twist()
            twist.linear.x = -0.2
            goal = rospy.Time.now() + rospy.Duration(2)
            # back up to read
            while rospy.Time.now() < goal:
                cmd_vel_pub.publish(twist) 

            twist.linear.x = 0.0  
            cmd_vel_pub.publish(twist) 
            self.readTime = 1
            rospy.sleep(2)

            if self.shapeLocated:
                self.led1.publish(1)
                self.sound.publish(0)
                rospy.sleep(1)
                self.led1.publish(0)

            goal = rospy.Time.now() + rospy.Duration(2.2)
            twist.linear.x = 0.2
            # rad shape
            while rospy.Time.now() < goal:
                cmd_vel_pub.publish(twist) 

            twist.linear.x = 0.0  
            cmd_vel_pub.publish(twist) 
                
            
            if self.shapeLocated:
                self.led1.publish(1)
                self.led2.publish(2)
                self.sound.publish(0)
                rospy.sleep(1)
            
            image_sub.unregister()
            if self.name == 'six':
                self.readTime = 0
                return 'checkAR5'
            # if not at last goto next waypoint
            else:
                self.readTime = 0
                return 'success'

        return 'Done'


    # if still have to find shape listen to callback when at waypoint
    # if shape found remove 'shape' from objectives list
    
    def image_callback(self, msg):
        global gshape, objectives
        if self.readTime and 'shape' in objectives:
            self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)  
            mask = self.threshold_hsv_360(110,110,320,255,255,20,hsv)       
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            im2, cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, cnts, -1, (0,255,0), 3)
            sd = ShapeDetector()

            shape = None
            for c in cnts:
                shape = sd.detect(c)
                if shape != None:
                    self.shape_list.append(shape)           

            if len(self.shape_list) >= 20:
                shape =  max(set(self.shape_list), key=self.shape_list.count)

                if gshape == shape:

                    self.shapeLocated = 1 
                    rospy.loginfo(shape)
                    objectives.remove('shape')

                self.found = 1
                
    def threshold_hsv_360(self,s_min, v_min, h_max, s_max, v_max, h_min, hsv):

        lower_color_range_0 = numpy.array([0, s_min, v_min],dtype=float)
        upper_color_range_0 = numpy.array([h_max/2., s_max, v_max],dtype=float)
        lower_color_range_360 = numpy.array([h_min/2., s_min, v_min],dtype=float)
        upper_color_range_360 = numpy.array([360/2., s_max, v_max],dtype=float)
        mask0 = cv2.inRange(hsv, lower_color_range_0, upper_color_range_0)
        mask360 = cv2.inRange(hsv, lower_color_range_360, upper_color_range_360)
        mask = mask0 | mask360
        return mask


    


# this function goes to the exit and enter waypoints, doing nothing but going to next state
class Exit_Enter_Waypoints(smach.State):
    def __init__(self, name, position, orientation):
        smach.State.__init__(self, outcomes=['success','Line','eight'])
        rospy.loginfo("Setting up client")
        self.initpos = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("ready1")
        self.client.wait_for_server()
        rospy.loginfo("ready2")
        self.name = name
        self.bridge = cv_bridge.CvBridge()
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = 0
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]
        self.first = 1


    def execute(self, userdata):
        global counter, objectives, random, parking_spot
        self.first = 1
        while not rospy.is_shutdown():
            # set initial position on map
            
            if self.name == 'exit1':
                self.calcInitial()
            
            #go to the goal
            if self.first:
                self.client.send_goal(self.goal)
                self.client.wait_for_result()
                rospy.loginfo(self.name)
                self.first = 0
            
            if self.name == 'exit3':
                return 'eight'

            if self.name == 'enter3':
                return 'Line'

            else:
                return 'success'

        return 'Done'

    def calcInitial(self):
        start = PoseWithCovarianceStamped()
        start.header.frame_id = 'map'
        start.pose.pose.position.x = 0.153
        start.pose.pose.position.y = 0.325
        start.pose.pose.position.z = 0
        start.pose.pose.orientation.x = 0
        start.pose.pose.orientation.y = 0
        start.pose.pose.orientation.z = 0
        start.pose.pose.orientation.w = 1
        self.initpos.publish(start)
        rospy.sleep(3)
        rospy.loginfo("before")
        self.cmd_vel_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=5)
        twist = Twist()


        goal = rospy.Time.now() + rospy.Duration(3)
        while rospy.Time.now() < goal:
            twist.angular.z = -1
            self.cmd_vel_pub.publish(twist)   

        goal = rospy.Time.now() + rospy.Duration(3)
        while rospy.Time.now() < goal:
            twist.angular.z = 1
            self.cmd_vel_pub.publish(twist)   

        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("after")

        '''def alvarCallback(self, msg):
        global objectives
        if self.readTime and 'ar_tag' in objectives:
            try:
                rospy.loginfo(msg.markers[0].id)
                marker = msg.markers[0]
                if marker.id != 0:
                    objectives.remove('ar_tag')
                    self.tagLocated = 1

            except:
                pass
                '''

class AR_Waypoints(smach.State):
    def __init__(self, name, position, orientation):
        smach.State.__init__(self, outcomes=['success','SideBox','enter1'])
        rospy.loginfo("Setting up client")
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.led1 = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size = 1 )
        rospy.loginfo("ready1")
        self.client.wait_for_server()
        rospy.loginfo("ready2")
        self.name = name
        self.sound = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size = 1)
        self.bridge = cv_bridge.CvBridge()
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = 0
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]
        self.first = 1
        self.found = 0
        self.readTime = 0
        self.tagLocated = 0
        self.color = 0


    def execute(self, userdata):
        global counter, objectives, random, parking_spot, BoxSpot, BoxGoal
        alvar_sub = rospy.Subscriber('ar_pose_marker', 
            AlvarMarkers, self.alvarCallback)
        self.tagLocated = 0
        self.color = 0
        self.first = 1
        self.readTime = 0
        self.found = 0
        self.led1.publish(0)
        
        twist = Twist()
        while not rospy.is_shutdown():

            #go to the goal
            if self.first:

                self.client.send_goal(self.goal)
                self.client.wait_for_result()
                rospy.loginfo(self.name)
                self.first = 0
                

            
            cmd_vel_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=5)
            

            twist.linear.x = 0.0  
            cmd_vel_pub.publish(twist) 

            goal = rospy.Time.now() + rospy.Duration(0.4)

            while  rospy.Time.now() < goal:
                twist.angular.z = -0.4
                #self.PID_Controller()
                cmd_vel_pub.publish(twist)   

            self.readTime = 1
            rospy.sleep(2)
                
            
            if self.tagLocated:
                self.led1.publish(self.color)
                self.sound.publish(0)
                rospy.sleep(1)
                self.led1.publish(0)
            
            alvar_sub.unregister()

            if self.name == 'checkAR1':
                if BoxSpot == None or BoxGoal == None:
                    return 'enter1'
                self.readTime = 0
                return 'SideBox'
            # if not at last goto next waypoint
            else:
                self.readTime = 0
                return 'success'

        return 'Done'

    def alvarCallback(self, msg):
        global objectives, BoxGoal, OtherAR, BoxAR, BoxSpot
        if self.readTime and 'ar_tag' in objectives:
            try:
                
                marker = msg.markers[0]
                if marker.id != 0:
                    if marker.id in OtherAR:
                        rospy.loginfo(msg.markers[0].id)
                        BoxGoal = self.name[-1]
                        self.tagLocated = 1
                        self.color = 1
                    elif marker.id in BoxAR:
                        rospy.loginfo(msg.markers[0].id)
                        BoxSpot = self.name[-1]
                        self.tagLocated = 1
                        self.color = 3

            except:
                pass

# this function goes to the exit and enter waypoints, doing nothing but going to next state
class SideBox(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['PushBox'])
        rospy.loginfo("Setting up client")
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.cmd_vel_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=5)

        self.twist = Twist()
        rospy.loginfo("ready1")
        self.client.wait_for_server()
        rospy.loginfo("ready2")
        self.first = 1


    def execute(self, userdata):
        global counter, objectives, BoxSpot,BoxGoal, push_waypoints
        self.first = 1
        while not rospy.is_shutdown():

            first =MoveBaseGoal()
            first.target_pose.header.frame_id = 'map'
            first.target_pose.pose.orientation.x = 0
            first.target_pose.pose.orientation.y = 0
            first.target_pose.pose.orientation.z = .6935
            first.target_pose.pose.orientation.w = .7203

            first.target_pose.pose.position.x = 2.12
            first.target_pose.pose.position.y = -0.64
            first.target_pose.pose.position.z = 0

            self.client.send_goal(first)
            self.client.wait_for_result()











            #go to the goal
            if self.first:
                self.goal = MoveBaseGoal()
                self.goal.target_pose.header.frame_id = 'map'
                if int(BoxSpot) < int(BoxGoal):



                    position = push_waypoints[abs(int(BoxSpot) - 2)][1]
                    orientation = push_waypoints[abs(int(BoxSpot) - 2)][2]


                    #TODO: fix orientation
                    self.goal.target_pose.pose.orientation.x = 0
                    self.goal.target_pose.pose.orientation.y = 0
                    self.goal.target_pose.pose.orientation.z = -1
                    self.goal.target_pose.pose.orientation.w = 0.06

                  
                else:
                    position = push_waypoints[abs(int(BoxSpot))][1]
                    orientation = push_waypoints[abs(int(BoxSpot))][2]
                    self.goal.target_pose.pose.orientation.x = 0
                    self.goal.target_pose.pose.orientation.y = 0
                    self.goal.target_pose.pose.orientation.z = 0
                    self.goal.target_pose.pose.orientation.w = 1



                self.goal.target_pose.pose.position.x = position[0]
                self.goal.target_pose.pose.position.y = position[1]
                self.goal.target_pose.pose.position.z = 0
                
                self.client.send_goal(self.goal)
                self.client.wait_for_result()



            #adjust orientation 

            if int(BoxSpot) < int(BoxGoal):

                goal = rospy.Time.now() + rospy.Duration(0.6)

                while  rospy.Time.now() < goal:
                    self.twist.angular.z = -0.6
                    #self.PID_Controller()
                    self.cmd_vel_pub.publish(self.twist)   

            else:

                goal = rospy.Time.now() + rospy.Duration(0.5) #TODO: may need new time


                while  rospy.Time.now() < goal:
                    self.twist.angular.z = 0.6
                    #self.PID_Controller()
                    self.cmd_vel_pub.publish(self.twist)   


            self.twist.angular.z = 0
                    #self.PID_Controller()
            self.cmd_vel_pub.publish(self.twist)   


            self.first = 0
            
            return 'PushBox'

class PushBox(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['enter1'])
        self.cmd_vel_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=5)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odomCallback)
        self.led1 = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size = 1 )
        self.led2 = rospy.Publisher('/mobile_base/commands/led2', Led, queue_size = 1 )
        self.sound = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size = 1)
        self.pose = None
        self.twist = Twist()
        self.first = None


    def execute(self, userdata):
        global BoxGoal, push_waypoints, BoxSpot
        goal = push_waypoints[int(BoxGoal)-1][1][0]
        self.twist = Twist()
        rospy.loginfo("excuting push to X")


        ind_diff  = abs(int(BoxSpot) - int(BoxGoal))


        while not rospy.is_shutdown():


            goal = rospy.Time.now() + rospy.Duration(4  * ind_diff + 3)

            
            while  rospy.Time.now() < goal:
                self.twist.linear.x = 0.2
                #self.PID_Controller()
                self.cmd_vel_pub.publish(self.twist)   


            self.twist.linear.x = 0.0
            self.cmd_vel_pub.publish(self.twist)   
            
            #at start
            self.led1.publish(1)
            self.led2.publish(3)
            self.sound.publish(0)
            rospy.sleep(2)
            self.led1.publish(0)
            self.led2.publish(0)

            self.twist.linear.x = 0.0
            self.cmd_vel_pub.publish(self.twist)   


            #back up

            goal = rospy.Time.now() + rospy.Duration(2)

            
            while  rospy.Time.now() < goal:
                self.twist.linear.x = -0.2
                #self.PID_Controller()
                self.cmd_vel_pub.publish(self.twist)   



            return 'enter1'
        
    


    def odomCallback(self,msg):
        self.pose = msg.pose.pose

    def PID_Controller(self):

        prev_err = 0
        integral = 0
        dt = 1

        err = self.pose.position.y - self.first.position.y
        Kp = .005 
        Ki = 0
        Kd = .003
        integral = integral + err * dt
        derivative = (err-prev_err) / dt
        prev_err = err
        output = (err * Kp) + (integral * Ki) + (derivative * Kd)
        self.twist.linear.x = 0.2
        self.twist.angular.z =  -output
        self.cmd_vel_pub.publish(self.twist)

    

def main():
    rospy.init_node('Comp3')
    rate = rospy.Rate(10)
    sm = smach.StateMachine(outcomes = ['DoneProgram'])
    sm.set_initial_state(['LineFollow'])

    with sm:
        
        #Compeition 2 states and transitions 
        smach.StateMachine.add('SleepState', SleepState(),
                                        transitions = {'Line': 'LineFollow',
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('LineFollow', LineFollow(),
                                        transitions = { 'Scan': 'ScanObject',
                                                        'TurnCounter': 'Turn90CounterClockwise',
                                                        'TurnClock': 'Turn90Clockwise',
                                                        'Stop': 'StopState',
                                                        #'GoToParkingStart' :  'GoToStart',
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('StopState', StopState(),
                                        transitions = {'Line': 'LineFollow',
                                                        'Waypoint':'exit1',
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('Turn90Clockwise', Turn90Clockwise(),
                                        transitions = {'Line': 'LineFollow',
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('Turn90CounterClockwise', Turn90CounterClockwise(),
                                        transitions = { 'Read': 'ReadShape',
                                                        'Scan': 'ScanObject',
                                                        'Line': 'LineFollow',       
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('Turn180', Turn180(),
                                        transitions = {'Line': 'LineFollow',
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('ScanObject', ScanObject(),
                                        transitions = { 'Read': 'ReadShape',
                                                        'TurnClock': 'Turn90Clockwise',
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('ReadShape', ReadShape(),
                                        transitions = { 'Turn180': 'Turn180',
                                                        'TurnClock': 'Turn90Clockwise',
                                                        'Done' : 'DoneProgram'})
        for i, w in enumerate(exit_enter_waypoints):
            StateMachine.add(w[0], Exit_Enter_Waypoints(w[0], w[1], w[2]),transitions={'success':exit_enter_waypoints[(i + 1) %  len(exit_enter_waypoints)][0],
                                                                      'Line': 'LineFollow',
                                                                      'eight':'eight'}  )

        #shape_waypoints
        for i, w in enumerate(shape_waypoints):
            StateMachine.add(w[0], Shape_Waypoints(w[0], w[1], w[2]),transitions={'success':shape_waypoints[(i + 1) %  len(shape_waypoints)][0],
                                                                      'checkAR5':'checkAR5'})

        for i, w in enumerate(ar_waypoints):
            StateMachine.add(w[0], AR_Waypoints(w[0], w[1], w[2]),transitions={'success':ar_waypoints[(i + 1) %  len(ar_waypoints)][0],
                                                                      'SideBox':'SideBox',
                                                                        'enter1':'enter1'})

        smach.StateMachine.add('SideBox', SideBox(),
                                        transitions = { 'PushBox': 'PushBox'})

        smach.StateMachine.add('PushBox', PushBox(),
                                        transitions = { 'enter1': 'enter1'})




    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    

    sis.start()
    
    outcome = sm.execute() 
    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
