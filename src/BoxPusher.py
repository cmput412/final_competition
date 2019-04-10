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

#ordered_waypoints = [
"""
 ['initial', (0.887,0.5), (0.0, 0.0, 0, 1)],              #(0,0,-1,0.006) faces chairs
 ['zero',(1.35,-0.15), (0.0, 0.0, -0.731, .682)],      #(0, 0,.6935,.7203) faces door
 ['one', (0.83,-0.872),  (0,0,-1,0.006)],             #(0.0, 0.0, 0, 1) faces window
 ['two', (0.955, -1.88),  (0.0, 0.0, -0.731, .682)],   # (0.955, -1.78) #(0.0, 0.0, -0.731, .682) faces wall
 ['three',(1.78, -1.75),  (0.0, 0.0, -0.731, .682)],  #(1.68, -1.75)  
 ['four', (2.47, -1.7),  (0.0, 0.0, -0.731, .682)],
 ['five', (3.21, -1.69),  (0.0, 0.0, -0.731, .682)],   #(3.21, -1.63)
 ['six', (3.9, -1.71),  (0.0, 0.0, -0.731, .682)],
 ['seven',(2.73, -0.502),  (0, 0,.6935,.7203)],     #(2.73, -0.582)
 ['eight', (1.9, -0.655),  (0, 0,.6935,.7203)],
 ['exit', (3.35, -1.38),  (0.0, 0.0fdone, 0, 1)],
 ['end', (3.85, 0.1),  (0, 0,.6935,.7203)]
 """
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
['checkAR5', (3.84 +.1, -1.75 -.15), (0.0, 0.0, -0.731, .682)],
['checkAR4', (3.12  +0.1 , -1.8 -.1 ),  (0.0, 0.0, -0.731, .682)],
['checkAR3', (2.43 -0.05 , -1.76 -0.1 ),(0.0, 0.0, -0.731, .682) ],
['checkAR2', (1.73 -.3, -1.76 -.1),  (0.0, 0.0, -0.731, .682)],
['checkAR1', (0.899 -.2, -1.8 ,), (0.0, 0.0, -0.731, .682)]
]
'''ar_waypoints = [
['checkAR5', (0.899, -1.2 - 0.1,), (0.0, 0.0, -0.731, .682)],

['checkAR4', (1.60, -0.9),  (0.0, 0.0, -0.731, .682)],
['checkAR3', (2.48  , -0.9),(0.0, 0.0, -0.731, .682) ],
['checkAR2', (3.12 , -0.9),  (0.0, 0.0, -0.731, .682)],

['checkAR1', (3.84 +.04, -1.1), (0.0, 0.0, -0.731, .682)]


]'''

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
        smach.State.__init__(self, outcomes=['Stop','Done'])
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
        global counter
        self.time = 1
        rospy.loginfo('Executing Line Follow state')
        self.stop = 0 
        self.twist = Twist()
        self.noLine = 0
        self.t1 = None
        self.led1.publish(0)
        self.led2.publish(0)

        countup = True
        while not rospy.is_shutdown():

            countup = True
            if self.end:
                return 'Done'

            


            elif self.stop: #encountered red line
                rospy.loginfo(counter)

                if counter == 5: #about to enter parking lot

                    self.twist = Twist()
                    self.cmd_vel_pub.publish(self.twist)
                    self.time = 0

                    counter += 1
                    return 'Stop'

                elif counter == 10: #done
                    counter = 0
                    
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

                elif countup:


                    if counter == 2:
                        counter = 5


                    else:
                        counter += 1
                    countup = False

                    self.twist.linear.x = 0.5

                    self.cmd_vel_pub.publish(self.twist)
                    rospy.sleep(0.5)
                    self.stop = 0 

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
                self.twist.linear.x = 0.5
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

    def PID_Controller(self, w):

        prev_err = 0
        integral = 0
        dt = 1

        cx = int(self.M['m10']/self.M['m00'])
        cy = int(self.M['m01']/self.M['m00'])
        cv2.circle(self.image, (cx, cy), 20, (0,0,255),-1)
        err = cx - w/2
        Kp = .0035 
        Ki = 0
        Kd = .004
        integral = integral + err * dt
        derivative = (err-prev_err) / dt
        prev_err = err
        output = (err * Kp) + (integral * Ki) + (derivative * Kd)
        self.twist.linear.x = 0.6
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
            
            time = rospy.Time.now() + rospy.Duration(2)
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
        while not rospy.is_shutdown():

            first = MoveBaseGoal()
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
                    self.cmd_vel_pub.publish(self.twist)   

            else:

                goal = rospy.Time.now() + rospy.Duration(0.5) #TODO: may need new time


                while  rospy.Time.now() < goal:
                    self.twist.angular.z = 0.6
                    self.cmd_vel_pub.publish(self.twist)   


            self.twist.angular.z = 0
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
    rospy.init_node('Comp5')
    rate = rospy.Rate(10)
    sm = smach.StateMachine(outcomes = ['DoneProgram'])
    sm.set_initial_state(['LineFollow'])

    with sm:
        
        #Compeition 2 states and transitions 
        smach.StateMachine.add('SleepState', SleepState(),
                                        transitions = {'Line': 'LineFollow',
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('LineFollow', LineFollow(),
                                        transitions = {'Stop': 'StopState',
                                                        'Done' : 'DoneProgram'})

        smach.StateMachine.add('StopState', StopState(),
                                        transitions = {'Line': 'LineFollow',
                                                        'Waypoint':'exit1',
                                                        'Done' : 'DoneProgram'})

        for i, w in enumerate(exit_enter_waypoints):
            StateMachine.add(w[0], Exit_Enter_Waypoints(w[0], w[1], w[2]),transitions={'success':exit_enter_waypoints[(i + 1) %  len(exit_enter_waypoints)][0],
                                                                      'Line': 'LineFollow',
                                                                      'eight': 'checkAR5'   }  )

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