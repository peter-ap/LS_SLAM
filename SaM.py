import numpy as np
import math as m 
import os


#read in file:
def readData(path):
    path = os.getcwd() + "/" + path
    print(path)

    try:
        with open(path, 'r') as l:
            lasers = []
            # first, go through the laser file
            for line in l:
                tokens = line.split(' ')
                if tokens[0] == 'LASER':
                    # get amount of readings
                    num_readings = int(tokens[2])
                    # get opening angle from laser
                    opening_angle = float(tokens[3])
                    # get each laser range from scan
                    scans = np.array(tokens[4:4+num_readings], dtype=np.float64)
                    noise = np.random.normal(-0.03,0.03,len(scans))
                    scans = scans+noise
                    # get angle of each beam 
                    index = np.arange(-opening_angle/2, opening_angle/2 + opening_angle/num_readings, opening_angle/num_readings)
                    angles = np.delete(index, num_readings//2)
                    converted_scans = []
                    converted_scans = np.array([np.cos(angles), np.sin(angles)]).T * scans[:, np.newaxis]

                    lasers.append(np.array(converted_scans))

       

        return lasers, num_readings, opening_angle
    
    except:
        print(path, " is not a correct path. Check it!!")


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def split_and_merge(cluster):
    if len(cluster)>2:
        p_1 = np.array([cluster[0][0], cluster[0][1]])
        p_2 = np.array([cluster[-1][0], cluster[-1][1]])
        distance = []
        for i in range(0,len(cluster)):
            p_3 = np.array([cluster[i][0], cluster[i][1]])
            d = np.linalg.norm(np.cross(p_2-p_1, p_1-p_3))/np.linalg.norm(p_2-p_1)
            distance.append(abs(d))
        
        l1,l2=[],[]
        max_value = max(distance)
        max_index = None
        #finding a break point
        if max_value > .3:
            max_index = distance.index(max_value)
            for i in range(max_index+1):
                l1.append(cluster[i])
            for i in range(max_index+1, len(cluster)):
                l2.append(cluster[i])
            
            return l1, l2

        else:
            return 0,0

    else:
        return 0,0


def calculate_line(points):
    y_sum,x_sum, xy_sum, x2_sum = 0,0,0,0
    for point in points:
        x_sum = x_sum + point[0]
        y_sum = y_sum + point[1]
        xy_sum = xy_sum + point[0]*point[1]
        x2_sum = x2_sum + point[0]*point[0]

    intercept = ((y_sum * x2_sum) - (x_sum * xy_sum)) / (len(points)*x2_sum - (x_sum*x_sum))
    slope = (len(points)*(xy_sum) - x_sum*y_sum) / (len(points)*x2_sum - (x_sum*x_sum))
    
    return intercept, slope


class laserscan:

    def __init__(self, x , y, max_range, min_range, num_scans, angle):
        
        self.num_scans = num_scans
        self.angle = angle
        self.max_range = max_range
        self.min_range = min_range
        self.x_list = x
        self.y_list = y
        self.corners = []
        self.scan_points = []
        self.rough_clusters = []
        self.line_clusters = []
        self.successfull_split = []

    def set_scan_points(self):
        for i in range(len(self.x_list)):
            d = m.sqrt((self.x_list[i]*self.x_list[i]) + (self.y_list[i]* self.y_list[i]))
            if(d > self.min_range):
                self.scan_points.append([self.x_list[i], self.y_list[i]])


    def large_split(self):
        sub_angle = self.angle/(self.num_scans-1)
        max_distance = (np.sin(sub_angle)*self.max_range)*6      
        cluster=[]

        for i in range(1,len(self.scan_points)):
            prev_point = self.scan_points[i-1]
            cur_point = self.scan_points[i]
            dist = m.sqrt(((prev_point[0]-cur_point[0])*(prev_point[0]-cur_point[0])) + ((prev_point[1]-cur_point[1])*(prev_point[1]-cur_point[1])))
            cluster.append(prev_point)

            if dist > max_distance:
                self.rough_clusters.append(cluster)
                cluster = []

            if i == len(self.scan_points)-1:
                #last point against first point in case 360
                if dist > max_distance:   
                    dist = m.sqrt(((self.scan_points[0][0]- self.scan_points[-1][0])**2) + ((self.scan_points[0][0]- self.scan_points[-1][1])**2))
                    if dist > max_distance:
                        self.rough_clusters.append(cluster)
                        cluster = []
                        cluster.append(cur_point)
                        self.rough_clusters.append(cluster)
                    else:
                        self.rough_clusters[0].insert(0,cur_point)
                else:
                    cluster.append(cur_point)
                    self.rough_clusters.append(cluster)
                    cluster = []
    


    def init_split_to_lines(self):
        self.line_clusters = self.rough_clusters
        for elem in self.line_clusters:
            self.successfull_split.append(1)


    def split_to_lines(self):
        self.init_split_to_lines()
        while(1 in self.successfull_split):
            line_clusters = []
            successfull_split=[]
            for i in range(len(self.line_clusters)):
                cluster = self.line_clusters[i]
                if self.successfull_split[i]==1:
                    splitList1, splitList2 = split_and_merge(cluster)
                    if(splitList1==splitList2==0):
                        line_clusters.append(cluster)
                        successfull_split.append(0)
                    else:
                        line_clusters.append(splitList1)
                        successfull_split.append(1)
                        line_clusters.append(splitList2)
                        successfull_split.append(1)
                else:
                    line_clusters.append(cluster)
                    successfull_split.append(0)
            self.line_clusters = line_clusters
            self.successfull_split = successfull_split
    
    ##using first and last point
    # def calculate_corners_(self,min_split_value = .2, filter=None):
        
    #     corners = []
    #     for i in range(len(self.line_clusters)-1):
    #         if(len(self.line_clusters[i])>1 and len(self.line_clusters[i+1])>1):
    #             line1 = (self.line_clusters[i][0],self.line_clusters[i][1])
    #             line2 = (self.line_clusters[i+1][0],self.line_clusters[i+1][1])

    #             xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    #             ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    #             div = det(xdiff, ydiff)
                
    #             if div != 0:
    #                 d = (det(*line1), det(*line2))
    #                 x = det(d, xdiff) / div
    #                 y = det(d, ydiff) / div
                
    #                 if(filter!=None):
    #                     for point in self.line_clusters[i]:
    #                         dist = m.sqrt((point[0]-x)*(point[0]-x) + ((point[1]-y)*(point[1]-y)))
    #                         if (dist <min_split_value):
    #                             corners.append([x,y])
    #                             break
    #                 else:
    #                     corners.append([x,y])

        
    #     self.corners = corners

    def calculate_corners(self,min_split_value = .2, filter=None):
        corners = []
        for i in range(len(self.line_clusters)-1):
            if(len(self.line_clusters[i])>1 and len(self.line_clusters[i+1])>1):
                try:
                    b1, m1 = calculate_line(self.line_clusters[i])
                    b2, m2 = calculate_line(self.line_clusters[i+1])

                    x = (b1-b2) / (m2-m1)
                    y = m1*x+b1
                    if(filter!=None):
                        for point in self.line_clusters[i]:
                            dist = m.sqrt((point[0]-x)*(point[0]-x) + ((point[1]-y)*(point[1]-y)))
                            if (dist <min_split_value):
                                corners.append([x,y])
                                break
                    else:
                        corners.append([x,y])
                except:
                    print("corner calculation exception")
                    pass        
        self.corners = corners

    def get_corners(self, min_split_value =.2 ,filter = None):
        self.set_scan_points()
        self.large_split()
        self.split_to_lines()
        self.calculate_corners(min_split_value, filter)

        return self.corners