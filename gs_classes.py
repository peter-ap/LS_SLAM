import numpy as np
import matplotlib.pyplot as plt
import time 

from gs_functions import *

from SaM import *
import dataAssociation as da

import csv

class Vertex:
    """
    Pose Graph Vertex(Node) Class
    """
    def __init__(self, x, y, theta, id = None):
        """
        Vertex constructor
        """
        self.id = id       # ID of vertex
        self.x = x         # x coordinate position [m]
        self.y = y         # y coordinate position [m]
        self.theta = theta # yaw - rotation [radians]
        
class Landmark:
    """
    Pose Graph Landmark(Node) Class
    """
    def __init__(self, x, y, x_guess=None, y_guess=None,  id=None):

        self.id = id        # ID of landmark 
        self.x = x          # x coordinate position [m]
        self.y = y          # y coordinate position [m]
        self.x_guess = x_guess
        self.y_guess = y_guess # Using x_guess and y_guess to compensate for drift during data-association

class Edge:
    """
    Pose Graph Edge Class
    """
    def __init__(self, id_from, id_to, mean, inf_mat=None):
        """
        Edge constructor
        """
        if inf_mat is None:
            inf_mat = np.eye(3)  # if no information matrix is given, set identity matrix
                        
        self.id_from = id_from    # viewing frame of this edge
        self.id_to = id_to        # pose being observed from the viewing frame
        self.mean = mean          # Predicted Virtual measurement - z_ij
        self.inf_mat = inf_mat    # Edge Information matrix - omega_ij (~ 1 / covariance)

class LandmarkEdge:
    """"
    Pose Graph Edge to Landmark Class
    """
    def __init__(self, id_from, id_to, mean, inf_mat=None):
        """
        Edge constructor
        """
        self.id_from = id_from    # viewing frame of this edge
        self.id_to = id_to        # pose being observed from the viewing frame
        self.mean = mean          # Predicted Virtual measurement - z_ij
        if inf_mat is None:
            inf_mat = np.eye(2)
        self.inf_mat = inf_mat    # Edge Information matrix - omega_ij (~ 1 / covariance)



class Graph:
    """
    Graph class to build up back-end
    """
    def __init__(self, vertices = [], edges = [], landmarks = [], landmarkEdges = [], allVertices = [], SIMPLE_lEVENBERG_MARQUARDT=False, lambda_=1, error=0,  verbose = True):
        """
        Pose Graph constructor
        """
        self.vertices = vertices            # Pose graph vertices(nodes)
        self.edges = edges                  # Pose graph edges(constraints)
        self.landmarks = landmarks          # Pose graph landmarks(nodes)
        self.landmarkEdges = landmarkEdges  # Pose graph edges from position to landmark(constraints)
        self.H = []                         # Information matrix of the system (constraints contribution)
        self.SIMPLE_lEVENBERG_MARQUARDT = SIMPLE_lEVENBERG_MARQUARDT  
        self.lambda_ = lambda_             
        self.b = []                         # Coefficient vector
        self.verbose = verbose              # Show optimization steps
        self.allVertices = allVertices
        self.error = error

    #------------------------------------------------------------
    def read_from_data_files(self, robot_data, laser_data, encoder_inf=None,  laser_inf=None):
        
        #add robot poses to list
        vertices = np.loadtxt(robot_data, usecols = range(2,5)) 
        for i in range(len(vertices)):
            v = vertices[i]
            self.vertices.append(Vertex(id = i,
                                        x = v[0], 
                                        y = v[1], 
                                        theta = v[2]))

        #add edges between every robot pose to list
            if i!=0:
                mean = t2v(np.dot(inv(v2t(vertices[i-1])), v2t(v)))
                # e_mean = v - vertices[i-1]
                e_mean = mean
                self.edges.append(Edge(id_from = i-1, 
                                    id_to = i,
                                    mean = e_mean,
                                    inf_mat = encoder_inf))
        
        #get corners from laserdata
        l_d, num_readings,opening_angle = readData(laser_data)
        print("length of readings= ", len(l_d))
        for i in range(len(l_d)):
            laser = l_d[i]
            x,y=[],[]
            for elem in laser:
                x.append(elem[0])
                y.append(elem[1])

            ls = laserscan(x,y, 10, 0.2 , num_readings, opening_angle)
            corners = ls.get_corners(filter=1)
            #Match corners to add to list 
            
            for j in range(len(corners)):
                corner = corners[j]
                index, meas_world = da.dataAssociationPoints(self.landmarks, corner, self.vertices[i])
                meas = np.array([corner[0], corner[1]])

                if(index ==-1):
                    from_ = i
                    to_ = len(self.landmarks)
                    self.landmarkEdges.append(LandmarkEdge(from_, to_, meas, inf_mat = laser_inf))
                    self.landmarks.append(Landmark(x = meas_world[0],y = meas_world[1], x_guess= meas_world[0], y_guess=meas_world[1], id=to_))
                else:
                    from_ = i
                    to_ = index
                    self.landmarkEdges.append(LandmarkEdge(from_, to_, meas, inf_mat = laser_inf))
                    #update x_guess and y_guess for data association
                    self.landmarks[index].x_guess = meas_world[0]
                    self.landmarks[index].y_guess = meas_world[1]


    #---------------------------------------------------------------------------
    def read_from_constraint_edges_files(self):
        import scipy.io
        mat = scipy.io.loadmat('Data/dlr.mat')
        g = mat['g']
        edges = g['edges'][0][0]
        poses = g['x'][0][0]  #take care, the values are not packed!!
        existing_vertices=[]
        extisting_landmarks = []
        
        for e in edges:
            if(e['type']=='P'):
                v1_exists = False
                v2_exists = False
                fromId = int(e["fromIdx"])
                toId = int(e["toIdx"])
                id_from, id_to = -1, -1
                for v in existing_vertices:
                    if v[0]==fromId:
                        id_from = v[1]
                        v1_exists = True
                        break
                
                if v1_exists == False:
                    id_from = len(self.vertices)
                    existing_vertices.append([fromId, id_from])
                    self.vertices.append(Vertex(id = id_from,
                                                x = poses[fromId-1][0], 
                                                y = poses[fromId][0], 
                                                theta = poses[fromId+1][0]))

                for v in existing_vertices:
                    if v[0]==toId:
                        id_to = v[1]
                        v2_exists = True
                        break
                
                if v2_exists == False:
                    id_to = len(self.vertices)
                    existing_vertices.append([toId, id_to])

                    self.vertices.append(Vertex(id = id_to,
                                                x = poses[toId-1][0], 
                                                y = poses[toId][0], 
                                                theta = poses[toId+1][0]))
            
                #adding edge
                # meas = np.array([e['measurement'][0],e['measurement'][1],e['measurement'][2]])
                meas = e['measurement'][0]
                inf = e['information'][0]
                self.edges.append(Edge(id_from = id_from, 
                                    id_to = id_to,
                                    mean = meas,
                                    inf_mat = inf))            


            elif(e['type']=='L'):
                fromId = int(e["fromIdx"])
                toId = int(e["toIdx"])
                v1_exists = False
                l_exists = False
                for v in existing_vertices:
                    if v[0]==fromId:
                        v1_exists = True
                        id_from = v[1]
                        break
                if v1_exists==False:
                    id_from = len(self.vertices)
                    existing_vertices.append([fromId, id_from])
                    self.vertices.append(Vertex(id = id_from,
                                                x = poses[fromId-1][0], 
                                                y = poses[fromId+0][0], 
                                                theta = poses[fromId+1][0]))
                
                for l in extisting_landmarks:
                    if l[0]==toId:
                        l_exists = True
                        id_to = l[1]
                        break
            
                if l_exists==False:
                    id_to= len(self.landmarks)
                    extisting_landmarks.append([toId, id_to])
                    self.landmarks.append(Landmark( id = id_to,
                                                    x = poses[toId-1][0], 
                                                    y = poses[toId][0]))
                meas = e['measurement'][0]
                inf_mat = e['information'][0]
                self.landmarkEdges.append(LandmarkEdge(id_from, id_to, mean=meas, inf_mat = inf_mat))
    
    
    
    
    def linearize_and_solve(self):

        #linearize and add contributions/fill H and -b vector
        self.linearize_and_add()
        #Solve the linear system:
        # Keep first node fixed
        self.H[:3,:3] += np.eye(3)*1000

        if(self.SIMPLE_lEVENBERG_MARQUARDT==False):
            dX = solve(H = self.H, b = self.b, sparse_solve=True)
            #apply solution to state vector
            self.update_vertices(dX)
            self.error = self.compute_global_error()
            return dX

        else:
            self.H+=np.eye(self.H.shape[0])*self.lambda_
            dX = solve(H = self.H, b = self.b, sparse_solve=True)
            #apply solution to state vector
            self.update_vertices_SLM(dX)
            self.error = self.compute_global_error()
            return dX

    def update_vertices(self, dx):
        robot_update, landmark_update  = np.split(dx, [3*len(self.vertices)]) 
        robot_update = robot_update.reshape(len(self.vertices), 3)
        landmark_update = landmark_update.reshape(len(self.landmarks), 2)
        
        #update robot poses:
        for i in range(len(robot_update)):
            self.vertices[i].x     += robot_update[i,0]
            self.vertices[i].y     += robot_update[i,1]
            self.vertices[i].theta += robot_update[i,2]


        #update landmark poses:
        for i in range(len(landmark_update)):
            self.landmarks[i].x += landmark_update[i,0]
            self.landmarks[i].y += landmark_update[i,1]

    def update_vertices_SLM(self, dx):
        robot_update, landmark_update  = np.split(dx, [3*len(self.vertices)]) 
        robot_update = robot_update.reshape(len(self.vertices), 3)
        landmark_update = landmark_update.reshape(len(self.landmarks), 2)
        
        prev_error = self.compute_global_error()

        temp_vertices = np.copy(self.vertices)
        temp_landmarks = np.copy(self.landmarks)

        #update robot poses:
        for i in range(len(robot_update)):
            self.vertices[i].x     += robot_update[i,0]
            self.vertices[i].y     += robot_update[i,1]
            self.vertices[i].theta += robot_update[i,2]


        #update landmark poses:
        for i in range(len(landmark_update)):
            self.landmarks[i].x += landmark_update[i,0]
            self.landmarks[i].y += landmark_update[i,1]

        error = self.compute_global_error()

        if(prev_error < error):
            self.vertices = temp_vertices
            self.landmarks = temp_landmarks
            self.lambda_ *=2
        else:
            self.lambda_/=2


    def linearize_and_add(self):
        # allocate sparse matrix H and vector b
        l = 3*len(self.vertices) + 2*len(self.landmarks)
        self.H = np.zeros((l,l),dtype = np.float64)
        self.b = np.zeros((l,1), dtype=np.float64)

        # linearize and build the system
        # POSE-POSE constraint
        for e in self.edges:
             # get indexes of connected nodes
            i = e.id_from
            j = e.id_to
            omega_ij = e.inf_mat
            x_i = np.array([self.vertices[i].x, self.vertices[i].y, self.vertices[i].theta])
            x_j = np.array([self.vertices[j].x, self.vertices[j].y, self.vertices[j].theta])
            z_ij = e.mean
            # Computing the error and the Jacobians
            # e the error vector
            # A Jacobian wrt x_i
            # B Jacobian wrt x_j
            e, A_ij, B_ij = linearize_pose_pose_constraint(x_i, x_j, z_ij)

            #compute contributions to H an -b
            b_i  = -np.dot(np.dot(e.T, omega_ij), A_ij).T
            b_j  = -np.dot(np.dot(e.T, omega_ij), B_ij).T

            H_ii = np.dot(np.dot(A_ij.T, omega_ij), A_ij)
            H_ij = np.dot(np.dot(A_ij.T, omega_ij), B_ij)
            H_ji = H_ij.T
            H_jj = np.dot(np.dot(B_ij.T, omega_ij), B_ij)

            #Add to H and b
            self.H[3*i:3*(i+1), 3*i:3*(i+1)] += H_ii
            self.H[3*i:3*(i+1), 3*j:3*(j+1)] += H_ij
            self.H[3*j:3*(j+1), 3*i:3*(i+1)] += H_ji
            self.H[3*j:3*(j+1), 3*j:3*(j+1)] += H_jj

            self.b[3*i:3*(i+1)] +=b_i.reshape(3,1)
            self.b[3*j:3*(j+1)] +=b_j.reshape(3,1)

        # POSE-LANDMARK constraints
        for l in self.landmarkEdges:
            i = l.id_from
            j_ = l.id_to
            # j = j_ + (len(self.vertices)-1) #length of the vertices -1 is the last ID of the robot poses

            omega_ij = l.inf_mat
            x_i = np.array([self.vertices[i].x, self.vertices[i].y, self.vertices[i].theta])
            x_j = np.array([self.landmarks[j_].x, self.landmarks[j_].y])
            z_ij = l.mean

            e, A_ij, B_ij = linearize_pose_landmark_constraint(x_i, x_j, z_ij)

            #compute contributions to H an -b
            b_i = -np.dot(np.dot(e.T, omega_ij), A_ij).T
            b_j = -np.dot(np.dot(e.T, omega_ij), B_ij).T


            H_ii = np.dot(np.dot(A_ij.T,omega_ij),A_ij)
            H_ij = np.dot(np.dot(A_ij.T, omega_ij), B_ij)
            H_ji = H_ij.T
            H_jj = np.dot(np.dot(B_ij.T, omega_ij), B_ij)


            #adding H and b contributions to matrices
            j = 3*len(self.vertices) + 2*j_  #index of landmark keeping in mind the size.
            _j = 3*len(self.vertices) + 2*(j_+1)

            self.H[3*i:3*(i+1), 3*i:3*(i+1)]    += H_ii
            self.H[3*i:3*(i+1), j:_j]           += H_ij
            self.H[j:_j, 3*i:3*(i+1)]           += H_ji
            self.H[j:_j, j:_j]                  += H_jj

            self.b[3*i:3*(i+1)]     +=b_i.reshape(3,1)
            self.b[j:_j]            +=b_j.reshape(2,1)

    def get_error(self):
        return self.error

    def compute_global_error(self):
        #return value 
        Fx = 0

        #loop over pose-pose edges
        for i in range(len(self.edges)):
            e = self.edges[i]
            robot_i = self.vertices[e.id_from]   #pose from 
            x_i = np.array([robot_i.x, robot_i.y, robot_i.theta])
            robot_j = self.vertices[e.id_to]     #pose to 
            x_j = np.array([robot_j.x, robot_j.y, robot_j.theta])
            z_ij = e.mean                    #measurement
            inf_mat = e.inf_mat              #information matrix 

            es=  t2v(np.dot(inv(v2t(z_ij)),  np.dot(inv(v2t(x_i)),v2t(x_j))))
            Fx = Fx + np.dot(es.T, np.dot(inf_mat, es))

        #loop over pose-landmark edges
        for i in range(len(self.landmarkEdges)):
            l = self.landmarkEdges[i]
            robot_i = self.vertices[l.id_from]   #pose from 
            x_i = np.array([robot_i.x, robot_i.y, robot_i.theta])
            landmark_j = self.landmarks[l.id_to]    #landmark to 
            xt_j = np.array([[landmark_j.x], [landmark_j.y], [1]]) #homogenous vector
            # z_ij = np.array([[l.mean[0]],[l.mean[1]]])
            z_ij = l.mean.reshape(2,1)
            inf_mat = l.inf_mat     

            # make homogeneous
            zt_ij = np.vstack([z_ij, 1])

            es = np.dot(inv(v2t(x_i)), xt_j) - zt_ij
            es = es[:2]
            Fx = Fx + np.dot(es.T, np.dot(inf_mat, es))

        return Fx
        


    def plot_graph(self,title=None, show_constraints= False, error_path = None, gt_path=None):
        node_x = np.array([])
        node_y = np.array([])
        mark_x = np.array([])
        mark_y = np.array([])

        for v in self.vertices:
            node_x = np.append(node_x, v.x)
            node_y = np.append(node_y, v.y)

        for l in self.landmarks:
            mark_x = np.append(mark_x, l.x)
            mark_y = np.append(mark_y, l.y)

        plt.scatter(node_x, node_y, s = 4, color = 'b', label = 'robot poses')
        plt.scatter(mark_x, mark_y, s = 4, color = 'black', label = 'landmark poses')
    
        if show_constraints==True:
            for e in self.edges:
                plt.plot([self.vertices[e.id_from].x, self.vertices[e.id_to].x],
                         [self.vertices[e.id_from].y, self.vertices[e.id_to].y],
                         color = 'red', lw = 0.5,)
        

            for l in self.landmarkEdges:
                plt.plot([self.vertices[l.id_from].x, self.landmarks[l.id_to].x],
                [self.vertices[l.id_from].y, self.landmarks[l.id_to].y],
                color = 'green', lw = 0.5,)

        if gt_path!=None:
            vertices = np.loadtxt(gt_path, usecols = range(2,5)) 
            node_x = np.array([])
            node_y = np.array([])
            for v in vertices:
                node_x = np.append(node_x, v[0])
                node_y = np.append(node_y, v[1])           
            plt.scatter(node_x, node_y, s = 4, color = 'y', label = 'GT poses')

        if error_path!=None:
            vert = np.loadtxt(error_path, usecols = range(2,5)) 
            node_x = np.array([])
            node_y = np.array([])
            for v in vert:
                node_x = np.append(node_x, v[0])
                node_y = np.append(node_y, v[1])           
            plt.scatter(node_x, node_y, s = 4, color = 'r', label = 'error poses')

        plt.title(title)
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.legend()

    def write_poses_to_file(self):
        
        f= open('poses','w')
        writer = csv.writer(f)
        for v in self.vertices:
            row = [str(v.x), str(v.y), str(v.theta)]
            writer.writerow(row)
        
