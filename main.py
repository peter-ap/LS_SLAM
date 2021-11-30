from gs_classes import *

#Variables:

robot_path = "Data_1/encoder_data.txt"
laser_path = "Data_1/laserscanner_data.txt"
iterations = 10
EPSILON = 0.0001


if __name__ == '__main__':
    # Create Graph object
    gs = Graph()

    # gs.read_from_data_files(robot_data = robot_path, laser_data=laser_path)
    gs.read_from_constraint_edges_files()

    fig = plt.figure(1)
    gs.plot_graph("seperate data unoptimized", show_constraints = False)
    plt.draw()
    for i in range(iterations):
        dx = gs.linearize_and_solve()
        # print(dx)
        #apply solution to state vector
        gs.update_vertices(dx)
        #show new estimated paths:
        plt.clf()
        gs.plot_graph("seperate data optimized", show_constraints = False)
        plt.draw()
        plt.pause(0.0001)
        #calculate global error
        err = gs.compute_global_error()
        print('---Global error---')
        print(err)

    
        #termination criterion
        if(np.amax(dx)) < EPSILON:
            print("converged with error =", err)
            break
    
    gs.plot_graph("seperate data optimized", show_constraints = False)
    plt.show()
