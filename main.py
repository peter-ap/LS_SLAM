from os import error
from gs_classes import *

#Variables:

robot_path = "Data_2/encoder_data.txt"
laser_path = "Data_2/laserscanner_data.txt"
gt_path = "Data_2/ground_data.txt"
ITERATIONS = 200
EPSILON = 0.0001
SIMPLE_lEVENBERG_MARQUARDT = False

if __name__ == '__main__':
    # Create Graph object
    gs = Graph()
    gs.SIMPLE_lEVENBERG_MARQUARDT = SIMPLE_lEVENBERG_MARQUARDT

    # gs.read_from_data_files(robot_data = robot_path, laser_data=laser_path)
    gs.read_from_constraint_edges_files()
    gs.write_poses_to_file()
    fig = plt.figure(1)
    gs.plot_graph("seperate data unoptimized", show_constraints = True)
    print("initial error =" )
    print(gs.compute_global_error())
    fig2 = plt.figure(2)
    for i in range(ITERATIONS):
        dx = gs.linearize_and_solve()
        # show new estimated paths:
        plt.clf()
        gs.plot_graph("seperate data optimized", show_constraints = False)
        plt.draw()
        plt.pause(1)
        # calculate global error
        err = gs.get_error()
        print('---Global error--- iteration = ', i)
        print(err)


        #termination criterion
        if(np.amax(dx)) < EPSILON:
            print("converged with error =", err)
            break
    


    #plot true poses

    print('--------------------')
    print('End of optimization')
    fig2 = plt.figure(3)

    gs.plot_graph("seperate data optimized", show_constraints = True, error_path = robot_path, gt_path = gt_path )
    plt.show()