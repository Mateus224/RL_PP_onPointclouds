import numpy as np
import bpy
import mathutils
import datetime
import math
import random 

import time


PI = math.pi


def matrix_from_angle(basis, angle):
    """Compute passive rotation matrix from rotation about basis vector.
    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)
    angle : float
        Rotation angle
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if basis == 0:
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, c, s],
                      [0.0, -s, c]])
    elif basis == 1:
        R = np.array([[c, 0.0, -s],
                      [0.0, 1.0, 0.0],
                      [s, 0.0, c]])
    elif basis == 2:
        R = np.array([[c, s, 0.0],
                      [-s, c, 0.0],
                      [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return R

passive_matrix_from_angle = matrix_from_angle

def matrix_from_euler_xyz(e):
    """Compute passive rotation matrix from intrinsic xyz Tait-Bryan angles.
    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes (intrinsic rotations)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = passive_matrix_from_angle(0, alpha).dot(
        passive_matrix_from_angle(1, beta)).dot(
        passive_matrix_from_angle(2, gamma))
    return R




def Action(reset,step_size,step_choice,action_state,limit_length,limit_width,camera,scene):
    
        if(reset==1):
            camera.rotation_euler[2]=0 * (PI/ 180.0)

        action_vector=action_state[step_choice-1,:]

        

        rotation_x=matrix_from_angle(0, camera.rotation_euler[0]+action_vector[3])
        rotation_y=matrix_from_angle(1, camera.rotation_euler[1]+action_vector[4])
        rotation_z=matrix_from_angle(2, camera.rotation_euler[2]+action_vector[5])
        

        #R_t= np.matmul(np.matmul(matrix_from_angle(0, camera.rotation_euler[0]),
        #                       matrix_from_angle(1, camera.rotation_euler[1])),
        #                       matrix_from_angle(2, camera.rotation_euler[2]))

        R_t= np.matmul(matrix_from_angle(1, camera.rotation_euler[1]),
                    matrix_from_angle(2, camera.rotation_euler[2]))
                            

        

        final_step=np.matmul(R_t,action_vector[0:3])
        #final_step=np.matmul(rotation_z,step_vector)

        simulated_pos_x=camera.location.x+ step_size* final_step[1]
        simulated_pos_y=camera.location.y+ step_size* final_step[0]
        simulated_pos_z=camera.location.z+ step_size* final_step[2]

        #print(str(simulated_pos_x)+" "+str(simulated_pos_y))

        if(abs(simulated_pos_x)>abs(limit_length)):
            #print("Length limit" +str(limit_length)+" exceded at x="+str(simulated_pos_x))
            return 0
        if(abs(simulated_pos_y)>abs(limit_width)):
            #print("Width limit" +str(limit_width)+" exceded at y="+str(simulated_pos_y))
            return 0

        if( (abs(simulated_pos_x)<=abs(limit_length)) and (abs(simulated_pos_y)<=abs(limit_width))):
            camera.location.x =camera.location.x+ step_size* final_step[1]
            camera.location.y =camera.location.y+ step_size* final_step[0] 
            camera.location.z =camera.location.z+ final_step[2] * step_size

            camera.rotation_euler[0]=camera.rotation_euler[0]+action_vector[3] 
            camera.rotation_euler[1]=camera.rotation_euler[1]+action_vector[4]
            camera.rotation_euler[2]=camera.rotation_euler[2]+action_vector[5]

            bpy.ops.blensor.scan()

            obs = []
            for ob in scene.objects:
                # whatever objects you want to join...
                if ob.type == 'MESH' and ob.name.startswith('Scan'):
                    obs.append(ob)
            ctx = bpy.context.copy()
            ctx['active_object'] = obs[0]
            ctx['selected_objects'] = obs
            ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
            bpy.ops.object.join(ctx)


            return 1

def Q_guided_action(sorted_actions,step_size,action_state,limit_length,limit_width,camera,scene):
    ok=0
    for j in sorted_actions:

        ok= Action(reset=2,
                    step_size=step_size,
                    step_choice=j,
                    action_state=action_state,
                    limit_length=limit_length,
                    limit_width=limit_width,
                    camera=camera,
                    scene=scene)
        if (ok==1):
            print("Action "+str(i)+" :"+str(j))
            return ok
            
    return ok


scene = bpy.context.scene
camera = bpy.data.objects['Camera']
camera.rotation_mode = 'XYZ'
camera.rotation_euler[0]=90 * (PI/ 180.0)
camera.rotation_euler[1]=0 * (PI/ 180.0)



# bpy.data.objects['Camera'].select=False
bpy.data.objects['Camera'].select=True

bpy.context.object.scan_type='kinect'
bpy.context.object.save_scan=True

# bpy.context.object.location[0]=-1
# bpy.context.object.location[1]=-2
# bpy.context.object.location[2]=0

bpy.context.object.kinect_xres=20
bpy.context.object.kinect_yres=20
bpy.context.object.local_coordinates=False

reset=2
angle_camera= 30
# action_state=np.array([[1,0,0,0,0,-angle_camera* (PI/ 180.0)],
#                        [-1,0,0,0,0,-angle_camera* (PI/ 180.0)],
#                        [0,-1,0,0,0,-angle_camera* (PI/ 180.0)],
#                        [0,1,0,0,0,-angle_camera* (PI/ 180.0)],

#                        [1,0,0,0,0,0* (PI/ 180.0)],
#                        [-1,0,0,0,0,0* (-PI/ 180.0)],
#                        [0,-1,0,0,0,0* (-PI/ 180.0)],
#                        [0,1,0,0,0,0* (-PI/ 180.0)],

#                        [1,0,0,0,0,angle_camera* (PI/ 180.0)],
#                        [-1,0,0,0,0,angle_camera* (PI/ 180.0)],
#                        [0,-1,0,0,0,angle_camera* (PI/ 180.0)],
#                        [0,1,0,0,0,angle_camera* (PI/ 180.0)]])

action_state=np.array([[1,0,0,0,0,0], #Up
                       [-1,0,0,0,0,0], #Down
                       [0,-1,0,0,0,0], #Left
                       [0,1,0,0,0,0], #Right

                       [0,0,0,0,0,-angle_camera* (PI/ 180.0)], #Rotate right
                       [0,0,0,0,0,angle_camera* (PI/ 180.0)]]) #Rotate left

# all_step_vector=np.array([[1,0,0],[-1,0,0],[0,-1,0],[0,1,0]])

# angles=np.array([0,0,angle_camera])* (-PI/ 180.0)

###############################################################
#Configuration in Blensor

step_size=0.2
step_choice=4
nr_steps=1
limit_length=3
limit_width=3



actions=np.array(range(1,13))



start_time = time.time()

for i in range(nr_steps):
    #bpy.ops.blensor.delete_scans()
    sorted_actions=np.random.permutation(actions)

    # ok=Q_guided_action(sorted_actions=sorted_actions,
    #                    step_size=step_size,
    #                    action_state=action_state,
    #                    limit_length=limit_length,
    #                    limit_width=limit_width,
    #                    camera=camera)

               
    # if (ok==0):
    #     print("Cannot make any move")
    #     break

    ok= Action(reset=2,
                step_size=step_size,
                step_choice=step_choice,
                action_state=action_state,
                limit_length=limit_length,
                limit_width=limit_width,
                camera=camera,
                scene=scene)


    

now = datetime.datetime.now()
current_time = now.strftime("%Y%m%d%H%M%S")
end_time = time.time()

print(end_time-start_time)

    #path_file="/home/alex-pop/Desktop/DAte/cl_pcd_"+"box"+str(Object_nr)+"_"+str(dist_choice)+"_"+str(angle_choice)+"_"+ str(corner_choice)+"_"+str(current_time)+"_"+str(t)+".pcd"
path_file="/home/alex-pop/Desktop/Doctorat/Blender_views/Sim_output/out_"+str(current_time)+".pcd"


f=open(path_file,"w") #File location to modify
i = 0; #Storage point cloud number
for item in bpy.data.objects:
    if item.type == 'MESH' and item.name.startswith('Scan'):
        print('write once')
        for sp in item.data.vertices:
            #print('X=%+#5.3f\tY=%+#5.3f\tZ=%+#5.3f' % (sp.co[0], sp.co[1],sp.co[2]));
            if(  (np.isnan(sp.co[0])==0) and(np.isnan(sp.co[1])==0) and (np.isnan(sp.co[2])==0) ):
                string_out='%#5.3f\t%#5.3f\t%#5.3f \n' % (sp.co[0], sp.co[1],sp.co[2])
                i = i+1
                f.write(string_out);  
f.close()          
f=open(path_file,"r+") #File location to modify
new_data = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" %(i,i))
old = f.read()
f.seek(0)
f.write(new_data)
f.write(old)


         
        


