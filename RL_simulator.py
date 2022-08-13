


import numpy as np
import bpy

import mathutils
import open3d as o3d
import datetime

import math
import random 


import time


PI = math.pi

def read_verts(mesh):
        mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
        mesh.vertices.foreach_get("co", mverts_co)
        return np.reshape(mverts_co, (len(mesh.vertices), 3))      


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




def transition(step_size,step_choice,action_state,limit_length,limit_width,camera,scene):
        
        pointcloud=np.array([0,0,0])
        
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
            return 0,pointcloud
        if(abs(simulated_pos_y)>abs(limit_width)):
            #print("Width limit" +str(limit_width)+" exceded at y="+str(simulated_pos_y))
            return 0,pointcloud

        if( (abs(simulated_pos_x)<=abs(limit_length)) and (abs(simulated_pos_y)<=abs(limit_width))):
            camera.location.x =camera.location.x+ step_size* final_step[1]
            camera.location.y =camera.location.y+ step_size* final_step[0] 
            camera.location.z =camera.location.z+ final_step[2] * step_size

            camera.rotation_euler[0]=camera.rotation_euler[0]+action_vector[3] 
            camera.rotation_euler[1]=camera.rotation_euler[1]+action_vector[4]
            camera.rotation_euler[2]=camera.rotation_euler[2]+action_vector[5]

            bpy.ops.blensor.scan()

            pointcloud=append_scan(camera,scene)

            return 1,pointcloud

def append_scan(camera,scene):
    obs_scan = []
    for ob in scene.objects:
        # whatever objects you want to join...
        if ob.type == 'MESH' and ob.name.startswith('Scan'):
            obs_scan.append(ob)
            
    ctx = bpy.context.copy()
    ctx['active_object'] = obs_scan[0]
    ctx['selected_objects'] = obs_scan
    ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs_scan]
    bpy.ops.object.join(ctx)

   

    pointcloud=read_verts(obs_scan[0].data)

    print(pointcloud.shape)

   

    pointcloud=pointcloud[~np.isnan(pointcloud).any(axis=1), :]

    stupid_cloud=pointcloud[np.isnan(pointcloud).any(axis=1), :]

    print(stupid_cloud.shape)
    print(stupid_cloud)

    #obs_scan[0].data=pointcloud

  
    

    return pointcloud

def q_guided_transition(sorted_actions,step_size,action_state,limit_length,limit_width,camera,scene):
    ok=0
    for j in sorted_actions:

        ok,pointcloud= transition(
                    step_size=step_size,
                    step_choice=j,
                    action_state=action_state,
                    limit_length=limit_length,
                    limit_width=limit_width,
                    camera=camera,
                    scene=scene)
        if (ok==1):
            print("Action "+str(i)+" :"+str(j))
            return ok,pointcloud
            
    return ok,pointcloud

def q_episode(action_state,step_size,nr_steps,limit_length,limit_width,camera,scene,on=0):
    pointcloud=np.array([0,0,0])
    actions=range(action_state.shape[0])

    if (on==1):
        for i in range(nr_steps):
            
            sorted_actions=np.random.permutation(actions)

            ok,pointcloud=q_guided_transition(sorted_actions=sorted_actions,
                            step_size=step_size,
                            action_state=action_state,
                            limit_length=limit_length,
                            limit_width=limit_width,
                            camera=camera,
                            scene=scene)

    return pointcloud

def simple_episode(step_size,nr_steps,limit_length,limit_width,camera,scene,on=0):
    pointcloud=np.array([0,0,0])
    if (on==1):
        for i in range(nr_steps):
            
           

                ok,pointcloud = transition(
                                            step_size=step_size,
                                            step_choice=step_choice,
                                            action_state=action_state,
                                            limit_length=limit_length,
                                            limit_width=limit_width,
                                            camera=camera,
                                            scene=scene)


                if (ok==0):
                    break

    return pointcloud


def delete_meshes(scene,on=0):
    if (on==1):
        meshes = set()
        # Get objects in the collection if they are meshes
        for obj in [o for o in scene.objects if (o.type == 'MESH' and not(o.name.startswith('Plane')))]:
            meshes.add( obj.data )
            bpy.data.objects.remove( obj)

        # Look at meshes that are orphean after objects removal
        for mesh in [m for m in meshes if m.users == 0]:
            # Delete the meshes
            bpy.data.meshes.remove( mesh )

def load_object_meshes(nr_objects,max_nr_objects,file_names,on=0):
        if(on==1) :
            object_array=np.random.randint(4, size=nr_objects)
            for i in object_array:
                imported_object = bpy.ops.import_scene.obj(filepath=str(file_names[i]))

        

def check_nr_objects(nr_objects,max_nr_objects):
    if(nr_objects >(max_nr_objects)):
            nr_objects= max_nr_objects
    return nr_objects

def create_obj_list(scene):
    obs = []       
    for ob in scene.objects:
        if ob.type == 'MESH' and not(ob.name.startswith('Plane')):
            #ob.rotation_euler[2]=(rotation_high-rotation_low) * (PI/ 180.0)* np.random.random_sample() + rotation_low * (PI/ 180.0)
            obs.append(ob)

    return obs

def configure_camera(camera,xres,yres,kinect_max_dist,kinect_min_dist=0,on=0):
    if (on==1):
        camera.rotation_mode = 'XYZ'
        camera.rotation_euler[0]=90 * (PI/ 180.0)
        camera.rotation_euler[1]=0 * (PI/ 180.0)



        camera.scan_type='kinect'

        camera.kinect_xres=20
        camera.kinect_yres=20
        camera.kinect_min_dist=kinect_min_dist
        camera.local_coordinates=False



def place_resize_objects(nr_objects,max_nr_objects,x_selection,y_selection,board_matrix,obs,L_square,W_square,H_square,limit_length,limit_width,limit_height,rotation_high,rotation_low,on=0):
    
    if(on==1):
        min_dimension=min([L_square,W_square,H_square])
        objects_placed=0
        i=0
        while(objects_placed<nr_objects) and (i<max_nr_objects):
                if board_matrix[x_selection[i],y_selection[i]]==0:
                    objects_placed=objects_placed+1
                    board_matrix[x_selection[i],y_selection[i]]=1

                    dimensions=np.array([obs[i].dimensions.x,obs[i].dimensions.y,obs[i].dimensions.z])

                    max_obj_dimension=max([obs[i].dimensions.x,obs[i].dimensions.y,obs[i].dimensions.z])
                    

                    diag_object=np.sqrt( (obs[i].dimensions.x)*(obs[i].dimensions.x)+
                                                (obs[i].dimensions.z)*(obs[i].dimensions.z)) *min_dimension/max_obj_dimension


                    #######################################################3
                    #Proportional with size of square
                    
                    obs[i].scale[0]=min_dimension/max_obj_dimension* min_dimension/diag_object *0.9

                    obs[i].scale[2]=min_dimension/max_obj_dimension* min_dimension/diag_object *0.9
                    obs[i].scale[1]=min_dimension/max_obj_dimension* min_dimension/diag_object *0.9


                    #################################################33
                    # Approximately same size as square

                    # obs[i].scale[0]=min_dimension/obs[i].dimensions.x

                    # obs[i].scale[2]=min_dimension/obs[i].dimensions.z
                    # obs[i].scale[1]=min_dimension/obs[i].dimensions.y

                    # obs[i].scale[0]=min_dimension/obs[i].dimensions.x* min_dimension/diag_object *0.9

                    # obs[i].scale[2]=min_dimension/obs[i].dimensions.z* min_dimension/diag_object *0.9
                    # obs[i].scale[1]=min_dimension/obs[i].dimensions.y* min_dimension/diag_object *0.9
                                    
                   ##############################################################

                    x_place=y_selection[i]*L_square
                    y_place=x_selection[i]*W_square

                    #print(str(x_selection[i])+" "+str(y_selection[i]))
                

                    obs[i].location[0]=x_place -limit_length/2
                    obs[i].location[1]=y_place -limit_width/2


                    obs[i].rotation_euler[2]=(rotation_high-rotation_low) * (PI/ 180.0)* np.random.random_sample() + rotation_low * (PI/ 180.0)

                i=i+1
        return i




def place_camera(i,max_nr_objects,board_matrix,x_selection,y_selection,L_square,W_square,limit_length,limit_width,rotation_high,rotation_low,camera,on=0):
    if (on==1):
        while(i<max_nr_objects+1):
            if board_matrix[x_selection[i],y_selection[i]]==0:
                x_place=y_selection[i]*L_square
                y_place=x_selection[i]*W_square
                camera.location[0]=x_place -limit_length/2
                camera.location[1]=y_place -limit_width/2

                camera.rotation_euler[2]=(rotation_high-rotation_low) * (PI/ 180.0)* np.random.random_sample() + rotation_low * (PI/ 180.0)
                i=i+1

def delete_scans(on=0):
    if (on==1):
        bpy.ops.blensor.delete_scans()
        

def save_pcd_file(path_save_pcd_folder,on=0):    
    if (on==1): 
        now = datetime.datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")

        path_pcd=path_save_pcd_folder+"out_"+str(current_time)+".pcd"  

        f=open(path_pcd,"w") #File location to modify
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
        f=open(path_pcd,"r+") #File location to modify
        new_data = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" %(i,i))
        old = f.read()
        f.seek(0)
        f.write(new_data)
        f.write(old)   

def save_pcd_o3d(path_save_pcd_folder,pointcloud,on=0):
    if (on==1):

        now = datetime.datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")

        path_pcd=path_save_pcd_folder+"out_"+str(current_time)+".pcd" 


        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(pointcloud)

        o3d.io.write_point_cloud(path_pcd, save_pcd)

########################################################################################################################3
#########################################################################################################################3

angle_camera= 30
action_state=np.array([[1,0,0,0,0,-angle_camera* (PI/ 180.0)],
                       [-1,0,0,0,0,-angle_camera* (PI/ 180.0)],
                       [0,-1,0,0,0,-angle_camera* (PI/ 180.0)],
                       [0,1,0,0,0,-angle_camera* (PI/ 180.0)],

                       [1,0,0,0,0,0* (PI/ 180.0)],
                       [-1,0,0,0,0,0* (-PI/ 180.0)],
                       [0,-1,0,0,0,0* (-PI/ 180.0)],
                       [0,1,0,0,0,0* (-PI/ 180.0)],

                       [1,0,0,0,0,angle_camera* (PI/ 180.0)],
                       [-1,0,0,0,0,angle_camera* (PI/ 180.0)],
                       [0,-1,0,0,0,angle_camera* (PI/ 180.0)],
                       [0,1,0,0,0,angle_camera* (PI/ 180.0)]])

# action_state=np.array([[1,0,0,0,0,0], #Up
#                        [-1,0,0,0,0,0], #Down
#                        [0,-1,0,0,0,0], #Left
#                        [0,1,0,0,0,0], #Right

#                        [0,0,0,0,0,-angle_camera* (PI/ 180.0)], #Rotate right
#                        [0,0,0,0,0,angle_camera* (PI/ 180.0)]]) #Rotate left



file_names=['/home/alex-pop/Desktop/Doctorat/Blender_views/Box/Separate_objects/Cube.obj',
            '/home/alex-pop/Desktop/Doctorat/Blender_views/Box/Separate_objects/Long_Cube.obj',
            '/home/alex-pop/Desktop/Doctorat/Blender_views/Box/Separate_objects/Edge_Cube.obj',
            '/home/alex-pop/Desktop/Doctorat/Blender_views/Box/Separate_objects/Pyramid.obj']

path_save_pcd_folder="/home/alex-pop/Desktop/Doctorat/Blender_views/Sim_output/"


#Configuration in Blensor

nr_rays_x_axis=20
nr_rays_y_axis=20

step_choice=2
nr_steps=1
limit_length=10
limit_width=10
limit_height=10
dim_reduction=10

rotation_high=180
rotation_low=-180

nr_objects=99

conf_camera=0
del_previous_meshes=0
load_meshes=0
place_rotate_obj=0
place_rotate_camera=0
delete_previous_scans=0
save_pcd=1

q_epi=1
s_epi=1

###############################################################

scene = bpy.context.scene
camera = bpy.data.objects['Camera']


delete_meshes(scene=scene,on=del_previous_meshes)

max_nr_objects=dim_reduction*dim_reduction-1
nr_objects=check_nr_objects(nr_objects,max_nr_objects)

board_matrix=np.zeros((dim_reduction,dim_reduction))

L_square=limit_length/dim_reduction
W_square=limit_width/dim_reduction
H_square=limit_height/dim_reduction
min_dimension=min([L_square,W_square,H_square])

configure_camera(camera=camera,xres=nr_rays_x_axis,yres=nr_rays_y_axis,kinect_min_dist=0,kinect_max_dist=5*min_dimension,on=conf_camera)

load_object_meshes(nr_objects,max_nr_objects,file_names,on=load_meshes)


obs=create_obj_list(scene)


place_selection= np.random.permutation(dim_reduction*dim_reduction)
x_selection,y_selection=np.divmod(place_selection, dim_reduction)


i=0
i=place_resize_objects(nr_objects,max_nr_objects,x_selection,y_selection,board_matrix,obs,L_square,W_square,H_square,limit_length,limit_width,limit_height,rotation_high,rotation_low,on=place_rotate_obj)

if(place_rotate_camera==place_rotate_obj):
    place_camera(i,max_nr_objects,board_matrix,x_selection,y_selection,L_square,W_square,limit_length,limit_width,rotation_high,rotation_low,camera,on=place_rotate_camera)

step_size=0.25 * min_dimension


delete_scans(on=delete_previous_scans)


pointcloud=np.array([0,0,0])

if ( (q_epi!=s_epi) and (q_epi==1)):
    pointcloud =q_episode(action_state,step_size,nr_steps,limit_length,limit_width,camera,scene,on=q_epi)

elif (s_epi==1):
    pointcloud =simple_episode(step_size,nr_steps,limit_length,limit_width,camera,scene,on=s_epi)
# if(q_epi!=s_epi):
#     pointcloud =q_episode(action_state,step_size,nr_steps,limit_length,limit_width,camera,scene,on=q_epi)

#     pointcloud =simple_episode(step_size,nr_steps,limit_length,limit_width,camera,scene,on=s_epi)
                
    

#     # ok,pointcloud = transition(
#     #             step_size=step_size,
#     #             step_choice=step_choice,
#     #             action_state=action_state,
#     #             limit_length=limit_length,
#     #             limit_width=limit_width,
#     #             camera=camera,
#     #             scene=scene)


#     # if (ok==0):
#     #     print("Cannot make move")
#     #     break


#print(pointcloud.shape)
#print(pointcloud)
save_pcd_o3d(path_save_pcd_folder=path_save_pcd_folder,pointcloud=pointcloud,on=save_pcd)









         
        


