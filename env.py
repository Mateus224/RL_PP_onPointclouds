import numpy as np
import bpy
import mathutils
import datetime
import math
import random
import utilis.math.f  

import time


PI = math.pi


class Env(object):
    def __init__(self,args,config):
        self.transition=Transition(config)


    def reset(self):
        self.timeout=False 
        self.done=False
        self.scene = self.creat_scean()

    def creat_scean(self):

        pass









def step( sorted_actions, camera,scene):
    action = self.transition.check_transition(sorted_actions)
    if action not None:
        observation=self.transition.make_action(action)





    return 

def Q_guided_action(sorted_actions,step_size,action_state,limit_length,limit_width,camera,scene):
    for j in sorted_actions:

        ok= step(reset=2,
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

    ok= step(reset=2,
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


         
        


