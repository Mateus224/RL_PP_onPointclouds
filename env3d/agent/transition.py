import json
import numpy as np
from lut_actions import Actions
from importlib import reload
import math_f
import bpy
reload(math_f)




class Transition():
    def __init__(self, config, scene):
        json_env_shape=config.get('ENV','shape')
        list_env_shape=json.loads(json_env_shape)
        self.env_shape=np.array(list_env_shape)
        self.step_size= config.getint('Agent01', 'step_size')
        actions=Actions(0.1)
        self.action_set=actions.ACTIONS2D

        self.x = self.env_shape[0]
        self.y = self.env_shape[1]
        self.z = self.env_shape[2]
        self.scene=scene
        self.pointcloud = np.zeros((10,3))

    def legal_transition(self,actions):
        actions_arg_sorted=np.argsort(-1*actions)
        for i in actions_arg_sorted:
            if self.check_transition(actions_arg_sorted[i]):
                return actions_arg_sorted[i]


    def check_transition(self, action):
        return True

    def read_verts(self, mesh):
        mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
        mesh.vertices.foreach_get("co", mverts_co)
        return np.reshape(mverts_co, (len(mesh.vertices), 3))

    def make_action(self, action):
        
        action_vector=self.action_set[action]
        

        

        rotation_x=math_f.matrix_from_angle(0, self.scene.camera.rotation_euler[0]+action_vector[3])
        rotation_y=math_f.matrix_from_angle(1, self.scene.camera.rotation_euler[1]+action_vector[4])
        rotation_z=math_f.matrix_from_angle(2, self.scene.camera.rotation_euler[2]+action_vector[5])
        

        #R_t= np.matmul(np.matmul(matrix_from_angle(0, camera.rotation_euler[0]),
        #                       matrix_from_angle(1, camera.rotation_euler[1])),
        #                       matrix_from_angle(2, camera.rotation_euler[2]))

        R_t= np.matmul(math_f.matrix_from_angle(1, self.scene.camera.rotation_euler[1]),
                    math_f.matrix_from_angle(2, self.scene.camera.rotation_euler[2]))
                            

        

        final_step=np.matmul(R_t,action_vector[0:3])
        #final_step=np.matmul(rotation_z,step_vector)
        step_size=1

        simulated_pos_x=self.scene.camera.location.x+ step_size* final_step[1]
        simulated_pos_y=self.scene.camera.location.y+ step_size* final_step[0]
        simulated_pos_z=self.scene.camera.location.z+ step_size* final_step[2]

        #print(str(simulated_pos_x)+" "+str(simulated_pos_y))

        #if(abs(simulated_pos_x)>abs(limit_length)):
            #print("Length limit" +str(limit_length)+" exceded at x="+str(simulated_pos_x))
            #return 0
        #if(abs(simulated_pos_y)>abs(limit_width)):
            #print("Width limit" +str(limit_width)+" exceded at y="+str(simulated_pos_y))
        #    return 0

        #if( (abs(simulated_pos_x)<=abs(limit_length)) and (abs(simulated_pos_y)<=abs(limit_width))):
        self.scene.camera.location.x =self.scene.camera.location.x+ step_size* final_step[1]
        self.scene.camera.location.y =self.scene.camera.location.y+ step_size* final_step[0] 
        self.scene.camera.location.z =self.scene.camera.location.z+ final_step[2] * step_size

        self.scene.camera.rotation_euler[0]=self.scene.camera.rotation_euler[0]+action_vector[3] 
        self.scene.camera.rotation_euler[1]=self.scene.camera.rotation_euler[1]+action_vector[4]
        self.scene.camera.rotation_euler[2]=self.scene.camera.rotation_euler[2]+action_vector[5]

        bpy.ops.blensor.scan()

        obs = []
        for ob in self.scene.scene.objects:
            #print(ob)
            # whatever objects you want to join...
            if  ob.name.startswith('Scan'): #ob.type =='MESH' and 
                pointcloud=self.read_verts(mesh=ob.data)
                #print(pointcloud,"mate")
                pointcloud=pointcloud[~np.isnan(pointcloud).any(axis=1), :]
                
                obs.append(ob)

        self.pointcloud = np.row_stack((self.pointcloud, pointcloud))
        #ctx = bpy.context.copy()
        #ctx['active_object'] = obs[0]
        # ctx['selected_objects'] = obs
        # ctx['selected_editable_bases'] = [self.scene.scene.object_bases[ob.name] for ob in obs]
        #bpy.ops.object.join(ctx)


        # pointcloud=self.read_verts(mesh=obs.data)
        # pointcloud=pointcloud[~np.isnan(pointcloud).any(axis=1), :]
        #print(self.pointcloud)


        #bpy.ops.blensor.delete_scans()  #Delete scans from Blensor

        return self.pointcloud