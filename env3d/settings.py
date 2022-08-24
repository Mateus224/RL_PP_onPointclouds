import bpy
import json
import numpy as np

class Scene_settings():

    def __init__(self,config):
        json_sensor_rays=config.get('Agent01','sensor_rays')
        list_sensor_rays=json.loads(json_sensor_rays)
        self.sensor_rays=np.array(list_sensor_rays)
        self.scene = bpy.context.scene
        self.camera = bpy.data.objects['Camera']
        bpy.context.scene.objects.active = self.camera
        self.camera.rotation_mode = 'XYZ'
        self.camera.rotation_euler[0]=90 * (np.pi/ 180.0)
        self.camera.rotation_euler[1]=0 * (np.pi/ 180.0)

        self.dim_reduction=10
        self.board_matrix=np.zeros((self.dim_reduction,self.dim_reduction))
        self.max_nr_objects=self.dim_reduction*self.dim_reduction-1

        self.limit_length=20
        self.limit_width=10
        self.limit_height=10

        self.L_square=self.limit_length/self.dim_reduction
        self.W_square=self.limit_width/self.dim_reduction
        self.H_square=self.limit_height/self.dim_reduction

        self.min_dimension=min([self.L_square,self.W_square,self.H_square])

        self.obs=[]

        self.place_selection= np.random.permutation(self.dim_reduction*self.dim_reduction)

        self.x_selection=np.divmod(self.place_selection, self.dim_reduction)[0]
        self.y_selection=np.divmod(self.place_selection, self.dim_reduction)[1]

        self.rotation_high=180
        self.rotation_low=-180

        self.current_matrix_position=0

      



        # bpy.data.objects['Camera'].select=False
        #self.camera.objects['Camera'].select=True

        self.camera.scan_type='tof'
        self.camera.save_scan=True

        # bpy.context.object.location[0]=-1
        # bpy.context.object.location[1]=-2
        # bpy.context.object.location[2]=0

        self.camera.kinect_xres=self.sensor_rays[0]
        self.camera.kinect_yres=self.sensor_rays[1]
        self.camera.local_coordinates=False


        self.dim_reduction=4 ###Need to load from file
        

        self.nr_objects= 55


        self.file_names=['./pcl_agent/scenario/separate_objects/Cube.obj',
            './pcl_agent/scenario/separate_objects/Long_Cube.obj',
            './pcl_agent/scenario/separate_objects/Edge_Cube.obj',
            './pcl_agent/scenario/separate_objects/Pyramid.obj']

    
 
    
    
    def delete_meshes(self):
            meshes = set()
            # Get objects in the collection if they are meshes
            for obj in [o for o in self.scene.objects if (o.type == 'MESH' and not(o.name.startswith('Plane')))]:
                meshes.add( obj.data )
                bpy.data.objects.remove(obj)

            # Look at meshes that are orphean after objects removal
            for mesh in [m for m in meshes if m.users == 0]:
                # Delete the meshes
                bpy.data.meshes.remove(mesh)

    def load_object_meshes(self):
        object_array=np.random.randint(4, size=self.nr_objects)
        for i in object_array:
            imported_object = bpy.ops.import_scene.obj(filepath=str(self.file_names[i]))

    def create_obj_list(self):
        self.obs = []       
        for ob in self.scene.objects:
            if ob.type == 'MESH' and not(ob.name.startswith('Plane')) and not(ob.name.startswith('Scan')):
                #ob.rotation_euler[2]=(rotation_high-rotation_low) * (PI/ 180.0)* np.random.random_sample() + rotation_low * (PI/ 180.0)
                self.obs.append(ob)

    def delete_meshes(self):
        meshes = set()
        # Get objects in the collection if they are meshes
        for obj in [o for o in self.scene.objects if (o.type == 'MESH' and not(o.name.startswith('Plane')))]:
            meshes.add(obj.data )
            bpy.data.objects.remove(obj)

        # Look at meshes that are orphean after objects removal
        for mesh in [m for m in meshes if m.users == 0]:
            # Delete the meshes
            bpy.data.meshes.remove( mesh )

        

    def place_resize_objects(self):

        min_dimension=min([self.L_square,self.W_square,self.H_square])
        objects_placed=0
        self.current_matrix_position=0
        while(objects_placed<self.nr_objects) and (self.current_matrix_position<self.max_nr_objects):
                if self.board_matrix[self.x_selection[self.current_matrix_position],self.y_selection[self.current_matrix_position]]==0:
                    objects_placed=objects_placed+1
                    self.board_matrix[self.x_selection[self.current_matrix_position],self.y_selection[self.current_matrix_position]]=1

                    dimensions=np.array([self.obs[self.current_matrix_position].dimensions.x,self.obs[self.current_matrix_position].dimensions.y,self.obs[self.current_matrix_position].dimensions.z])

                    max_obj_dimension=max([self.obs[self.current_matrix_position].dimensions.x,self.obs[self.current_matrix_position].dimensions.y,self.obs[self.current_matrix_position].dimensions.z])
                    

                    diag_object=np.sqrt( (self.obs[self.current_matrix_position].dimensions.x)*(self.obs[self.current_matrix_position].dimensions.x)+
                                                (self.obs[self.current_matrix_position].dimensions.z)*(self.obs[self.current_matrix_position].dimensions.z)) *self.min_dimension/max_obj_dimension


                    #######################################################3
                    #Proportional with size of square
                    
                    self.obs[self.current_matrix_position].scale[0]=self.min_dimension/max_obj_dimension* self.min_dimension/diag_object *0.9

                    self.obs[self.current_matrix_position].scale[2]=self.min_dimension/max_obj_dimension* self.min_dimension/diag_object *0.9
                    self.obs[self.current_matrix_position].scale[1]=self.min_dimension/max_obj_dimension* self.min_dimension/diag_object *0.9


                    #################################################33
                    # Approximately same size as square

                    # obs[i].scale[0]=min_dimension/obs[i].dimensions.x

                    # obs[i].scale[2]=min_dimension/obs[i].dimensions.z
                    # obs[i].scale[1]=min_dimension/obs[i].dimensions.y

                    # obs[i].scale[0]=min_dimension/obs[i].dimensions.x* min_dimension/diag_object *0.9

                    # obs[i].scale[2]=min_dimension/obs[i].dimensions.z* min_dimension/diag_object *0.9
                    # obs[i].scale[1]=min_dimension/obs[i].dimensions.y* min_dimension/diag_object *0.9
                                    
                   ##############################################################

                    x_place=self.y_selection[self.current_matrix_position]*self.L_square
                    y_place=self.x_selection[self.current_matrix_position]*self.W_square

                    #print(str(x_selection[i])+" "+str(y_selection[i]))
                

                    self.obs[self.current_matrix_position].location[0]=x_place -self.limit_length/2
                    self.obs[self.current_matrix_position].location[1]=y_place -self.limit_width/2


                    self.obs[self.current_matrix_position].rotation_euler[2]=(self.rotation_high-self.rotation_low) * (np.pi/ 180.0)* np.random.random_sample() + self.rotation_low * (np.pi/ 180.0)

                self.current_matrix_position+=1
        

    def place_camera(self):
        while( self.current_matrix_position<self.max_nr_objects+1):
            if self.board_matrix[self.x_selection[self.current_matrix_position],self.y_selection[self.current_matrix_position]]==0:
                x_place=self.y_selection[self.current_matrix_position]*self.L_square
                y_place=self.x_selection[self.current_matrix_position]*self.W_square
                self.camera.location[0]=x_place -self.limit_length/2
                self.camera.location[1]=y_place -self.limit_width/2

                self.camera.rotation_euler[2]=(self.rotation_high-self.rotation_low) * (np.pi/ 180.0)* np.random.random_sample() + self.rotation_low * (np.pi/ 180.0)
                self.current_matrix_position+=1



    def create_scene(self):
        self.load_object_meshes()
        self.create_obj_list()
        print("ALEX TEST")
        print(len(self.obs))
        self.place_resize_objects()
        #self.place_camera()

        self.camera.location[0]=0
        self.camera.location[1]=0
        self.camera.kinect_min_dist=0
        self.camera.kinect_max_dist=50000

        return 