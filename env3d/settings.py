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
        self.camera.rotation_mode = 'XYZ'
        self.camera.rotation_euler[0]=90 * (np.pi/ 180.0)
        self.camera.rotation_euler[1]=0 * (np.pi/ 180.0)

      



        # bpy.data.objects['Camera'].select=False
        camera.objects['Camera'].select=True

        camera.scan_type='kinect'
        camera.save_scan=True

        # bpy.context.object.location[0]=-1
        # bpy.context.object.location[1]=-2
        # bpy.context.object.location[2]=0

        camera.kinect_xres=self.sensor_rays[0]
        camera.kinect_yres=self.sensor_rays[1]
        camera.local_coordinates=False


        self.dim_reduction=4 ###Need to load from file
        

        self.nr_objects= 13


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


    
    def creat_scene(self):
        self.load_object_meshes()
        return 