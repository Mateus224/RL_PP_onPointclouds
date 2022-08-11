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
        self.camera.rotation_euler[0]=90 * (PI/ 180.0)
        self.camera.rotation_euler[1]=0 * (PI/ 180.0)



        # bpy.data.objects['Camera'].select=False
        bpy.data.objects['Camera'].select=True

        bpy.context.object.scan_type='kinect'
        bpy.context.object.save_scan=True

        # bpy.context.object.location[0]=-1
        # bpy.context.object.location[1]=-2
        # bpy.context.object.location[2]=0

        bpy.context.object.kinect_xres=self.sensor_rays[0]
        bpy.context.object.kinect_yres=self.sensor_rays[1]
        bpy.context.object.local_coordinates=False


    
    def creat_scene(self):

        return 