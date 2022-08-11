


class Transition():
    __init__(self, config):
        json_env_shape=config.get('ENV','shape')
        list_env_shape=json.loads(json_env_shape)
        self.env_shape=np.array(list_env_shape)
        self.step_size= config.getint(step_size)

        self.x = self.env_shape[0]
        self.y = self.env_shape[1]
        self.z = self.env_shape[2]


    def check_transition(actions):
    

    return None

    def read_verts(mesh):
        mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
        mesh.vertices.foreach_get("co", mverts_co)
        return np.reshape(mverts_co, (len(mesh.vertices), 3))

    def make_action(self, action):
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
            pointcloud=read_verts(mesh=obs[0].data)
            pointcloud = pointcloud[~np.isnan(pointcloud)]

        return pointcloud