import numpy as np
import pyglet

def _render_path(env):
    env._viewer2d.draw_polyline(env.path._points, linewidth=1, color=(0.3, 1.0, 0.3))

def _render_vessel(env):
    env._viewer2d.draw_polyline(env.vessel.path_taken, linewidth=1, color=(0.8, 0, 0))  # previous positions
    vertices = [
        (-env.vessel.width/2, -env.vessel.width/2),
        (-env.vessel.width/2, env.vessel.width/2),
        (env.vessel.width/2, env.vessel.width/2),
        (3/2*env.vessel.width, 0),
        (env.vessel.width/2, -env.vessel.width/2),
    ]

    env._viewer2d.draw_shape(vertices, env.vessel.position, env.vessel.heading, color=(0, 0, 0.8))

def _render_interceptions(env):
    for t, obst_intercept_array in enumerate(env.sensor_obst_intercepts_transformed_hist):
        for obst_intercept in obst_intercept_array:
            env._viewer2d.draw_circle(origin=obst_intercept, radius=1.0 - t/len(env.sensor_obst_intercepts_transformed_hist), res=30, color=(0.3, 1.0 - t/len(env.sensor_obst_intercepts_transformed_hist), 0.3))

def _render_sensors(env):
    for isensor, sensor_angle in enumerate(env.vessel._sensor_angles):
        isector = env.config["sector_partition_fun"](env, isensor) # isensor // env.config["n_sensors_per_sector"]
        #distance = env.vessel._last_sensor_dist_measurements[isensor]
        distance = env.vessel._last_sensor_dist_measurements[isensor]
        p0 = env.vessel.position
        p1 = (
            p0[0] + np.cos(sensor_angle+env.vessel.heading)*distance,
            p0[1] + np.sin(sensor_angle+env.vessel.heading)*distance
        )

        closeness = env.vessel._last_sector_dist_measurements[isector]
        redness = 0.5 + 0.5*max(0, closeness)
        greenness = 1 - max(0, closeness)
        blueness = 0.5 if abs(isector - int(np.floor(env.config["n_sectors"]/2) + 1))  % 2 == 0 and not env.config["sensor_rotation"] else 1
        alpha = 0.5
        env._viewer2d.draw_line(p0, p1, color=(redness, greenness, blueness, alpha))


def _render_feasible_distances(env):
    for isensor, sensor_angle in enumerate(env.vessel._sensor_angles):
        isector = env.config["sector_partition_fun"](env, isensor)  # isensor // env.config["n_sensors_per_sector"]
        distance = env.vessel._last_sector_feasible_dists[isector]
        p0 = env.vessel.position
        p1 = (
            p0[0] + np.cos(sensor_angle + env.vessel.heading) * distance,
            p0[1] + np.sin(sensor_angle + env.vessel.heading) * distance
        )

        closeness = env.vessel._last_sector_dist_measurements[isector]
        redness = 0.5 + 0.5 * max(0, closeness)
        greenness = 1 - max(0, closeness)
        blueness = 0.5 if abs(isector - int(np.floor(env.config["n_sectors"] / 2) + 1)) % 2 == 0 and not env.config[
            "sensor_rotation"] else 1
        alpha = 0.5
        env._viewer2d.draw_circle(origin=p1, radius=1, res=30, color=(max(0, closeness), 1, 0.5))

def _render_progress(env):
    ref_point = env.path(env.vessel._last_navi_state_dict['vessel_arclength']).flatten()
    env._viewer2d.draw_circle(origin=ref_point, radius=1, res=30, color=(0.8, 0.3, 0.3))

    target_point = env.path(env.vessel._last_navi_state_dict['target_arclength']).flatten()
    env._viewer2d.draw_circle(origin=target_point, radius=1, res=30, color=(0.3, 0.8, 0.3))

def _render_obstacles(env):
    for i, obst in enumerate(env.obstacles):
        c = (0.8, 0.8, 0.8)

        if isinstance(obst, CircularObstacle):
            env._viewer2d.draw_circle(obst.position, obst.radius, color=c)

        elif isinstance(obst, PolygonObstacle):
            env._viewer2d.draw_shape(obst.points, color=c)

        elif isinstance(obst, VesselObstacle):
            env._viewer2d.draw_shape(list(obst.boundary.exterior.coords), color=c)


def _render_tiles(env, win):
    global env_bg
    global bg

    if env_bg is None:
        # Initialise background
        from pyglet.gl.gl import GLubyte
        data = np.zeros((env_bg_h, env_bg_w, 3))
        k = env_bg_h//100
        for x in range(0, data.shape[0], k):
            for y in range(0, data.shape[1], k):
                data[x:x+k, y:y+k, :] = np.array((
                    int(255*min(1.0, 0.3 + 0.025 * (np.random.random() - 0.5))),
                    int(255*min(1.0, 0.7 + 0.025 * (np.random.random() - 0.5))),
                    int(255*min(1.0, 0.8 + 0.025 * (np.random.random() - 0.5)))
                ))

        pixels = data.flatten().astype('int').tolist()
        raw_data = (GLubyte * len(pixels))(*pixels)
        bg = pyglet.image.ImageData(width=env_bg_w, height=env_bg_h, format='RGB', data=raw_data)
        if not os.path.exists('./resources'):
            os.mkdir('./resources')
        bg.save('./resources/bg.png')
        env_bg = pyglet.sprite.Sprite(bg, x=env.vessel.position[0] - env_bg_w/2, y=env.vessel.position[1] - env_bg_h/2)
        env_bg.scale = 1

    if env.t_step % 250 == 0:
        env_bg = pyglet.sprite.Sprite(bg, x=env.vessel.position[0] - env_bg_w/2, y=env.vessel.position[1] - env_bg_h/2)
        env_bg.scale = 1

    env_bg.draw()

def _render_indicators(env, W, H):

    prog = W/40.0
    h = H/40.0
    gl.glBegin(gl.GL_QUADS)
    gl.glColor4f(0,0,0,1)
    gl.glVertex3f(W, 0, 0)
    gl.glVertex3f(W, 5*h, 0)
    gl.glVertex3f(0, 5*h, 0)
    gl.glVertex3f(0, 0, 0)
    gl.glEnd()

    env._viewer2d.reward_text_field.text = "Current Reward:"
    env._viewer2d.reward_text_field.draw()
    #env._viewer2d.reward_value_field.text = "{:2.3f} * {:2.3f} * {:2.3f} - {:2.1f} = {:2.3f}".format(env.rewarder.speed_term, env.rewarder.heading_term, env.rewarder.cte_term, env.rewarder.living_penalty, env.last_reward)
    env._viewer2d.reward_value_field.text = "{:2.3f}".format(env.last_reward)
    env._viewer2d.reward_value_field.draw()

    env._viewer2d.cum_reward_text_field.text = "Cumulative Reward:"
    env._viewer2d.cum_reward_text_field.draw()
    env._viewer2d.cum_reward_value_field.text = "{:2.3f}".format(env.cumulative_reward)
    env._viewer2d.cum_reward_value_field.draw()

    env._viewer2d.time_step_text_field.text = "Time Step:"
    env._viewer2d.time_step_text_field.draw()
    env._viewer2d.time_step_value_field.text = str(env.t_step)
    env._viewer2d.time_step_value_field.draw()

    env._viewer2d.episode_text_field.text = "Episode:"
    env._viewer2d.episode_text_field.draw()
    env._viewer2d.episode_value_field.text = str(env.episode)
    env._viewer2d.episode_value_field.draw()

    env._viewer2d.lambda_text_field.text = "Speed:"
    env._viewer2d.lambda_text_field.draw()
    env._viewer2d.lambda_value_field.text = "{:2.1f}m/s".format(env.rewarder._vessel.speed*10) # why *10? 
    env._viewer2d.lambda_value_field.draw()

    env._viewer2d.eta_text_field.text = "CTE:"
    env._viewer2d.eta_text_field.draw()
    env._viewer2d.eta_value_field.text = "{:2.1f}m".format(env.rewarder._vessel.req_latest_data()['navigation']['cross_track_error']*1000)
    env._viewer2d.eta_value_field.draw()

    env._viewer2d.input_text_field.text = "Input:"
    env._viewer2d.input_text_field.draw()
    env._viewer2d.input_value_field.text = "T_u: {:2.2f} [N], T_r: {:2.2f} [Nm]".format(env.vessel._input[0], env.vessel._input[1])
    env._viewer2d.input_value_field.draw()

    env._viewer2d.navi_text_field.text = "Input:"
    env._viewer2d.navi_text_field.draw()
    env._viewer2d.navi_value_field.text = "{:1.1f} {:1.1f} {:1.1f} {:1.1f} {:1.1f} {:1.1f}".format(*[env.vessel._last_navi_state_dict[state] for state in env.vessel.NAVIGATION_FEATURES])
    env._viewer2d.navi_value_field.draw()

def render_env(env, mode):
    global rot_angle

    def render_objects():
        t = env._viewer2d.transform
        t.enable()
        _render_sensors(env)
        #_render_interceptions(env)
        if env.path is not None:
            _render_path(env)
        _render_vessel(env)
        _render_tiles(env, win)
        _render_obstacles(env)
        #_render_feasible_distances(env)
        if env.path is not None:
            _render_progress(env)

        #_render_interceptions(env)

        # Visualise path error (DEBUGGING)
        # p = np.array(env.vessel.position)
        # dir = rotate(env.past_obs[-1][0:2], env.vessel.heading)
        # env._viewer2d.draw_line(p, p + 10*np.array(dir), color=(0.8, 0.3, 0.3))

        for geom in env._viewer2d.onetime_geoms:
           geom.render()

        t.disable()

        if env.config["show_indicators"]:
            _render_indicators(env, WINDOW_W, WINDOW_H)

    scroll_x = env.vessel.position[0]
    scroll_y = env.vessel.position[1]
    ship_angle = -env.vessel.heading + np.pi/2
    if (rot_angle is None):
        rot_angle = ship_angle
    else:
        rot_angle += CAMERA_ROTATION_SPEED * geom.princip(ship_angle - rot_angle)

    if DYNAMIC_ZOOM:
        if (int(env.t_step/1000) % 2 == 0):
            env._viewer2d.camera_zoom = 0.999*env._viewer2d.camera_zoom + 0.001*(ZOOM - env._viewer2d.camera_zoom)
        else:
            env._viewer2d.camera_zoom = 0.999*env._viewer2d.camera_zoom + 0.001*(1 - env._viewer2d.camera_zoom)

    env._viewer2d.transform.set_scale(env._viewer2d.camera_zoom, env._viewer2d.camera_zoom)
    env._viewer2d.transform.set_translation(
        WINDOW_W/2 - (scroll_x*env._viewer2d.camera_zoom*cos(rot_angle) - scroll_y*env._viewer2d.camera_zoom*sin(rot_angle)),
        WINDOW_H/2 - (scroll_x*env._viewer2d.camera_zoom*sin(rot_angle) + scroll_y*env._viewer2d.camera_zoom*cos(rot_angle))
    )
    env._viewer2d.transform.set_rotation(rot_angle)

    win = env._viewer2d.window
    win.switch_to()
    x = win.dispatch_events()
    win.clear()
    gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
    render_objects()
    arr = None

    if mode == 'rgb_array':
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(WINDOW_H, WINDOW_W, 4)
        arr = arr[::-1, :, 0:3]

    win.flip()

    env._viewer2d.onetime_geoms = []

    return arr

def init_env_viewer(env):
    env._viewer2d = Viewer2D(WINDOW_W, WINDOW_H)

    env._viewer2d.reward_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 30.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.reward_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 30.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.cum_reward_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 50.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.cum_reward_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 50.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.time_step_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 70.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.time_step_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 70.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.episode_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 90.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.episode_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 90.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.lambda_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 110.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.lambda_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 110.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.eta_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 130.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.eta_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=260, y=WINDOW_H - 130.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.input_text_field = pyglet.text.Label('0000', font_size=10,
                                            x=20, y=WINDOW_H - 150.00, anchor_x='left', anchor_y='center',
                                            color=(0, 0, 0, 255))
    env._viewer2d.input_value_field = pyglet.text.Label('0000', font_size=10,
                                            x=360, y=WINDOW_H - 150.00, anchor_x='right', anchor_y='center',
                                            color=(0, 0, 0, 255))

    env._viewer2d.navi_text_field = pyglet.text.Label('0000', font_size=10,
                                                       x=20, y=WINDOW_H - 170.00, anchor_x='left', anchor_y='center',
                                                       color=(0, 0, 0, 255))
    env._viewer2d.navi_value_field = pyglet.text.Label('0000', font_size=10,
                                                        x=260, y=WINDOW_H - 170.00, anchor_x='right', anchor_y='center',
                                                        color=(0, 0, 0, 255))

    print('Initialized 2D viewer')
