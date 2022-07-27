
import gym
import os
import numpy as np
import io
import xml.dom.minidom
import xml.etree.ElementTree as ET
import copy
import pyquaternion

import mjcf_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generalized_hammer(interp_param=0., \
        return_xml_only=False, \
        **env_kwargs):
    return generalized_hand_env(interp_param=interp_param, \
        return_xml_only=return_xml_only, \
        base_env_name='hammer-v0', \
        **env_kwargs)

def generalized_relocate(interp_param=0., \
        return_xml_only=False, \
        **env_kwargs):
    return generalized_hand_env(interp_param=interp_param, \
        return_xml_only=return_xml_only, \
        base_env_name='relocate-v0', \
        **env_kwargs)

def generalized_door(interp_param=0., \
        return_xml_only=False, \
        **env_kwargs):
    return generalized_hand_env(interp_param=interp_param, \
        return_xml_only=return_xml_only, \
        base_env_name='door-v0', \
        **env_kwargs)


def generalized_hand_env(interp_param=0., \
        return_xml_only=False, \
        base_env_name='hammer-v0', \
        phase_change=0.5, \
        **env_kwargs):

    first_interp_param = \
        (1 / phase_change * interp_param) * (interp_param < phase_change) + \
        (interp_param >= phase_change) * 1
    second_interp_param = 1/(1-phase_change) * (interp_param - phase_change) * \
        (interp_param >= phase_change)
    if np.abs(first_interp_param-1) < 1e-8:
        first_interp_param = 1. - 1e-8
    if np.abs(second_interp_param-1) < 1e-8:
        second_interp_param = 1. - 1e-8
    if np.abs(interp_param-1) < 1e-8:
        interp_param = 1. - 1e-8

    shrink_interp_param = first_interp_param
    merge_interp_param = second_interp_param

    env = gym.make(base_env_name)
    env.reset()
    xml_string = env.env.model.get_xml()

    ##### change to absolute paths
    tree = ET.ElementTree(ET.fromstring(xml_string))
    compilers = [elem for elem in tree.iter() if (elem.tag == 'compiler') ]
    for compiler in compilers:
        if 'meshdir' in compiler.attrib:
            compiler.attrib['meshdir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                    'dependencies/Adroit/resources/meshes')
        if 'texturedir' in compiler.attrib:
            compiler.attrib['texturedir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                    'dependencies/Adroit/resources/textures')

    ##### switch to contact models
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    for body in bodies:
        body_children = body.getchildren()
        idx = 0
        while idx < len(body_children):
            child = body_children[idx]
            if 'geom' == child.tag:
                if 'name' in child.attrib:
                    if child.attrib['name'].startswith('V_'):
                        if child in body.getchildren():
                            body.remove(child)
                    if child.attrib['name'].startswith('C_'):
                        child_visual = copy.deepcopy(child)
                        child_visual.attrib['name'] = child.attrib['name'].replace('C_', 'V_')
                        child_visual.attrib['class'] = 'D_Vizual'
                        if 'rgba' in child_visual.attrib:
                            del child_visual.attrib['rgba']
                        body.insert(idx, child_visual)
            idx += 1

    ##### interpolate step 1
    # shrink other three fingers
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    other_fingers = ['mf', 'rf', 'lf']
    bodies = [body for body in bodies if any([body.attrib['name'].startswith(f) for f in other_fingers])]
    finger_parts = ['proximal', 'middle', 'distal']
    bodies = [body for body in bodies if any([body.attrib['name'].endswith(f) for f in finger_parts])]
    for body in bodies:
        body_children = body.getchildren()
        for child in body_children:
            if child.tag == 'inertial':
                pos_split = child.attrib['pos'].split()
                pos_split[2] = str(float(pos_split[2])*(1-shrink_interp_param))
                child.attrib['pos'] = ' '.join(pos_split)
                child.attrib['mass'] = str(float(child.attrib['mass']) * (1 - shrink_interp_param))

            if 'pos' in child.attrib:
                pos_split = child.attrib['pos'].split()
                pos_split[2] = str(float(pos_split[2])*(1-shrink_interp_param))
                child.attrib['pos'] = ' '.join(pos_split)
            if 'size' in child.attrib:
                size_split = child.attrib['size'].split()
                size_split[1] = str(float(size_split[1])*(1-shrink_interp_param))
                child.attrib['size'] = ' '.join(size_split)

    # fix last finger joint
    joints = [elem for elem in tree.iter() if (elem.tag == 'joint') ]
    joints = [joint for joint in joints if 'name' in joint.attrib]

    joint = [joint for joint in joints if joint.attrib['name'] == 'LFJ4'][0]
    joint_range_split = joint.attrib['range'].split()
    joint.attrib['range'] = ' '.join([
            str(float(joint_range_split[0]) * (1 - interp_param)),
            str(float(joint_range_split[1]) * (1 - interp_param))
            ])

    ##### interpolate step 2
    # gradually fix joint
    joint_names = ['FFJ0', 'FFJ1', 'FFJ3', 'THJ0', 'THJ1', 'THJ2', 'THJ4']
    joints_to_fix = [joint for joint in joints if any([(j in joint.attrib['name']) for j in joint_names])]
    for joint in joints_to_fix:
        joint_range_split = joint.attrib['range'].split()
        joint.attrib['range'] = ' '.join([
                str(float(joint_range_split[0]) * (1 - interp_param)),
                str(float(joint_range_split[1]) * (1 - interp_param))
                ])

    # change joint range for FFJ2
    joint_names = ['FFJ2']
    joints_to_change = [joint for joint in joints if \
            any([j in joint.attrib['name'] for j in joint_names])]
    target_range = np.array([np.pi/6, np.pi/6*5])
    for joint in joints_to_change:
        joint_range_split = joint.attrib['range'].split()
        joint.attrib['range'] = ' '.join([
            str(float(joint_range_split[0])*(1 - interp_param) + target_range[0]*interp_param),
            str(float(joint_range_split[1])*(1 - interp_param) + target_range[1]*interp_param)
            ])
    actuators = [elem for elem in tree.iter() if (elem.tag == 'general') ]
    actuators = [actuator for actuator in actuators if 'name' in actuator.attrib]
    actuators_to_change = [actuator for actuator in actuators if \
            any([j in actuator.attrib['name'] for j in joint_names])]
    target_range = target_range + np.array([-0.01, 0.01])
    for actuator in actuators_to_change:
        ctrlrange_split = actuator.attrib['ctrlrange'].split()
        actuator.attrib['ctrlrange'] = ' '.join([
            str(float(ctrlrange_split[0])*(1 - interp_param) + target_range[0]*interp_param),
            str(float(ctrlrange_split[1])*(1 - interp_param) + target_range[1]*interp_param)
            ])

    # change joint range for THJ3
    joints_to_change = [joint for joint in joints if \
            ('THJ3' in joint.attrib['name'])]
    target_range = np.array([np.pi/6, np.pi/6*5])
    for joint in joints_to_change:
        joint_range_split = joint.attrib['range'].split()
        joint.attrib['range'] = ' '.join([
            str(float(joint_range_split[0])*(1 - interp_param) + target_range[0]*interp_param),
            str(float(joint_range_split[1])*(1 - interp_param) + target_range[1]*interp_param)
            ])
    actuators = [elem for elem in tree.iter() if (elem.tag == 'general') ]
    actuators = [actuator for actuator in actuators if 'name' in actuator.attrib]
    actuators_to_change = [actuator for actuator in actuators if \
            ('THJ3' in actuator.attrib['name'])]
    target_range = target_range + np.array([-0.01, 0.01])
    for actuator in actuators_to_change:
        ctrlrange_split = actuator.attrib['ctrlrange'].split()
        actuator.attrib['ctrlrange'] = ' '.join([
            str(float(ctrlrange_split[0])*(1 - interp_param) + target_range[0]*interp_param),
            str(float(ctrlrange_split[1])*(1 - interp_param) + target_range[1]*interp_param)
            ])

    # change orientation for thbase
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    body = [body for body in bodies if body.attrib['name'] == 'thbase'][0]
    original_quat = pyquaternion.Quaternion([float(q) for q in body.attrib['quat'].split()])
    target_quat = pyquaternion.Quaternion([1, 0, 0, 0])
    quat = pyquaternion.Quaternion.slerp(original_quat, target_quat, interp_param)
    body.attrib['quat'] = ' '.join([str(quat[0]), str(quat[1]), str(quat[2]), str(quat[3])])

    # change all fingers' width
    target_finger_width = 0.007
    geoms = [elem for elem in tree.iter() if (elem.tag == 'geom') ]
    geoms = [geom for geom in geoms if 'type' in geom.attrib]
    geoms = [geom for geom in geoms if geom.attrib['type'] == 'capsule']
    geoms = [geom for geom in geoms if 'name' in geom.attrib]
    for geom in geoms:
        geom_size_split = geom.attrib['size'].split()
        if ('_th' in geom.attrib['name']) or \
           ('_ff' in geom.attrib['name']):
            size = float(geom_size_split[0])*(1 - interp_param) + \
                    target_finger_width * interp_param
            geom.attrib['size'] = ' '.join([str(size), geom_size_split[1]])
        elif any(['_{}'.format(f) in geom.attrib['name'] for f in other_fingers]):
            size = float(geom_size_split[0])*(1 - interp_param) + 1e-8*interp_param
            geom.attrib['size'] = ' '.join([str(size), geom_size_split[1]])

    # change palm's size and pos
    geoms = [elem for elem in tree.iter() if (elem.tag == 'geom') ]
    geoms = [geom for geom in geoms if 'type' in geom.attrib]
    geoms = [geom for geom in geoms if 'box' in geom.attrib['type']]
    geoms = [geom for geom in geoms if 'size' in geom.attrib]
    geoms = [geom for geom in geoms if 'name' in geom.attrib]
    for geom in geoms:
        geom_size_split = geom.attrib['size'].split()
        if '_palm0' in geom.attrib['name']:
            target_size = [0.0111, 0.0111, 0.05]
            # target_size = [1e-8, 0.0111, 1e-8]
            x_pos = 0.
        if '_palm1' in geom.attrib['name']:
            target_size = [1e-8, 0.0111, 1e-8]
            x_pos = 0.
        if '_lfmetacarpal' in geom.attrib['name']:
            target_size = [1e-8, 0.0111, 1e-8]
            x_pos = 0.017
        geom.attrib['size'] = ' '.join([
            str(float(geom_size_split[0])*(1-interp_param)+target_size[0]*interp_param),
            str(float(geom_size_split[1])*(1-interp_param)+target_size[1]*interp_param),
            str(float(geom_size_split[2])*(1-interp_param)+target_size[2]*interp_param),
            ])
        geom_pos_split = geom.attrib['pos'].split()
        geom.attrib['pos'] = ' '.join([\
            str(float(geom_pos_split[0])*(1-interp_param) + x_pos*interp_param), \
            geom_pos_split[1], geom_pos_split[2]])

    # change pos of four fingers
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    bodies = [body for body in bodies if 'pos' in body.attrib]
    body_names = ['thbase', 'ffknuckle', 'mfknuckle', 'rfknuckle', 'lfknuckle']
    bodies = [body for body in bodies if \
            any([b in body.attrib['name'] for b in body_names])]
    target_poses = {'thbase': np.array([0., -0.01, 0.0]), \
            'ffknuckle': np.array([0, -0.01, 0.08])}
    add_offsets = { \
        'ff': {'mf': np.array([-0.005, 0.011, 0.01]), \
               'rf': np.array([-0.010, 0.011, 0.01]), \
               'lf': np.array([-0.015, 0.011, 0.01])}, \
        'mf': {'ff': np.array([ 0.005, 0.011, 0.01]), \
               'rf': np.array([-0.005, 0.011, 0.01]), \
               'lf': np.array([-0.010, 0.011, 0.01])} }
    for finger_name in other_fingers:
        add_offset = add_offsets['ff'][finger_name]
        target_poses['{}knuckle'.format(finger_name)] = \
            target_poses['ffknuckle'] + add_offset
    all_bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    lfmetacarpal = [b for b in all_bodies if b.attrib['name'] == 'lfmetacarpal'][0]
    lfmetacarpal_init_pos = lfmetacarpal.attrib['pos']
    target_poses['lfknuckle'] = np.array([-0.015,0.011,0.01]) + \
            target_poses['ffknuckle'] - \
            np.array([float(p) for p in lfmetacarpal_init_pos.split()])

    for body in bodies:
        name = body.attrib['name']
        target_pos = target_poses[name]

        body_pos_split = [float(p) for p in body.attrib['pos'].split()]
        body_pos_split = np.array(body_pos_split)

        if 'thbase' == name:
            interp_param_to_use = interp_param
        else:
            interp_param_to_use = merge_interp_param
        body_pos = body_pos_split * (1 - interp_param_to_use) + \
                target_pos * interp_param_to_use
        body.attrib['pos'] = ' '.join([str(p) for p in body_pos])

    ##### prevent contact penetrating and harden joint limit
    geoms = [e for e in tree.iter() if (e.tag =='geom')]
    for geom in geoms:
        geom.attrib['margin'] = '0'
        geom.attrib['solimp'] = '0.999 0.9999 0.001 0.01 6'
        geom.attrib['solref'] = '2e-2 1'
    default = mjcf_utils.find_elements(tree.getroot(), \
            tags='default', attribs={'class': 'main'})
    default_geom = [g for g in default if g.tag == 'geom'][0]
    default_geom.attrib['margin'] = '0'

    if return_xml_only:
        return tree

    with io.StringIO() as string:
        string.write(ET.tostring(tree.getroot(), encoding="unicode"))
        xml_string = string.getvalue()
        parsed_xml = xml.dom.minidom.parseString(xml_string)
        xml_string = parsed_xml.toprettyxml(newl="", )

    env = gym.make('generalized-{}'.format(base_env_name), \
            model_xml=xml_string, **env_kwargs)
    env.reset()

    return env


generalized_envs = {
        'door-v0-finger-shrink': generalized_door,
        'hammer-v0-finger-shrink': generalized_hammer,
        'relocate-v0-finger-shrink': generalized_relocate,
        }


if __name__ == '__main__':
    import hand_manipulation_suite
    import pickle
    import torch
    import time

    interp_param = 0.0
    view = True

    generalized_env = 'hammer-v0-finger-shrink'
    policy_file = 'log_hammer-v0_vf_recover/iterations/policy_2000.pickle'

    generalized_env = 'relocate-v0-finger-shrink'
    policy_file = 'log_relocate-v0_vf_recover/iterations/policy_3999.pickle'

    generalized_env = 'door-v0-finger-shrink'
    policy_file = 'log_door-v0_vf_recover/iterations/policy_3999.pickle'

    env = generalized_envs[generalized_env]( \
            interp_param=interp_param, \
            dense_reward=False)

    policy = pickle.load(open(policy_file, 'rb'))
    policy.to(torch.device('cpu'))

    num_episodes = 20

    horizon = 200
    for ep in range(num_episodes):
        o = env.reset()
        d = False
        t = 0
        score = 0.0
        while t < horizon and d is False:
            a = policy.get_action(o)[1]['evaluation']
            o, r, d, _ = env.step(a)
            if view:
                env.render()
            t = t + 1
            score = score + r
        print("Episode score = %f" % score)

