
import gym
import os
import numpy as np
import copy
import io
import xml.dom.minidom
import xml.etree.ElementTree as ET

import mjcf_utils


def generalized_humanoid_original(interp_param=0., tmp_file_dir=None):
    env = gym.make('Humanoid-v2')
    env.reset()

    return env


def generalized_humanoid_leg_length_mass(interp_param=0., tmp_file_dir='/tmp/', **kwargs):

    original_ant_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs/mujoco/assets/humanoid.xml')
    tree = ET.parse(original_ant_filename)

    length_change = 0. * (1 - interp_param) + 0.1 * interp_param
    size_change = 0. * (1 - interp_param) + 0.01 * interp_param

    thigh_length_change = {'left': length_change, 'right': -length_change}
    shin_length_change = {'left': -length_change, 'right': length_change}
    thigh_size_change = size_change
    shin_size_change = -size_change

    thigh_rgba = ' '.join([str(i) for i in \
            np.array([0.8, 0.6, 0.4, 1]) * (1 - interp_param) + \
            np.array([0.2, 0.2, 0.2, 1]) * interp_param])
    shin_rgba = ' '.join([str(i) for i in \
            np.array([0.8, 0.6, 0.4, 1]) * (1 - interp_param) + \
            np.array([1, 1, 1., 1]) * interp_param])

    ##### thigh
    for side in ['left', 'right']:
        thigh_geom = mjcf_utils.find_elements(tree.getroot(), \
                tags='geom', attribs={'name': '{}_thigh1'.format(side)})
        thigh_fromto = thigh_geom.attrib['fromto'].split()
        thigh_length = float(thigh_fromto[-1]) + (-thigh_length_change[side])
        thigh_geom.attrib['fromto'] = ' '.join(thigh_fromto[:-1] + [str(thigh_length)])
        thigh_geom.attrib['size'] = str(float(thigh_geom.attrib['size']) + thigh_size_change)
        thigh_geom.attrib['rgba'] = thigh_rgba

        shin_body = mjcf_utils.find_elements(tree.getroot(), \
                tags='body', attribs={'name': '{}_shin'.format(side)})
        shin_pos = shin_body.attrib['pos'].split()
        shin_pos_z = float(shin_pos[-1]) + (-thigh_length_change[side])
        shin_body.attrib['pos'] = ' '.join(shin_pos[:-1] + [str(shin_pos_z)])

    for side in ['left', 'right']:
        shin_geom = mjcf_utils.find_elements(tree.getroot(), \
                tags='geom', attribs={'name': '{}_shin1'.format(side)})
        shin_fromto = shin_geom.attrib['fromto'].split()
        shin_length = float(shin_fromto[-1]) + (-shin_length_change[side])
        shin_geom.attrib['fromto'] = ' '.join(shin_fromto[:-1] + [str(shin_length)])
        shin_geom.attrib['size'] = str(float(shin_geom.attrib['size']) + shin_size_change)
        shin_geom.attrib['rgba'] = shin_rgba

        foot_body = mjcf_utils.find_elements(tree.getroot(), \
                tags='body', attribs={'name': '{}_foot'.format(side)})
        foot_pos = foot_body.attrib['pos'].split()
        foot_pos_z = float(foot_pos[-1]) + (-shin_length_change[side])
        foot_body.attrib['pos'] = ' '.join(foot_pos[:-1] + [str(foot_pos_z)])

    with io.StringIO() as string:
        string.write(ET.tostring(tree.getroot(), encoding="unicode"))
        xml_string = string.getvalue()
        parsed_xml = xml.dom.minidom.parseString(xml_string)
        xml_string = parsed_xml.toprettyxml(newl="", )

    env = gym.make('GeneralizedHumanoid-v0', model_xml=xml_string)
    env.reset()

    return env


if __name__ == '__main__':
    import envs.mujoco

    interp_param = 1.0

    env = generalized_humanoid_leg_length_mass(interp_param=interp_param)

    viz = True
    while True:
        env.step(np.zeros([17]))
        if viz:
            env.render()

