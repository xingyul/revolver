
import gym
import os
import numpy as np
import io
import xml.dom.minidom
import xml.etree.ElementTree as ET

import mjcf_utils


def generalized_ant_original(interp_param=0., tmp_file_dir=None):
    env = gym.make('Ant-v2')
    env.reset()

    return env


def generalized_ant_leg_emerge(interp_param=0., tmp_file_dir='/tmp/', **kwargs):

    original_ant_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs/mujoco/assets/ant.xml')
    tree = ET.parse(original_ant_filename)
    geoms = [elem for elem in tree.iter() if (elem.tag == 'geom') ]
    geoms = [geom for geom in geoms if 'name' in geom.attrib]

    aux_length = 0.2 * (1 - interp_param) + 0.2 * interp_param
    ankle_length = 0.4 * (1 - interp_param) + 0.35 * interp_param
    foot_length = 1e-7 * (1 - interp_param) + 0.12 * interp_param

    ##### adjust aux and ankle length
    for geom in geoms:
        if 'aux_geom' in geom.attrib['name']:
            fromto = np.array([float(i) for i in geom.attrib['fromto'].split()])
            fromto = fromto * (aux_length / 0.2)
            geom.attrib['fromto'] = ' '.join([str(i) for i in fromto])
        if 'ankle_geom' in geom.attrib['name']:
            fromto = np.array([float(i) for i in geom.attrib['fromto'].split()])
            fromto = fromto * (ankle_length / 0.4)
            geom.attrib['fromto'] = ' '.join([str(i) for i in fromto])

    actuator = mjcf_utils.find_elements(tree.getroot(), tags='actuator')
    foot_rgba = np.array([0.5, 0.2, 0.2, 1])
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    bodies = [body for body in bodies if 'name' in body.attrib]
    bodies = sorted(bodies, key=lambda b: b.attrib['name']) # extremely important!
    ##### add foot
    for body in bodies:
        if body.attrib['name'].endswith('_ankle'):
            geom = mjcf_utils.find_elements(body, tags='geom')
            fromto = geom.attrib['fromto'].split()
            pos = ' '.join([str(i) for i in fromto[3:]])

            prefix = body.attrib['name'].split('_ankle')[0]
            new_body = mjcf_utils.new_body(name=prefix+'_foot', pos=pos)

            joint = mjcf_utils.find_elements(body, tags='joint')
            axis = [-1, 1, 0] if (float(fromto[3])*float(fromto[4]) > 0.) \
                    else [1, 1, 0]
            joint_range = [-20, -10] if float(fromto[3]) < 0. \
                    else [10, 20]

            new_joint_name = 'foot_{}'.format(joint.attrib['name'].split('_')[1])
            new_joint = mjcf_utils.new_joint(name=new_joint_name, \
                    pos=[0, 0, 0], range=joint_range, axis=axis, type='hinge')

            new_fromto = [0,0,0, np.sign(float(fromto[3]))*foot_length, \
                    np.sign(float(fromto[4]))*foot_length, 0]
            new_geom = mjcf_utils.new_geom( \
                    name='{}_foot_geom'.format(prefix), size=0.07, \
                    type='capsule', fromto=new_fromto, rgba=foot_rgba)
            new_body.append(new_joint)
            new_body.append(new_geom)
            body.append(new_body)

            gear = 60 * interp_param + 1e-8 * (1 - interp_param)
            new_motor = mjcf_utils.new_element(tag='motor', name=None, \
                    ctrllimited='true', ctrlrange=[-1.0,1.0], gear=gear, \
                    joint=new_joint_name)
            actuator.append(new_motor)

    with io.StringIO() as string:
        string.write(ET.tostring(tree.getroot(), encoding="unicode"))
        xml_string = string.getvalue()
        parsed_xml = xml.dom.minidom.parseString(xml_string)
        xml_string = parsed_xml.toprettyxml(newl="", )

    env = gym.make('GeneralizedAnt-v0', model_xml=xml_string)
    env.reset()

    return env


if __name__ == '__main__':
    import envs.mujoco

    interp_param = 0.0

    env = generalized_ant_leg_length_mass(interp_param=interp_param)

    viz = True
    while True:
        env.step(np.zeros([8]))
        if viz:
            env.render()

