import json 
import maya.cmds as cmds
import pymel.core as pm
import maya.OpenMaya as om
import math
def get_rotate(p1, p2):
    punkt_a = om.MPoint(p1[0], p1[1], p1[2])
    punkt_b = om.MPoint(p2[0], p2[1], p2[2])
    rot_vector = punkt_a - punkt_b
    
    world = om.MVector(0, 1, 0)
    quat = om.MQuaternion(world, rot_vector, 1) 
    
    mat = om.MTransformationMatrix()
    util = om.MScriptUtil()
    util.createFromDouble(0, 0, 0)
    rot_i = util.asDoublePtr()
    mat.setRotation(rot_i, om.MTransformationMatrix.kXYZ)
    mat = mat.asMatrix() * quat.asMatrix()
    quat = om.MTransformationMatrix(mat).rotation()
    m_rotation = om.MVector(math.degrees(quat.asEulerRotation().x),
                                        math.degrees(quat.asEulerRotation().y),
                                        math.degrees(quat.asEulerRotation().z)
                                        )
                                        
    return (m_rotation[0],m_rotation[1],m_rotation[2])
    
jnt_mapping = {'root': [{'jnt_1': ['jnt_2','jnt_6', 'jnt_10']},
                #pass spine
                #{'jnt_10': 'jnt_11'},
                #{'jnt_11': 'jnt_12'},
                {'jnt_12': ['jnt_14', 'jnt_18','jnt_13']}
                ],
                "left_leg":[{"jnt_{0}".format(n):"jnt_{0}".format(n+1)} for n in range(2,5)],
                "right_leg":[{"jnt_{0}".format(n):"jnt_{0}".format(n+1)} for n in range(6,9)],
                "left_arm":[{"jnt_{0}".format(n):"jnt_{0}".format(n+1)} for n in range(14,17)],
                "right_arm":[{"jnt_{0}".format(n):"jnt_{0}".format(n+1)} for n in range(18,21)]}
with open("/home/flyn/test_maya.txt") as json_data:
    d = json.load(json_data)
        
def load_skeleton():
    for frame, jnt in d.iteritems():
        if not cmds.objExists("anim_joint"):
            anim_grp = cmds.group(n="anim_joint", em=True)
        for jnt_id, trans in jnt.iteritems():
            if not cmds.objExists("anim_jnt_driver_{0}".format(jnt_id)):
                cmds.select(clear=True)
                jnt = cmds.joint(n="jnt_{0}".format(jnt_id), relative=True)
                cmds.setAttr("{0}.displayLocalAxis".format(jnt), 1)
                cmds.move(trans["translate"][0],trans["translate"][1],trans["translate"][2], jnt)
                cmds.parent(jnt, anim_grp)
                driver = cmds.spaceLocator(n="anim_jnt_driver_{0}".format(jnt_id))
                cmds.pointConstraint(driver, jnt)
            cmds.setKeyframe("anim_jnt_driver_{0}".format(jnt_id), t=frame, v=trans["translate"][0], at='translateX')
            cmds.setKeyframe("anim_jnt_driver_{0}".format(jnt_id), t=frame, v=trans["translate"][1], at='translateY')
            cmds.setKeyframe("anim_jnt_driver_{0}".format(jnt_id), t=frame, v=trans["translate"][2], at='translateZ')
            
def parent_skeleton():
    for body_part, jnt_map in jnt_mapping.iteritems():
        for map_dict in jnt_map:
            for parent_jnt, child_jnt in map_dict.iteritems():
                print parent_jnt, child_jnt 
                if isinstance(child_jnt, list):
                    pass
                    #for child in child_jnt:
                        #cmds.parent(child,parent_jnt)
                else:
                    cmds.parent(child_jnt,parent_jnt)
def set_orient():
    for frame, jnt in d.iteritems():
        cmds.currentTime(int(frame))
        for body_part, jnt_map in jnt_mapping.iteritems():
            for map_dict in jnt_map:
                for parent_jnt, child_jnt in map_dict.iteritems():
                    if not isinstance(child_jnt, list):
                        p1 = cmds.xform(parent_jnt, q=True, t=True, ws=True)
                        p2 = cmds.xform(child_jnt, q=True, t=True, ws=True)
                        rotation = get_rotate(p1,p2)
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[0], at='rotateX')
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[1], at='rotateY')
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[2], at='rotateZ')
                        
load_skeleton()
parent_skeleton()
set_orient()
