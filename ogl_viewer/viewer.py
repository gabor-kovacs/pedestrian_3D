from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import ctypes
import sys
import math
from threading import Lock
import numpy as np
import array
from enum import IntEnum

from cv_viewer.utils import *
import ogl_viewer.zed_model as zm
import pyzed.sl as sl
import time



SPHERE_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec3 in_Normal;
out vec4 b_color;
out vec3 b_position;
out vec3 b_normal;
uniform mat4 u_mvpMatrix;
uniform vec4 u_color;
uniform vec4 u_pt;
void main() {
   b_color = u_color;
   b_position = in_Vertex;
   b_normal = in_Normal;
   gl_Position =  u_mvpMatrix * (u_pt + vec4(in_Vertex, 1));
}
"""

VERTEX_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec4 in_Color;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    b_color = in_Color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""
# out_Color = b_color;
FRAGMENT_SHADER = """
# version 330 core
in vec4 b_color;
layout(location = 0) out vec4 out_Color;
void main() {
   out_Color = b_color;
}
"""

POINTCLOUD_VERTEX_SHADER ="""
#version 330 core
layout(location = 0) in vec4 in_VertexRGBA;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    uint vertexColor = floatBitsToUint(in_VertexRGBA.w);
    vec3 clr_int = vec3((vertexColor & uint(0x000000FF)), (vertexColor & uint(0x0000FF00)) >> 8, (vertexColor & uint(0x00FF0000)) >> 16);
    b_color = vec4(clr_int.r / 255.0f, clr_int.g / 255.0f, clr_int.b / 255.0f, 1.f);
    gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);
}
"""

POINTCLOUD_FRAGMENT_SHADER = """
#version 330 core
in vec4 b_color;
layout(location = 0) out vec4 out_Color;
void main() {
   out_Color = b_color;
}
"""

POINTCLOUD_HIGHLIGHT_VERTEX_SHADER ="""
#version 330 core
layout(location = 0) in vec4 in_VertexRGBA;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    uint vertexColor = floatBitsToUint(in_VertexRGBA.w);
    vec3 clr_int = vec3(255.0, 0.0, 0.0);
    b_color = vec4(clr_int.r / 255.0f, clr_int.g / 255.0f, clr_int.b / 255.0f, 1.f);
    gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);
}
"""

POINTCLOUD_HIGHLIGHT_FRAGMENT_SHADER = """
#version 330 core
in vec4 b_color;
layout(location = 0) out vec4 out_Color;
void main() {
   out_Color = b_color;
}
"""



TEXT_VERTEX_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec4 in_Color;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    b_color = in_Color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

TEXT_FRAGMENT_SHADER = """
# version 330 core
uniform vec4 u_color;
layout(location = 0) out vec4 out_Color;
void main() {
   out_Color = u_color;
}
"""


# POINTCLOUD_TEXT_FRAGMENT_SHADER = """
# #version 330 core
# in vec4 b_color;
# layout(location = 0) out vec4 out_Color;
# void main() {
#    out_Color = b_color;
# }
# """

M_PI = 3.1415926

GRID_SIZE = 9.0

def generate_color_id(_idx):
    clr = np.divide(generate_color_id_u(_idx), 255.0)
    clr[0], clr[2] = clr[2], clr[0]
    return clr

class Shader:
    def __init__(self, _vs, _fs):

        self.program_id = glCreateProgram()
        vertex_id = self.compile(GL_VERTEX_SHADER, _vs)
        fragment_id = self.compile(GL_FRAGMENT_SHADER, _fs)

        glAttachShader(self.program_id, vertex_id)
        glAttachShader(self.program_id, fragment_id)
        glBindAttribLocation( self.program_id, 0, "in_vertex")
        glBindAttribLocation( self.program_id, 1, "in_texCoord")
        glLinkProgram(self.program_id)

        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.program_id)
            glDeleteProgram(self.program_id)
            glDeleteShader(vertex_id)
            glDeleteShader(fragment_id)
            raise RuntimeError('Error linking program: %s' % (info))
        glDeleteShader(vertex_id)
        glDeleteShader(fragment_id)

    def compile(self, _type, _src):
        try:
            shader_id = glCreateShader(_type)
            if shader_id == 0:
                print("ERROR: shader type {0} does not exist".format(_type))
                exit()

            glShaderSource(shader_id, _src)
            glCompileShader(shader_id)
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                glDeleteShader(shader_id)
                raise RuntimeError('Shader compilation failed: %s' % (info))
            return shader_id
        except:
            glDeleteShader(shader_id)
            raise

    def get_program_id(self):
        return self.program_id


class Simple3DObject:
    def __init__(self, _is_static, pts_size = 3, clr_size = 3):
        self.is_init = False
        self.drawing_type = GL_TRIANGLES
        self.is_static = _is_static
        self.clear()
        self.pt_type = pts_size
        self.clr_type = clr_size

    def add_pt(self, _pts):  # _pts [x,y,z]
        for pt in _pts:
            self.vertices.append(pt)

    def add_clr(self, _clrs):    # _clr [r,g,b]
        for clr in _clrs:
            self.colors.append(clr)

    def add_normal(self, _normals):
        for normal in _normals:
            self.normals.append(normal)

    def add_point_clr(self, _pt, _clr):
        self.add_pt(_pt)
        self.add_clr(_clr)
        self.indices.append(len(self.indices))


    def add_point_clr_norm(self, _pt,  _norm):
        self.add_pt(_pt)
        self.add_normal(_norm)
        self.indices.append(len(self.indices))
    
    def add_line(self, _p1, _p2, _clr) :
        self.add_point_clr(_p1, _clr)
        self.add_point_clr(_p2, _clr)
            
    def addFace(self, p1, p2, p3, clr) :
        self.add_point_clr(p1, clr)
        self.add_point_clr(p2, clr)
        self.add_point_clr(p3, clr)
        
    def add_full_edges(self, _pts, _clr):
        start_id = int(len(self.vertices) / 3)

        for i in range(len(_pts)):
            self.add_pt(_pts[i])
            self.add_clr(_clr)

        box_links_top = np.array([0, 1, 1, 2, 2, 3, 3, 0])
        i = 0
        while i < box_links_top.size:
            self.indices.append(start_id + box_links_top[i])
            self.indices.append(start_id + box_links_top[i+1])
            i = i + 2

        box_links_bottom = np.array([4, 5, 5, 6, 6, 7, 7, 4])
        i = 0
        while i < box_links_bottom.size:
            self.indices.append(start_id + box_links_bottom[i])
            self.indices.append(start_id + box_links_bottom[i+1])
            i = i + 2
            
    def __add_single_vertical_line(self, _top_pt, _bottom_pt, _clr):
        current_pts = np.array(
            [_top_pt, 
            ((GRID_SIZE - 1) * np.array(_top_pt) + np.array(_bottom_pt)) / GRID_SIZE,
            ((GRID_SIZE - 2) * np.array(_top_pt) + np.array(_bottom_pt) * 2) / GRID_SIZE,
            (2 * np.array(_top_pt) + np.array(_bottom_pt) * (GRID_SIZE - 2)) / GRID_SIZE,
            (np.array(_top_pt) + np.array(_bottom_pt) * (GRID_SIZE - 1)) / GRID_SIZE,
            _bottom_pt
            ], np.float32)
        start_id = int(len(self.vertices) / 3)
        for i in range(len(current_pts)):
            self.add_pt(current_pts[i])
            if (i == 2 or i == 3):
                _clr[3] = 0
            else:
                _clr[3] = 0.75
            self.add_clr(_clr)

        box_links = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        i = 0
        while i < box_links.size:
            self.indices.append(start_id + box_links[i])
            self.indices.append(start_id + box_links[i+1])
            i = i + 2

    def add_vertical_edges(self, _pts, _clr):
        self.__add_single_vertical_line(_pts[0], _pts[4], _clr)
        self.__add_single_vertical_line(_pts[1], _pts[5], _clr)
        self.__add_single_vertical_line(_pts[2], _pts[6], _clr)
        self.__add_single_vertical_line(_pts[3], _pts[7], _clr)

    def add_top_face(self, _pts, _clr):
        _clr[3] = 0.5
        for pt in _pts:
            self.add_point_clr(pt, _clr)

    def __add_quad(self, _quad_pts, _alpha1, _alpha2, _clr):
        for i in range(len(_quad_pts)):
            self.add_pt(_quad_pts[i])
            if i < 2:
                _clr[3] = _alpha1
            else:
                _clr[3] = _alpha2
            self.add_clr(_clr)

        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))

    def add_vertical_faces(self, _pts, _clr):
    	# For each face, we need to add 4 quads (the first 2 indexes are always the top points of the quad)
        quads = [[0, 3, 7, 4]       # Front face
                , [3, 2, 6, 7]      # Right face
                , [2, 1, 5, 6]      # Back face
                , [1, 0, 4, 5]]     # Left face

        alpha = .5

        # Create gradually fading quads
        for quad in quads:
            quad_pts_1 = [
                _pts[quad[0]],
                _pts[quad[1]],
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[1]]) + 0.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[0]]) + 0.5 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_1, alpha, alpha, _clr)

            quad_pts_2 = [
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[0]]) + 0.5 * np.array(_pts[quad[3]])) / GRID_SIZE,
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[1]]) + 0.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[1]]) + np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[0]]) + np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_2, alpha, 2 * alpha / 3, _clr)

            quad_pts_3 = [
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[0]]) + np.array(_pts[quad[3]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[1]]) + np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[1]]) + 1.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[0]]) + 1.5 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_3, 2 * alpha / 3, alpha / 3, _clr)

            quad_pts_4 = [
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[0]]) + 1.5 * np.array(_pts[quad[3]])) / GRID_SIZE,
                    ((GRID_SIZE - 1.5) * np.array(_pts[quad[1]]) + 1.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                    ((GRID_SIZE - 2.0) * np.array(_pts[quad[1]]) + 2.0 * np.array(_pts[quad[2]])) / GRID_SIZE,
                    ((GRID_SIZE - 2.0) * np.array(_pts[quad[0]]) + 2.0 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_4, alpha / 3, 0.0, _clr)

            quad_pts_5 = [
                (np.array(_pts[quad[1]]) * 2.0 + (GRID_SIZE - 2.0) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 2.0 + (GRID_SIZE - 2.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_5, 0.0, alpha / 3, _clr)

            quad_pts_6 = [
                (np.array(_pts[quad[1]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_6, alpha / 3, 2 * alpha / 3, _clr)

            quad_pts_7 = [
                (np.array(_pts[quad[1]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_7, 2 * alpha / 3, alpha, _clr)

            quad_pts_8 = [
                (np.array(_pts[quad[0]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[2]])) / GRID_SIZE,
                np.array(_pts[quad[2]]), np.array(_pts[quad[3]])
            ]
            self.__add_quad(quad_pts_8, alpha, alpha, _clr)



    def add_sphere(self): 
        m_radius = 0.025
        m_stack_count = 16
        m_sector_count = 16

        for i in range(m_stack_count+1):
            lat0 = M_PI * (-0.5 + (i - 1) / m_stack_count)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)

            lat1 = M_PI * (-0.5 + i / m_stack_count)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            for j in range(m_sector_count):
                lng = 2 * M_PI * (j - 1) / m_sector_count
                x = math.cos(lng)
                y = math.sin(lng)

                v = [m_radius * x * zr0, m_radius * y * zr0, m_radius * z0]
                normal = [x * zr0, y * zr0, z0]
                self.add_point_clr_norm(v, normal)

                v = [m_radius * x * zr1, m_radius * y * zr1, m_radius * z1]
                normal = [x * zr1, y * zr1, z1]
                self.add_point_clr_norm(v, normal)

                lng = 2 * M_PI * j / m_sector_count
                x = math.cos(lng)
                y = math.sin(lng)

                v= [m_radius * x * zr1, m_radius * y * zr1, m_radius * z1]
                normal = [x * zr1, y * zr1, z1]
                self.add_point_clr_norm(v, normal)

                v = [m_radius * x * zr0, m_radius * y * zr0, m_radius * z0]
                normal = [x * zr0, y * zr0, z0]                
                self.add_point_clr_norm(v, normal)


    def push_to_GPU(self):
        if(self.is_init == False):
            self.vboID = glGenBuffers(3)
            self.is_init = True

        if(self.is_static):
            type_draw = GL_STATIC_DRAW
        else:
            type_draw = GL_DYNAMIC_DRAW

        if len(self.vertices):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vertices) * self.vertices.itemsize, (GLfloat * len(self.vertices))(*self.vertices), type_draw)
        
        if len(self.colors):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, len(self.colors) * self.colors.itemsize, (GLfloat * len(self.colors))(*self.colors), type_draw)

        if len(self.indices):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,len(self.indices) * self.indices.itemsize,(GLuint * len(self.indices))(*self.indices), type_draw)

        self.elementbufferSize = len(self.indices)
        
    def init(self, res):
        if(self.is_init == False):
            self.vboID = glGenBuffers(3)
            self.is_init = True

        if(self.is_static):
            type_draw = GL_STATIC_DRAW
        else:
            type_draw = GL_DYNAMIC_DRAW

        self.elementbufferSize = res.width * res.height

        glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
        glBufferData(GL_ARRAY_BUFFER, self.elementbufferSize * self.pt_type * self.vertices.itemsize, None, type_draw)
        
        if(self.clr_type):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, self.elementbufferSize * self.clr_type * self.colors.itemsize, None, type_draw)

        for i in range (0, self.elementbufferSize):
            self.indices.append(i+1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,len(self.indices) * self.indices.itemsize,(GLuint * len(self.indices))(*self.indices), type_draw)
        
    def setPoints(self, pc):
        glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
        # glBufferSubData(GL_ARRAY_BUFFER, 0, self.elementbufferSize * self.pt_type * self.vertices.itemsize, ctypes.c_void_p(pc.get_pointer()))
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.elementbufferSize * self.pt_type * self.vertices.itemsize, pc)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def clear(self):        
        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.normals = array.array('f')
        self.indices = array.array('I')
        self.elementbufferSize = 0

    def set_drawing_type(self, _type):
        self.drawing_type = _type

    def draw(self):
        if (self.elementbufferSize):            
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0,self.pt_type,GL_FLOAT,GL_FALSE,0,None)

            if(self.clr_type):
                glEnableVertexAttribArray(1)
                glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
                glVertexAttribPointer(1,self.clr_type,GL_FLOAT,GL_FALSE,0,None)
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glDrawElements(self.drawing_type, self.elementbufferSize, GL_UNSIGNED_INT, None)      
            
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)





class Skeleton:
    def __init__(self, _body_format = sl.BODY_FORMAT.POSE_18):
        self.clr = [0,0,0,1]
        self.kps = []
        self.joints = Simple3DObject(False, 3, 4)
        self.Z = 1
        self.body_format = _body_format
        self.gaze = Simple3DObject(False, 3, 4)

        self.id = 0
        self.angle = 0.
        self.head_pos = [0,0,0]

    def set(self, obj, pc_hilite):
        self.joints.set_drawing_type(GL_LINES)
        self.gaze.set_drawing_type(GL_LINES)
        self.clr = generate_color_id(obj.id)
        self.Z = abs(obj.position[2])

        # Draw skeletons
        if obj.keypoint.size > 0:
            if self.body_format == sl.BODY_FORMAT.POSE_34:
                for bone in sl.BODY_BONES_POSE_34:
                    kp_1 = obj.keypoint[bone[0].value]
                    kp_2 = obj.keypoint[bone[1].value]
                    if math.isfinite(kp_1[0]) and math.isfinite(kp_2[0]):
                        self.joints.add_line(kp_1, kp_2, self.clr)

                for part in range(len(sl.BODY_PARTS_POSE_34)-1):
                    kp = obj.keypoint[part]
                    norm = np.linalg.norm(kp)
                    if math.isfinite(norm):
                        self.kps.append(kp)

                self.head_pos = obj.keypoint[sl.BODY_PARTS_POSE_34.HEAD.value]
                self.id = obj.id

                # create gaze defined from ears and nose
                left_ear = obj.keypoint[sl.BODY_PARTS_POSE_34.LEFT_EAR.value]
                right_ear = obj.keypoint[sl.BODY_PARTS_POSE_34.RIGHT_EAR.value]
                nose = obj.keypoint[sl.BODY_PARTS_POSE_34.NOSE.value]
                gaze_vector =  nose - (right_ear + left_ear) / 2 
                gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)

                # self.gaze.add_line(nose, nose + gaze_vector,  [1,0,0,1])

                gaze_vector_step = gaze_vector * 0.1 # 10 cm steps


                # * angle between gaze vector and camera-nose vector
                nose_vector = nose / np.linalg.norm(nose)
                angle = np.arccos(np.clip(np.dot(nose_vector, gaze_vector), -1.0, 1.0))
                # radian to degree
                angle = angle * 180 / np.pi
                angle = 180 - angle

                gaze_clr = [0,0,0,1]
                if angle < 45:
                    gaze_clr = [0,1,0,1]
                elif angle < 90:
                    gaze_clr = [1,0.5,0,1]
                else:
                    gaze_clr = [1,0,0,1]

                self.angle = angle

                
                xyz = np.around(pc_hilite[:,:,0:3], decimals=1).reshape(-1,3)
                xyz = np.unique(xyz,axis=0)

                points = []

                for step in range(100):
                    pos = np.around(nose + (gaze_vector_step * step), decimals=1)
                    if any((xyz[:] == pos).all(1)):
                        points.append(pos)
                        # self.gaze.add_line(nose, pos,  [1,0,1,1])

                # if the array is empty, add 10m gaze vector
                if len(points) == 0:
                    self.gaze.add_line(nose, nose + gaze_vector * 10,  gaze_clr)
                # otherwise, get the furthest point
                else:
                    furthest_point = points[0]
                    for point in points:
                        if np.linalg.norm(point - nose) > np.linalg.norm(furthest_point - nose):
                            furthest_point = point
                    # if it is a self-gaze, add 10m gaze vector
                    if np.linalg.norm(furthest_point - nose) < 0.3:
                        self.gaze.add_line(nose, nose + gaze_vector * 10,  gaze_clr)
                    else:
                        self.gaze.add_line(nose, furthest_point,  gaze_clr)

                
                # end_time = time.time()
                # print("time: ", end_time - start_time)
            
    def push_to_GPU(self):
        self.joints.push_to_GPU()
        self.gaze.push_to_GPU()

    # def draw(self, shader_sk_clr, sphere, shader_mvp, projection):
    def draw(self):
        self.joints.draw()
        self.gaze.draw()

    def print_info(self):
        # glDisable( GL_DEPTH_TEST )
        # glDepthMask( GL_FALSE )

        glRasterPos3f(self.head_pos[0], self.head_pos[1] + 0.5, self.head_pos[2])

        info = "ID: " + str(self.id)
        for i in range(len(info)):
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ctypes.c_int(ord(info[i])))

        glutSwapBuffers()
        # glRasterPos3f(self.head_pos[0], self.head_pos[1] + 0.3, self.head_pos[2])
        # info = "Angle: " + str(int(self.angle))
        # for i in range(len(info)):
        #     glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ctypes.c_int(ord(info[i])))   
        


    def drawKPS(self, shader_clr, sphere, shader_pt):
        glUniform4f(shader_clr, self.clr[0],self.clr[1],self.clr[2],self.clr[3])         
        for k in self.kps:
            glUniform4f(shader_pt, k[0],k[1],k[2], 1)
            sphere.draw()
            sphere.draw()


    
class GLViewer:
    def __init__(self):
        self.available = False
        self.mutex = Lock()

        self.bodies = []

        self.camera = CameraGL()
        self.wheelPosition = 0.
        self.mouse_button = [False, False]
        self.mouseCurrentPosition = [0., 0.]
        self.previousMouseMotion = [0., 0.]
        self.mouseMotion = [0., 0.]
        self.zedModel = Simple3DObject(True)
        self.BBox_faces = Simple3DObject(False, 3, 4)
        self.BBox_edges = Simple3DObject(False, 3, 4)
        self.point_cloud = Simple3DObject(False, 4)
        self.point_cloud_highlight = Simple3DObject(False, 4)
        self.is_tracking_on = False         # Show tracked objects only
        self.basic_sphere = Simple3DObject(True)
        self.body_format = sl.BODY_FORMAT.POSE_18

    def init(self, camera_model, res, hilite_res, is_tracking_on, _body_format):
        glutInit(sys.argv)
        wnd_w = int(glutGet(GLUT_SCREEN_WIDTH)*0.9)
        wnd_h = int(glutGet(GLUT_SCREEN_HEIGHT) *0.9)
        glutInitWindowSize(wnd_w, wnd_h)
        glutInitWindowPosition(int(wnd_w*0.05), int(wnd_h*0.05))

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_SRGB | GLUT_DEPTH)
        glutCreateWindow("ZED Object Detection Birds View")
        glViewport(0, 0, wnd_w, wnd_h)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_DEPTH_TEST)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        self.is_tracking_on = is_tracking_on

        # Compile and create the shader for 3D objects
        self.shader_image = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
        self.shader_image_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")

        # ! get color location
        # self.shader_image_color = glGetUniformLocation(self.shader_image.get_program_id(), "u_color")
        
        self.shader_pc = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER)
        self.shader_pc_MVP = glGetUniformLocation(self.shader_pc.get_program_id(), "u_mvpMatrix")

        self.shader_pc_hilite = Shader(POINTCLOUD_HIGHLIGHT_VERTEX_SHADER, POINTCLOUD_HIGHLIGHT_FRAGMENT_SHADER)
        self.shader_pc_hilite_MVP = glGetUniformLocation(self.shader_pc_hilite.get_program_id(), "u_mvpMatrix")
        
        self.shader_text = Shader(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.shader_text_MVP = glGetUniformLocation(self.shader_text.get_program_id(), "u_mvpMatrix")
        self.shader_text_clr = glGetUniformLocation(self.shader_text.get_program_id(), "u_color")


        self.shader_sphere_image = Shader(SPHERE_SHADER, FRAGMENT_SHADER)
        self.shader_sphere_MVP = glGetUniformLocation(self.shader_sphere_image.get_program_id(), "u_mvpMatrix")
        self.shader_sphere_clr = glGetUniformLocation(self.shader_sphere_image.get_program_id(), "u_color")
        self.shader_sphere_pt = glGetUniformLocation(self.shader_sphere_image.get_program_id(), "u_pt")

        # ! CHANGE BACKGROUND COLOR
        self.bckgrnd_clr = np.array([23/255., 23/255., 23/255.])

        self.body_format = _body_format

        if(camera_model == sl.MODEL.ZED):
            for i in range(0, zm.NB_ALLUMINIUM_TRIANGLES * 3, 3):
                for j in range(3):
                    index = int(zm.alluminium_triangles[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices[index * 3], zm.vertices[index * 3 + 1], zm.vertices[index * 3 + 2]], [zm.ALLUMINIUM_COLOR.r, zm.ALLUMINIUM_COLOR.g, zm.ALLUMINIUM_COLOR.b] )

            for i in range(0, zm.NB_DARK_TRIANGLES * 3, 3):
                for j in range(3):
                    index = int(zm.dark_triangles[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices[index * 3], zm.vertices[index * 3 + 1], zm.vertices[index * 3 + 2]], [zm.DARK_COLOR.r, zm.DARK_COLOR.g, zm.DARK_COLOR.b] )
        elif(camera_model == sl.MODEL.ZED_M):
            for i in range(0, zm.NB_AL_ZEDM_TRI * 3, 3):
                for j in range(3):
                    index = int(zm.al_triangles_m[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices_m[index * 3], zm.vertices_m[index * 3 + 1], zm.vertices_m[index * 3 + 2]], [zm.ALLUMINIUM_COLOR.r, zm.ALLUMINIUM_COLOR.g, zm.ALLUMINIUM_COLOR.b] )

            for i in range(0, zm.NB_DARK_ZEDM_TRI * 3, 3):
                for j in range(3):
                    index = int(zm.dark_triangles_m[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices_m[index * 3], zm.vertices_m[index * 3 + 1], zm.vertices_m[index * 3 + 2]], [zm.DARK_COLOR.r, zm.DARK_COLOR.g, zm.DARK_COLOR.b] )

            for i in range(0, zm.NB_GRAY_ZEDM_TRI * 3, 3):
                for j in range(3):
                    index = int(zm.gray_triangles_m[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices_m[index * 3], zm.vertices_m[index * 3 + 1], zm.vertices_m[index * 3 + 2]], [zm.GRAY_COLOR.r, zm.GRAY_COLOR.g, zm.GRAY_COLOR.b] )

            for i in range(0, zm.NB_YELLOW_ZEDM_TRI * 3, 3):
                for j in range(3):
                    index = int(zm.yellow_triangles_m[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices_m[index * 3], zm.vertices_m[index * 3 + 1], zm.vertices_m[index * 3 + 2]], [zm.YELLOW_COLOR.r, zm.YELLOW_COLOR.g, zm.YELLOW_COLOR.b] )

        elif(camera_model == sl.MODEL.ZED2):
            for i in range(0, zm.NB_ALLUMINIUM_TRIANGLES * 3, 3):
                for j in range(3):
                    index = int(zm.alluminium_triangles[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices[index * 3], zm.vertices[index * 3 + 1], zm.vertices[index * 3 + 2]], [zm.DARK_COLOR.r, zm.DARK_COLOR.g, zm.DARK_COLOR.b] )

            for i in range(0, zm.NB_DARK_TRIANGLES * 3, 3):
                for j in range(3):
                    index = int(zm.dark_triangles[i + j] - 1)
                    self.zedModel.add_point_clr([zm.vertices[index * 3], zm.vertices[index * 3 + 1], zm.vertices[index * 3 + 2]], [zm.GRAY_COLOR.r, zm.GRAY_COLOR.g, zm.GRAY_COLOR.b] )
        self.zedModel.set_drawing_type(GL_TRIANGLES)
        self.zedModel.push_to_GPU()

        self.point_cloud.init(res)
        self.point_cloud.set_drawing_type(GL_POINTS)

        self.point_cloud_highlight.init(hilite_res)
        self.point_cloud_highlight.set_drawing_type(GL_POINTS)
        
        self.BBox_edges.set_drawing_type(GL_LINES)
        self.BBox_faces.set_drawing_type(GL_QUADS)



        self.basic_sphere.add_sphere()        
        self.basic_sphere.set_drawing_type(GL_QUADS)
        self.basic_sphere.push_to_GPU()
        
        # Register GLUT callback functions
        glutDisplayFunc(self.draw_callback)
        glutIdleFunc(self.idle)
        glutKeyboardFunc(self.keyPressedCallback)
        glutCloseFunc(self.close_func)
        glutMouseFunc(self.on_mouse)
        glutMotionFunc(self.on_mousemove)
        glutReshapeFunc(self.on_resize)  

        self.available = True

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def render_object(self, _object_data):      # _object_data of type sl.ObjectData
        if self.is_tracking_on:
            return (_object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
        else:
           return (_object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK or _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF) 


    def update_view(self, pc, pc_hilite, _objs):       # _objs of type sl.Objects
        self.mutex.acquire()
        self.point_cloud.setPoints(pc)
        self.point_cloud_highlight.setPoints(pc_hilite)
        # Clear objects
        self.BBox_edges.clear() 
        self.BBox_faces.clear()
        self.bodies.clear()
        # Only show tracked objects
        for obj in _objs.object_list:
            if self.render_object(obj):
                current_sk = Skeleton(self.body_format)
                current_sk.set(obj, pc_hilite)
                self.bodies.append(current_sk)

                bounding_box = np.array(obj.bounding_box)
                if bounding_box.any():
                    color_id = generate_color_id(obj.id)
                    self.create_bbox_rendering(bounding_box, color_id)

        self.mutex.release()

        
    def create_bbox_rendering(self, _bbox, _bbox_clr):
        # First create top and bottom full edges
        self.BBox_edges.add_full_edges(_bbox, _bbox_clr)
        # Add faded vertical edges
        self.BBox_edges.add_vertical_edges(_bbox, _bbox_clr)
        # Add faces
        self.BBox_faces.add_vertical_faces(_bbox, _bbox_clr)
        # Add top face
        self.BBox_faces.add_top_face(_bbox, _bbox_clr)

    def idle(self):
        if self.available:
            glutPostRedisplay()

    def exit(self):
        if self.available:
            self.available = False

    def close_func(self):
        if self.available:
            self.available = False

    def keyPressedCallback(self, key, x, y):
        if ord(key) == 27:          # 'Esc' key
            self.close_func()

    def on_mouse(self,*args,**kwargs):
        (key,Up,x,y) = args
        if key==0:
            self.mouse_button[0] = (Up == 0)
        elif key==2 :
            self.mouse_button[1] = (Up == 0)  
        elif(key == 3):
            self.wheelPosition = self.wheelPosition + 1
        elif(key == 4):
            self.wheelPosition = self.wheelPosition - 1
        
        self.mouseCurrentPosition = [x, y]
        self.previousMouseMotion = [x, y]
        
    def on_mousemove(self,*args,**kwargs):
        (x,y) = args
        self.mouseMotion[0] = x - self.previousMouseMotion[0]
        self.mouseMotion[1] = y - self.previousMouseMotion[1]
        self.previousMouseMotion = [x, y]
        glutPostRedisplay()

    def on_resize(self,Width,Height):
        glViewport(0, 0, Width, Height)
        self.camera.setProjection(Height / Width)

    def draw_callback(self):
        if self.available:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(self.bckgrnd_clr[0], self.bckgrnd_clr[1], self.bckgrnd_clr[2], 1.)

            self.mutex.acquire()
            self.update()
            self.draw()
            self.mutex.release()  

            glutSwapBuffers()
            glutPostRedisplay()

    def update(self):
        if(self.mouse_button[0]):
            r = sl.Rotation()
            vert=self.camera.vertical_
            tmp = vert.get()
            vert.init_vector(tmp[0] * 1.,tmp[1] * 1., tmp[2] * 1.)
            r.init_angle_translation(self.mouseMotion[0] * 0.02, vert)
            self.camera.rotate(r)

            r.init_angle_translation(self.mouseMotion[1] * 0.02, self.camera.right_)
            self.camera.rotate(r)
        
        if(self.mouse_button[1]):
            t = sl.Translation()
            tmp = self.camera.right_.get()
            scale = self.mouseMotion[0] *0.04
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)
            
            tmp = self.camera.up_.get()
            scale = self.mouseMotion[1] * 0.04
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

        if (self.wheelPosition != 0):          
            t = sl.Translation()  
            tmp = self.camera.forward_.get()
            scale = self.wheelPosition * -0.2
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

        self.BBox_edges.push_to_GPU()
        self.BBox_faces.push_to_GPU()
        for body in self.bodies:
            body.push_to_GPU()
        
        self.camera.update()

        self.mouseMotion = [0., 0.]
        self.wheelPosition = 0

    def draw(self):
        vpMatrix = self.camera.getViewProjectionMatrix()

        glUseProgram(self.shader_pc.get_program_id())
        glUniformMatrix4fv(self.shader_pc_MVP, 1, GL_TRUE, (GLfloat * len(vpMatrix))(*vpMatrix))
        glPointSize(2.5)
        self.point_cloud.draw()



        glUseProgram(0)

        # glUseProgram(self.shader_pc_hilite.get_program_id())
        # glUniformMatrix4fv(self.shader_pc_hilite_MVP, 1, GL_TRUE, (GLfloat * len(vpMatrix))(*vpMatrix))
        # glPointSize(1.2)
        # self.point_cloud_highlight.draw()
        # glUseProgram(0)

        # glUniformMatrix4fv(self.shader_image_MVP, 1, GL_TRUE, (GLfloat * len(vpMatrix))(*vpMatrix))   

        glUseProgram(0)


        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_image_MVP, 1, GL_TRUE, (GLfloat * len(vpMatrix))(*vpMatrix))   

        
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(4.)

        for body in self.bodies:
            body.draw()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.zedModel.draw() 
        self.BBox_faces.draw()


        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(2.)
        self.BBox_edges.draw()

        glUseProgram(0)

        # ! print person info
        glUseProgram(self.shader_text.get_program_id())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glUniformMatrix4fv(self.shader_text_MVP, 1, GL_TRUE, (GLfloat * len(vpMatrix))(*vpMatrix))   
        for body in self.bodies:
            glUniform4f(self.shader_text_clr, body.clr[0], body.clr[1], body.clr[2], 1.)
            body.print_info()
        glUseProgram(0)

        
        
class CameraGL:
    def __init__(self):
        self.ORIGINAL_FORWARD = sl.Translation()
        self.ORIGINAL_FORWARD.init_vector(0,0,1)
        self.ORIGINAL_UP = sl.Translation()
        self.ORIGINAL_UP.init_vector(0,1,0)
        self.ORIGINAL_RIGHT = sl.Translation()
        self.ORIGINAL_RIGHT.init_vector(1,0,0)
        self.znear = 0.5
        self.zfar = 100.
        self.horizontalFOV = 70.
        self.orientation_ = sl.Orientation()
        self.position_ = sl.Translation()
        self.forward_ = sl.Translation()
        self.up_ = sl.Translation()
        self.right_ = sl.Translation()
        self.vertical_ = sl.Translation()
        self.vpMatrix_ = sl.Matrix4f()
        self.offset_ = sl.Translation()
        self.offset_.init_vector(0,0,5)
        self.projection_ = sl.Matrix4f()
        self.projection_.set_identity()
        self.setProjection(1.78)

        self.position_.init_vector(0., 0., 0.)
        tmp = sl.Translation()
        tmp.init_vector(0, 0, -.1)
        tmp2 = sl.Translation()
        tmp2.init_vector(0, 1, 0)
        self.setDirection(tmp, tmp2)        

    def update(self): 
        dot_ = sl.Translation.dot_translation(self.vertical_, self.up_)
        if(dot_ < 0.):
            tmp = self.vertical_.get()
            self.vertical_.init_vector(tmp[0] * -1.,tmp[1] * -1., tmp[2] * -1.)
        transformation = sl.Transform()

        tmp_position = self.position_.get()
        tmp = (self.offset_ * self.orientation_).get()
        new_position = sl.Translation()
        new_position.init_vector(tmp_position[0] + tmp[0], tmp_position[1] + tmp[1], tmp_position[2] + tmp[2])
        transformation.init_orientation_translation(self.orientation_, new_position)
        transformation.inverse()
        self.vpMatrix_ = self.projection_ * transformation
        
    def setProjection(self, im_ratio):
        fov_x = self.horizontalFOV * 3.1416 / 180.
        fov_y = self.horizontalFOV * im_ratio * 3.1416 / 180.

        self.projection_[(0,0)] = 1. / math.tan(fov_x * .5)
        self.projection_[(1,1)] = 1. / math.tan(fov_y * .5)
        self.projection_[(2,2)] = -(self.zfar + self.znear) / (self.zfar - self.znear)
        self.projection_[(3,2)] = -1.
        self.projection_[(2,3)] = -(2. * self.zfar * self.znear) / (self.zfar - self.znear)
        self.projection_[(3,3)] = 0.

    def getProjection(self):
        tmp = self.projection_.m
        Mat = array.array('f')
        for row in tmp:
            for v in row:
                Mat.append(v)
        return Mat
    
    def getViewProjectionMatrix(self):
        tmp = self.vpMatrix_.m
        vpMat = array.array('f')
        for row in tmp:
            for v in row:
                vpMat.append(v)
        return vpMat
        
    def getViewProjectionMatrixRT(self, tr):
        tmp = self.vpMatrix_
        tmp.transpose()
        tr.transpose()
        tmp =  (tr * tmp).m
        vpMat = array.array('f')
        for row in tmp:
            for v in row:
                vpMat.append(v)
        return vpMat

    def setDirection(self, dir, vert):
        dir.normalize()
        tmp = dir.get()
        dir.init_vector(tmp[0] * -1.,tmp[1] * -1., tmp[2] * -1.)
        self.orientation_.init_translation(self.ORIGINAL_FORWARD, dir)
        self.updateVectors()
        self.vertical_ = vert
        if(sl.Translation.dot_translation(self.vertical_, self.up_) < 0.):
            tmp = sl.Rotation()
            tmp.init_angle_translation(3.14, self.ORIGINAL_FORWARD)
            self.rotate(tmp)
    
    def translate(self, t):
        ref = self.position_.get()
        tmp = t.get()
        self.position_.init_vector(ref[0] + tmp[0], ref[1] + tmp[1], ref[2] + tmp[2])

    def setPosition(self, p):
        self.position_ = p

    def rotate(self, r): 
        tmp = sl.Orientation()
        tmp.init_rotation(r)
        self.orientation_ = tmp * self.orientation_
        self.updateVectors()

    def setRotation(self, r):
        self.orientation_.init_rotation(r)
        self.updateVectors()

    def updateVectors(self):
        self.forward_ = self.ORIGINAL_FORWARD * self.orientation_
        self.up_ = self.ORIGINAL_UP * self.orientation_
        right = self.ORIGINAL_RIGHT
        tmp = right.get()
        right.init_vector(tmp[0] * -1.,tmp[1] * -1., tmp[2] * -1.)
        self.right_ = right * self.orientation_
