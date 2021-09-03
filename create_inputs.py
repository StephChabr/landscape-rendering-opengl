import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import imageio
import pyrr
import struct
import imageio
import cv2

from textureLoader import load_texture, load_all_textures
from camera import Camera

R_earth = 6.378e+06

cam = Camera()
WIDTH, HEIGHT = 1920, 1080


# Create the arrays of vertices and indices from the DEM
def create_arrays():
    print("Create DEM")
    image = imageio.imread('D:\schabril\Documents\MNT\MNTDown15.tif')
    image = np.clip(image,0,30000)
    center = [-121,-107]
    lenx,leny = 15786,13758
    ratiox,ratioy = 1.*lenx/(image.shape[0]-1), 1.*leny/(image.shape[1]-1)
    vertices = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            newx = center[0]+lenx/2.-i*ratiox
            newy = center[1]-leny/2.+j*ratioy
            vertices += [newx,image[i,j]-(R_earth - np.sqrt(R_earth*R_earth-newx*newx-newy*newy)),newy, (8000.+1.*newx)/16000., (7000.+1.*newy)/14000.]
    indices = []
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            ind1 = i*image.shape[1]+j
            ind2 = (i+1)*image.shape[1]+j
            indices += [ind1,ind2,ind1+1,ind2,ind1+1,ind2+1]
    print("Done")
    return np.array(vertices, dtype=np.float32),np.array(indices, dtype=np.uint32)


vertex_src = """
# version 430

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

uniform mat4 projection;
uniform mat4 view;

out vec2 v_color;

void main()
{
    vec4 camera_space_pos = view * vec4(a_position, 1.0);
    gl_Position = projection * camera_space_pos;
    v_color = vec2(a_position.y/3000.,log(length(camera_space_pos))/10.);
}
"""

fragment_src = """
# version 430

in vec2 v_color;

out vec4 out_color;

void main()
{
    out_color = vec4(v_color,0,1);
}
"""



# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(WIDTH, HEIGHT, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

# Create shaders
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))


# Create the vertices and indices arrays
vertices,indices = create_arrays()

VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(12))

# Create the output texture
FBO = glGenFramebuffers(1)
text = glGenTextures(1)
depth = glGenRenderbuffers(1)

glBindTexture(GL_TEXTURE_2D,text)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH,HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

glBindRenderbuffer(GL_RENDERBUFFER, depth)
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT)

glBindFramebuffer(GL_FRAMEBUFFER,FBO)
glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,text,0)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth)

# Projection matrix

projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH/HEIGHT, 0.1, 100000)

proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

glBindFramebuffer(GL_FRAMEBUFFER,FBO)
glUseProgram(shader)
glEnable(GL_DEPTH_TEST)
# the main application loop
for set_number in range(1,10):
    print(f'Set {set_number}')
    final_cams = np.load(f'E:\Stephane\data\set{set_number}\\final_cams.npy')
    N_cams = final_cams.shape[0]
    for i in range(N_cams):
        if i%100==0:
            print(i)
        cam.set_cam(final_cams[i,:])
        view = cam.get_view_matrix()

        # Main frame buffer
        glClearColor(1, 1, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        image = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
        image = np.frombuffer(image, np.float32)
        image = image.reshape((HEIGHT,WIDTH,4))
        image = image[::-1,:,:]

        imageA = np.array(255*image,np.uint8)
        imageA = imageA[60:-60,:,:3]

        path_B = f'E:\Stephane\data\set{set_number}\images\\rend.{i+1:04d}.tif'
        imageB = cv2.imread(path_B, 1)[60:-60,:,:]

        """imageAB = np.concatenate([imageA, imageB], 1)

        if i<800:
            path_AB = f'E:\Stephane\data\pix2pix2\\train\set{set_number}_{i+1:04d}.tif'
        elif i<900:
            path_AB = f'E:\Stephane\data\pix2pix2\\val\set{set_number}_{i+1:04d}.tif'
        else:
            path_AB = f'E:\Stephane\data\pix2pix2\\test\set{set_number}_{i+1:04d}.tif'

        cv2.imwrite(path_AB, imageAB)"""

        cv2.imwrite(f'E:\Stephane\data\pix2pixHD\\train_A\set{set_number}_{i+1:04d}.tif', imageA)
        cv2.imwrite(f'E:\Stephane\data\pix2pixHD\\train_B\set{set_number}_{i+1:04d}.tif', imageB)

        glfw.swap_buffers(window)


# terminate glfw, free up allocated resources
glfw.terminate()
