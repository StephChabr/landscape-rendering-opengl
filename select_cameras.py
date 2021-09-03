import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import imageio
import pyrr
import struct
import imageio

from textureLoader import load_texture, load_all_textures
from camera import Camera

R_earth = 6.378e+06

cam = Camera()
WIDTH, HEIGHT = 2048,1024

final_cams = []

# Count the number of pixels where texture is distorded
def count_distortions():
    glBindFramebuffer(GL_FRAMEBUFFER,FBO)
    im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    return np.sum(im<0.7)


def count_colors():
    ''' Count each color from the image where :
        black = normel texture
        blue = sky
        red = bad pixels the texture
        green = horizon'''
    glBindFramebuffer(GL_FRAMEBUFFER,FBO2)
    im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im = im.reshape((HEIGHT,WIDTH,4))
    nbr_red= np.sum(im[:,:,0]>0.5)
    nbr_blue= np.sum(im[:,:,2]>0.5)
    nbr_green = np.sum(im[:,:,1]>0.5)
    nbr_black = WIDTH*HEIGHT-nbr_red-nbr_blue-nbr_green
    return nbr_black,nbr_blue,nbr_red, nbr_green


# Create the arrays of vertices and indices from the DEM
def create_arrays():
    print("Create DEM")
    image = imageio.imread('D:\schabril\Documents\MNT\MNTDown5.tif')
    image = np.clip(image,0,3000)
    #image = np.zeros((30,40))
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

out vec2 v_texture;

void main()
{
    gl_Position = projection * view * vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 430

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture);
}
"""

fragment_second = """
/*
    return the distorsion of the texture as a color
*/
# version 430

in vec2 v_texture;

out vec4 out_color;

void main()
{
    out_color = vec4(50000*fwidth(v_texture),1.,1.);
}
"""

vertex_horizon = """
# version 430

layout(location = 2) in vec3 vertex;

uniform mat4 cameraToWorld;
uniform mat4 screenToCamera;

out vec3 dir;

void main()
{
    dir = (cameraToWorld * vec4((screenToCamera * vec4(vertex, 1.0)).xyz, 0.0)).xyz;
    gl_Position = vec4(vertex.xy,0.9999999, 1.0);
}
"""

fragment_horizon = """
# version 430

float R = 6.378e+06;

in vec3 dir;

uniform vec3 cameraPosition;

out vec4 out_color;

void main()
{
    vec3 c = cameraPosition + vec3(0,R,0);
    float cc = dot(c,c);
    float dd = dot(dir,dir);
    float cd = dot(c,dir);
    float check = step(0.,-cd)*step(dd*(cc-R*R),cd*cd);
    out_color = vec4(0,check,1-check,1);
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
second_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_second, GL_FRAGMENT_SHADER))
horizon_shader = compileProgram(compileShader(vertex_horizon, GL_VERTEX_SHADER), compileShader(fragment_horizon, GL_FRAGMENT_SHADER))


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

# Create the buffer for the horizon
vertices2 = np.array([-1,-1,0,-1,1,0,1,-1,0,1,1,0],dtype=np.float32)
indices2 = np.array([0,1,2,1,2,3], dtype=np.uint32)

VBO2 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO2)
glBufferData(GL_ARRAY_BUFFER, vertices2.nbytes, vertices2, GL_STATIC_DRAW)

EBO2 = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices2.nbytes, indices2, GL_STATIC_DRAW)

glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertices2.itemsize * 3, ctypes.c_void_p(0))

# Create two frame buffer objects
FBO = glGenFramebuffers(1)

new_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D,new_tex)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH,HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

depth_rb = glGenRenderbuffers(1)
glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT)

glBindFramebuffer(GL_FRAMEBUFFER,FBO)
glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,new_tex,0)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth_rb)


FBO2 = glGenFramebuffers(1)

new_tex2 = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D,new_tex2)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH,HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

depth_rb2 = glGenRenderbuffers(1)
glBindRenderbuffer(GL_RENDERBUFFER, depth_rb2)
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT)

glBindFramebuffer(GL_FRAMEBUFFER,FBO2)
glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,new_tex2,0)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth_rb2)


glBindFramebuffer(GL_FRAMEBUFFER,0)

# Load textures

texture = glGenTextures(1)
#texture = load_texture(b"D:\schabril\Documents\\textures\damier.jpg", texture)
texture = load_all_textures(texture)

texture2 = glGenTextures(1)
texture2 = load_texture("D:\schabril\Documents\\textures\cancel_texture3.png", texture2)

# Projection matrix

projection = pyrr.matrix44.create_perspective_projection_matrix(30, WIDTH/HEIGHT, 0.1, 100000)
screen2cam_mat = pyrr.matrix44.inverse(projection)

proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")
proj_loc2 = glGetUniformLocation(second_shader, "projection")
view_loc2 = glGetUniformLocation(second_shader, "view")
cam2world_loc = glGetUniformLocation(horizon_shader, "cameraToWorld")
screen2cam_loc = glGetUniformLocation(horizon_shader, "screenToCamera")
cameraPosition_loc = glGetUniformLocation(horizon_shader, "cameraPosition")


glEnable(GL_DEPTH_TEST)
current_number = 0
# the main application loop
while len(final_cams)<10 and not glfw.window_should_close(window):
    if len(final_cams)%1==0 and len(final_cams)!=current_number:
        current_number = len(final_cams)
        print(current_number)
    glfw.poll_events()
    cam.random_cam()
    view = cam.get_view_matrix()
    cam2world_mat = pyrr.matrix44.inverse(view)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)

    # Back buffer
    glBindFramebuffer(GL_FRAMEBUFFER,FBO)
    glUseProgram(second_shader)
    glClearColor(1,1,1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUniformMatrix4fv(proj_loc2, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc2, 1, GL_FALSE, view)

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # Back frame buffer 2
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,FBO2)
    glBindTexture(GL_TEXTURE_2D,texture2)
    glUseProgram(shader)
    glClearColor(0, 0, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # Draw horizon
    glUseProgram(horizon_shader)

    glBindBuffer(GL_ARRAY_BUFFER, VBO2)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO2)

    glUniform3f(cameraPosition_loc, cam.camera_pos[0], cam.camera_pos[1], cam.camera_pos[2])
    glUniformMatrix4fv(cam2world_loc, 1, GL_FALSE, cam2world_mat)
    glUniformMatrix4fv(screen2cam_loc, 1, GL_FALSE, screen2cam_mat)

    glDrawElements(GL_TRIANGLES, len(indices2), GL_UNSIGNED_INT, None)

    # Main frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER,0)
    glBindTexture(GL_TEXTURE_2D,texture)
    glUseProgram(shader)
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    image = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
    image = np.frombuffer(image, np.float32)
    image = image.reshape((HEIGHT,WIDTH,4))
    image = image[::-1,:,:]

    nbr_black,nbr_blue,nbr_red, nbr_green = count_colors()
    nbr_dist = count_distortions()

    if nbr_black>2*nbr_blue and nbr_red==0 and nbr_green==0 and nbr_dist==0:
        final_cams.append(cam.get_params())

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
np.save('final_cams.npy',final_cams)
glfw.terminate()