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
lastX, lastY = WIDTH / 2, HEIGHT / 2
isPressed = False
setValues = False
left, right, forward, backward = False, False, False, False

final_cams = np.load('final_cams.npy')
index_cam = 0
N_cams = final_cams.shape[0]
angle = 40
projection = pyrr.matrix44.create_perspective_projection_matrix(40, WIDTH/HEIGHT, 0.1, 100000)

index_fbo = 0

def count_distortions():
    glBindFramebuffer(GL_FRAMEBUFFER,FBOs[1])
    im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im = im.reshape((HEIGHT,WIDTH,4))
    #return np.sum(im<0.7)
    return np.sum(im[:,:,1]<0.1)
def count_colors():
    glBindFramebuffer(GL_FRAMEBUFFER,FBOs[2])
    im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im = im.reshape((HEIGHT,WIDTH,4))
    nbr_red= np.sum(im[:,:,0]>0.5)
    nbr_blue= np.sum(im[:,:,2]>0.5)
    nbr_green = np.sum(im[:,:,1]>0.5)
    nbr_black = WIDTH*HEIGHT-nbr_red-nbr_blue-nbr_green
    return nbr_black,nbr_blue,nbr_red, nbr_green


# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, ssbo, ssbo_values, compute_dist, index_cam, index_fbo, angle, projection, final_cams, N_cams
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False

    if key == glfw.KEY_O and action == glfw.PRESS:
        print(cam.camera_pos)
        print(cam.jaw,cam.pitch)
    if key == glfw.KEY_Q and action == glfw.PRESS:
        final_cams = np.load('final_cams.npy')
        index_cam = -1
        cam.set_cam(final_cams[index_cam,:])
        N_cams = final_cams.shape[0]
    if key == glfw.KEY_R and action == glfw.PRESS:
        cam.random_cam()
    if key == glfw.KEY_I and action == glfw.PRESS:
        index_cam = (index_cam+1)%N_cams
        print(index_cam)
        cam.set_cam(final_cams[index_cam,:])
    if key == glfw.KEY_J and action == glfw.PRESS:
        index_cam = (index_cam+10)%N_cams
        print(index_cam)
        cam.set_cam(final_cams[index_cam,:])
    if key == glfw.KEY_K and action == glfw.PRESS:
        index_cam = (index_cam-10)%N_cams
        print(index_cam)
        cam.set_cam(final_cams[index_cam,:])
    if key == glfw.KEY_P and action == glfw.PRESS:
        print(count_distortions())
        nbr_black,nbr_blue,nbr_red, nbr_green = count_colors()
        print(nbr_black,nbr_blue,nbr_red,nbr_green)
    if key == glfw.KEY_L and action == glfw.PRESS:
        glBindFramebuffer(GL_FRAMEBUFFER,FBOs[1])
        im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
        im = np.frombuffer(im, np.float32)
        im = im.reshape((HEIGHT,WIDTH,4))
        im = im[::-1,:,:]
        np.clip(im,0,1)
        imageio.imwrite('D:\schabril\Documents\\textures\\testing_dist.tif',im)
        glBindFramebuffer(GL_FRAMEBUFFER,FBOs[2])
        im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
        im = np.frombuffer(im, np.float32)
        im = im.reshape((HEIGHT,WIDTH,4))
        im = im[::-1,:,:]
        imageio.imwrite('D:\schabril\Documents\\textures\\testing_text.png',im)
        glBindFramebuffer(GL_FRAMEBUFFER,FBOs[0])
        im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
        im = np.frombuffer(im, np.float32)
        im = im.reshape((HEIGHT,WIDTH,4))
        im = im[::-1,:,:]
        imageio.imwrite('D:\schabril\Documents\\textures\\testing_render.png',im)
    if key == glfw.KEY_F and action == glfw.PRESS:
        index_fbo = (index_fbo+1)%3
    if key == glfw.KEY_Z and action == glfw.PRESS:
        angle /= 2
        projection = pyrr.matrix44.create_perspective_projection_matrix(angle, WIDTH/HEIGHT, 0.1, 100000)

# do the movement, call this function in the main loop
def do_movement():
    if left:
        cam.process_keyboard("LEFT", 20.)
    if right:
        cam.process_keyboard("RIGHT", 20.)
    if forward:
        cam.process_keyboard("FORWARD", 20.)
    if backward:
        cam.process_keyboard("BACKWARD", 20.)

# the mouse position callback function
def mouse_look_clb(window, xpos, ypos):
    global lastX, lastY, isPressed, setValues

    if isPressed:
        if setValues:
            lastX = xpos
            lastY = ypos
            setValues = False

        xoffset = xpos - lastX
        yoffset = lastY - ypos

        lastX = xpos
        lastY = ypos

        cam.process_mouse_movement(xoffset, yoffset)


# the mouse click callback function
def mouse_button_callback(window, button, action, mods):
    global isPressed, setValues

    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        isPressed = True
        setValues = True

    else:
        isPressed = False


def create_arrays():
    print("Create DEM")
    image = imageio.imread('D:\schabril\Documents\MNT\MNTdown5.tif')
    image = np.clip(image,0,3000)
    center = [-121,-107]
    lenx,leny = 15786,13758
    ratiox,ratioy = 1.*lenx/(image.shape[0]-1), 1.*leny/(image.shape[1]-1)
    vertices = np.zeros(8*image.shape[0]*image.shape[1])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            newx = center[0]+lenx/2.-i*ratiox
            newy = center[1]-leny/2.+j*ratioy
            vertices[8*(i*image.shape[1]+j):8*(i*image.shape[1]+j)+5] = [newx,image[i,j]-(R_earth - np.sqrt(R_earth*R_earth-newx*newx-newy*newy)),newy, (8000.+1.*newx)/16000., (7000.+1.*newy)/14000.]
    indices = []
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            ind1 = i*image.shape[1]+j
            ind2 = (i+1)*image.shape[1]+j
            indices += [ind1,ind2,ind1+1,ind2,ind1+1,ind2+1]
    print("Done")
    print("Compute normals")
    vertices = np.array(vertices, dtype=np.float32)
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            ind1 = 8*(i*image.shape[1]+j)
            ind2 = 8*((i+1)*image.shape[1]+j)
            vec1 = vertices[ind1:ind1+3]-vertices[ind2:ind2+3]
            vec2 = vertices[ind1+8:ind1+11]-vertices[ind2:ind2+3]
            vec3 = vertices[ind2+8:ind2+11]-vertices[ind2:ind2+3]
            normal1 = np.cross(vec2,vec1)
            normal2 = np.cross(vec3,vec2)
            normal1 = normal1 / np.sqrt(np.sum(normal1**2))
            normal2 = normal2 / np.sqrt(np.sum(normal2**2))
            vertices[ind1+5:ind1+8] = normal1
            vertices[ind1+13:ind1+16] = normal2
            vertices[ind2+5:ind2+8] = normal1
            vertices[ind2+13:ind2+16] = normal2
    print("Done")
    return np.array(vertices, dtype=np.float32),np.array(indices, dtype=np.uint32)

def create_arrays_smooth():
    print("Create DEM")
    image = imageio.imread('D:\schabril\Documents\MNT\smoothdown5.tif')
    image = np.clip(image,0,3000)
    center = [-121,-107]
    lenx,leny = 17786, 15758
    realx,realy = 15786,13758
    ratiox,ratioy = 1.*lenx/(image.shape[0]-1), 1.*leny/(image.shape[1]-1)
    minx,maxx = center[0]-realx/2.,center[0]+realx/2.
    miny,maxy = center[1]-realy/2.,center[1]+realy/2.
    vertices = np.zeros(8*image.shape[0]*image.shape[1])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            newx = center[0]+lenx/2.-i*ratiox
            newy = center[1]-leny/2.+j*ratioy
            if (minx<newx<maxx and miny<newy<maxy):
                vertices[8*(i*image.shape[1]+j):8*(i*image.shape[1]+j)+5] = [newx,image[i,j]-(R_earth - np.sqrt(R_earth*R_earth-newx*newx-newy*newy)),newy, (8000.+1.*newx)/16000., (7000.+1.*newy)/14000.]
            else:
                vertices[8*(i*image.shape[1]+j):8*(i*image.shape[1]+j)+5] = [newx,image[i,j]-(R_earth - np.sqrt(R_earth*R_earth-newx*newx-newy*newy)),newy, 0, 0]
    indices = []
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            ind1 = i*image.shape[1]+j
            ind2 = (i+1)*image.shape[1]+j
            indices += [ind1,ind2,ind1+1,ind2,ind1+1,ind2+1]
    print("Done")
    print("Compute normals")
    vertices = np.array(vertices, dtype=np.float32)
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            ind1 = 8*(i*image.shape[1]+j)
            ind2 = 8*((i+1)*image.shape[1]+j)
            vec1 = vertices[ind1:ind1+3]-vertices[ind2:ind2+3]
            vec2 = vertices[ind1+8:ind1+11]-vertices[ind2:ind2+3]
            vec3 = vertices[ind2+8:ind2+11]-vertices[ind2:ind2+3]
            normal1 = np.cross(vec2,vec1)
            normal2 = np.cross(vec3,vec2)
            normal1 = normal1 / np.sqrt(np.sum(normal1**2))
            normal2 = normal2 / np.sqrt(np.sum(normal2**2))
            vertices[ind1+5:ind1+8] = normal1
            vertices[ind1+13:ind1+16] = normal2
            vertices[ind2+5:ind2+8] = normal1
            vertices[ind2+13:ind2+16] = normal2
    print("Done")
    return np.array(vertices, dtype=np.float32),np.array(indices, dtype=np.uint32)

vertex_src = """
# version 430

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 3) in vec3 a_normal;

uniform mat4 projection;
uniform mat4 view;

out vec2 v_texture;
out vec3 v_normal;

void main()
{
    gl_Position = projection * view * vec4(a_position, 1.0);
    v_texture = a_texture;
    v_normal = a_normal;
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
# version 430

in vec2 v_texture;
in vec3 v_normal;

out vec4 out_color;

void main()
{
    out_color = vec4(50000*fwidth(v_texture),1.,1.);
    out_color = vec4(v_normal,1.);
    //float check = step(v_normal.y,0.1);
    //out_color = vec4(check,1-check,0,1);
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

vertex_print = """
# version 430

layout(location = 2) in vec3 a_position;

out vec2 v_texture;

void main()
{
    gl_Position = vec4(a_position, 1.0);
    v_texture = (a_position.xy+1)/2;
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

# set the callback functions for inputs
glfw.set_cursor_pos_callback(window, mouse_look_clb)
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_key_callback(window, key_input_clb)

# make the context current
glfw.make_context_current(window)

# Create shaders
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
second_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_second, GL_FRAGMENT_SHADER))
horizon_shader = compileProgram(compileShader(vertex_horizon, GL_VERTEX_SHADER), compileShader(fragment_horizon, GL_FRAGMENT_SHADER))
print_shader = compileProgram(compileShader(vertex_print, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

# Create the vertices and indices arrays
'''vertices,indices = create_arrays_smooth()
np.save(r'D:\schabril\Documents\\MNT\\vertices5.npy',vertices)
np.save(r'D:\schabril\Documents\\MNT\\indices5.npy',indices)
1/0'''

# Load the vertices and indices arrays
print("Load DEM")
vertices,indices = np.load(r'D:\schabril\Documents\\MNT\\vertices1.npy'), np.load(r'D:\schabril\Documents\\MNT\\indices1.npy')
print("Done")

VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(12))

glEnableVertexAttribArray(3)
glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(20))


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


# Create three frame buffer objects
FBOs = glGenFramebuffers(3)
texs = glGenTextures(3)
depths = glGenRenderbuffers(3)

# FBO 0
glBindTexture(GL_TEXTURE_2D,texs[0])
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH,HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

glBindRenderbuffer(GL_RENDERBUFFER, depths[0])
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT)

glBindFramebuffer(GL_FRAMEBUFFER,FBOs[0])
glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texs[0],0)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depths[0])

# FBO 1
glBindTexture(GL_TEXTURE_2D,texs[1])
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH,HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

glBindRenderbuffer(GL_RENDERBUFFER, depths[1])
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT)

glBindFramebuffer(GL_FRAMEBUFFER,FBOs[1])
glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texs[1],0)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depths[1])

# FBO 2
glBindTexture(GL_TEXTURE_2D,texs[2])
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH,HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

glBindRenderbuffer(GL_RENDERBUFFER, depths[2])
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT)

glBindFramebuffer(GL_FRAMEBUFFER,FBOs[2])
glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texs[2],0)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depths[2])


glBindFramebuffer(GL_FRAMEBUFFER,0)

# Load textures

texture = glGenTextures(1)
texture = load_texture(b"D:\schabril\Documents\\textures\damier.jpg", texture)
texture = load_all_textures(texture)

texture2 = glGenTextures(1)
texture2 = load_texture("D:\schabril\Documents\\textures\cancel_texture4.png", texture2)

# Projection matrix

projection = pyrr.matrix44.create_perspective_projection_matrix(40, WIDTH/HEIGHT, 0.1, 100000)
screen2cam_mat = pyrr.matrix44.inverse(projection)

proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")
proj_loc2 = glGetUniformLocation(second_shader, "projection")
view_loc2 = glGetUniformLocation(second_shader, "view")
cam2world_loc = glGetUniformLocation(horizon_shader, "cameraToWorld")
screen2cam_loc = glGetUniformLocation(horizon_shader, "screenToCamera")
cameraPosition_loc = glGetUniformLocation(horizon_shader, "cameraPosition")


# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    do_movement()

    view = cam.get_view_matrix()
    cam2world_mat = pyrr.matrix44.inverse(view)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
    glEnable(GL_DEPTH_TEST)

    # Back frame buffer 0
    glBindFramebuffer(GL_FRAMEBUFFER,FBOs[0])
    glBindTexture(GL_TEXTURE_2D,texture)
    glUseProgram(shader)
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # Back frame buffer 1
    glBindFramebuffer(GL_FRAMEBUFFER,FBOs[1])
    glUseProgram(second_shader)
    glClearColor(1,1,1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUniformMatrix4fv(proj_loc2, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc2, 1, GL_FALSE, view)

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # Back frame buffer 2
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,FBOs[2])
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
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0)
    glBindTexture(GL_TEXTURE_2D,texs[index_fbo])
    glUseProgram(print_shader)
    glClearColor(1,1,1, 1)
    glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)

    glDrawElements(GL_TRIANGLES, len(indices2), GL_UNSIGNED_INT, None)
    
    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()