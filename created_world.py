import sys
sys.path.insert(1, 'C:\\Users\schabril\Desktop\project\pix2pix\pix2pixHD')

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import imageio
import pyrr
import struct
from PIL import Image

from camera import Camera
from textureLoader import load_texture, load_all_textures

from options.test_options import TestOptions
from models.models import create_model
from data.base_dataset import get_params, get_transform

R_earth = 6.378e+06

cam = Camera()
WIDTH, HEIGHT = 1920,960
WIDTH, HEIGHT = 1024,512
lastX, lastY = WIDTH / 2, HEIGHT / 2
isPressed = False
setValues = False
left, right, forward, backward = False, False, False, False

final_cams = np.load('E:\Stephane\data\set7\\final_cams.npy')
index_cam = 0
N_cams = final_cams.shape[0]

index_fbo = 0
show_drawing = False

# load the pix2pix model
opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1 
opt.serial_batches = True
opt.no_flip = True 

opt.no_instance = True  
opt.label_nc = 0
opt.name = 'small_model'
opt.checkpoints_dir = 'E:\Stephane\data\pix2pixHD\checkpoints'
opt.dataroot = 'E:\Stephane\data\pix2pixHD'
opt.results_dir = 'E:\Stephane\data\pix2pixHD\results'

model = create_model(opt)

params = get_params(opt, (WIDTH,HEIGHT))
transform_A = get_transform(opt, params)

def apply_model():
    glBindFramebuffer(GL_FRAMEBUFFER,FBOs[0])
    im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im = im.reshape((HEIGHT,WIDTH,4))
    im = np.array(255*im,np.uint8)[::-1,:,:3]

    image = Image.fromarray(im)
    
    A_tensor = transform_A(image.convert('RGB')).unsqueeze(0)
    
    generated = model.inference(A_tensor, None)
    image = generated.cpu().float().numpy()[0,:,:,:]
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255)[::-1,:,:]
    image = image.astype(np.uint8)


    image = Image.fromarray(image)
    image = image.resize((WIDTH,HEIGHT))
    image = image.convert("RGBA").tobytes()

    glBindTexture(GL_TEXTURE_2D,texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    return

def save_screen():
    glBindFramebuffer(GL_FRAMEBUFFER,0)
    im = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im = im.reshape((HEIGHT,WIDTH,4))
    im = np.array(255*im,np.uint8)[::-1,:,:3]

    image = Image.fromarray(im)
    image.save('screen.tif')

# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, ssbo, ssbo_values, compute_dist, index_cam, index_fbo, show_drawing
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
    if key == glfw.KEY_F and action == glfw.PRESS:
        index_fbo = (index_fbo+1)%3
    if key == glfw.KEY_B and action == glfw.PRESS:
        save_screen()
    if key == glfw.KEY_H and action == glfw.PRESS:
        show_drawing = not(show_drawing)
    if key == glfw.KEY_N and action == glfw.PRESS:
        index_fbo = (index_fbo + 1) % 2

# do the movement, call this function in the main loop
def do_movement():
    if left:
        cam.process_keyboard("LEFT", 50.)
    if right:
        cam.process_keyboard("RIGHT", 50.)
    if forward:
        cam.process_keyboard("FORWARD", 50.)
    if backward:
        cam.process_keyboard("BACKWARD", 50.)

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
    image = imageio.imread('D:\schabril\Documents\MNT\MNTDown15.tif')
    image = np.clip(image,0,3000)
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
    out_color = vec4(0,v_color.y,v_color.x,1);
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

fragment_print = """
# version 430

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture);
}
"""

vertex_real = """
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

fragment_real = """
# version 430

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture);
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
print_shader = compileProgram(compileShader(vertex_print, GL_VERTEX_SHADER), compileShader(fragment_print, GL_FRAGMENT_SHADER))
real_shader = compileProgram(compileShader(vertex_real, GL_VERTEX_SHADER), compileShader(fragment_real, GL_FRAGMENT_SHADER))

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


# Create three frame buffer objects
FBOs = glGenFramebuffers(3)
texs = glGenTextures(3)
depths = glGenRenderbuffers(3)

# FBO 0
glBindTexture(GL_TEXTURE_2D,texs[0])
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH,HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
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

# Texture
texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE )
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE )
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, None)

texture2 = glGenTextures(1)
texture2 = load_texture(b"D:\schabril\Documents\\textures\damier.jpg", texture2)
texture2 = load_all_textures(texture2)

# Projection matrix

projection = pyrr.matrix44.create_perspective_projection_matrix(40, WIDTH/HEIGHT, 0.1, 100000)
screen2cam_mat = pyrr.matrix44.inverse(projection)

proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")
proj_loc_real = glGetUniformLocation(real_shader, "projection")
view_loc_real = glGetUniformLocation(real_shader, "view")


# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    do_movement()

    if show_drawing:
        apply_model()

    view = cam.get_view_matrix()
    cam2world_mat = pyrr.matrix44.inverse(view)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
    glEnable(GL_DEPTH_TEST)

    # Back frame buffer 0
    glBindFramebuffer(GL_FRAMEBUFFER,FBOs[0])
    glUseProgram(shader)
    glClearColor(0, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    # Back frame buffer 1
    glBindFramebuffer(GL_FRAMEBUFFER,FBOs[1])
    glBindTexture(GL_TEXTURE_2D,texture2)
    glUseProgram(real_shader)
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUniformMatrix4fv(proj_loc_real, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc_real, 1, GL_FALSE, view)

    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)


    # Main frame buffer
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0)
    glBindBuffer(GL_ARRAY_BUFFER, VBO2)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO2)
    if show_drawing:
        glBindTexture(GL_TEXTURE_2D,texture)
    else:
        glBindTexture(GL_TEXTURE_2D,texs[index_fbo])
    glUseProgram(print_shader)
    glClearColor(1,1,1, 1)
    glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)

    glDrawElements(GL_TRIANGLES, len(indices2), GL_UNSIGNED_INT, None)
    
    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()
