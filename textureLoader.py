from OpenGL.GL import glBindTexture, glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, \
    GL_TEXTURE_WRAP_T, GL_REPEAT, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,\
    glTexImage2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE 

from PIL import Image



# for use with GLFW
def load_texture(path,texture):
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE )
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # create final texture
    image = Image.open(path)
    image = image.transpose(Image.ROTATE_270)
    img_data = image.convert("RGBA").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return texture

def load_all_textures(texture):
    print('Start loading texture')
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE )
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load images
    all_images = []
    for i in range(4):
        for j in range(4):
            all_images.append(Image.open(f"D:\schabril\Documents\ortho_tiles\down{i}{j}.tif"))

    width = all_images[0].width
    height = all_images[0].height
    image = Image.new('RGB',(4*width,4*height))
    for i in range(4):
        for j in range(4):
            image.paste(all_images[4*i+j],(i*width,j*height))

    # create final texture
    image = image.transpose(Image.ROTATE_270)
    img_data = image.convert("RGBA").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    print('Done')
    return texture