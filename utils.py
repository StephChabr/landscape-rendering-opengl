import imageio
import numpy as np
from scipy import interpolate

image = imageio.imread('D:\schabril\Documents\MNT\MNT_filled.tif')

xmin = -8000
xmax = 7786
ymin = -7000
ymax = 6758

def pix_to_coord(x,y):
    center = [-121,-107]
    lenx,leny = 15786,13758
    ratiox,ratioy = 1.*lenx/(image.shape[0]-1), 1.*leny/(image.shape[1]-1)
    newx = center[0]+lenx/2.-x*ratiox
    newy = center[1]-leny/2.+y*ratioy
    return newx,newy

def get_height(x,y):
    ax1 = [int(np.floor(x)),int(np.floor(x)+1)]
    ax2 = [int(np.floor(y)),int(np.floor(y)+1)]
    z = np.array([[image[ax1[i],ax2[j]] for i in range(2)] for j in range(2)])
    f = interpolate.interp2d(ax1, ax2, z, kind='linear')
    return f(x,y)[0]

'''print("start")
ax1 = np.arange(0,image.shape[0])
ax2 = np.arange(0,image.shape[1])
z = np.array([[image[ax1[i],ax2[j]] for i in range(image.shape[0])] for j in range(image.shape[1])])
f2 = interpolate.interp2d(ax1, ax2, z, kind='linear')
np.save('finterpolate.npy',f2)
print("stop")'''

f2 = np.load('finterpolate.npy',allow_pickle=True)[()]
def get_height2(x,y):
    return f2(x,y)[0]

def safe_cos(values):
    tmp = np.cos(values)
    sgn = np.sign(tmp)
    return sgn * ((1.0 - 1.0e-3) * tmp + 1.0e-3 * sgn)

def rand_cam():
    altitude = -1.
    while(altitude < 0.):
        randx = np.random.rand()*7892
        randy = np.random.rand()*6878
        altitude = get_height2(randx, randy)
    
    cx,cy = pix_to_coord(randx,randy)
    
    zmin = 50
    zmax = 30000
    cz = altitude + 1. / (1./zmax+np.random.rand()*(1. / zmin - 1. / zmax))
    cz = altitude + np.random.rand()*1500
    
    thetas = np.arctan2([-1+ymin-cy, -1+ymin-cy, 1+ymax-cy, 1+ymax-cy],
                        [1+xmax-cx, -1+xmin-cx, -1+xmin-cx, 1+xmax-cx])
    
    theta0 = thetas + [0, -.5*np.pi, -np.pi, +.5*np.pi]
    theta1 = np.roll(thetas,-1) +  [0, -.5*np.pi, np.pi, +.5*np.pi]
    
    s0 = np.sin(theta0)*.99
    s1 = np.sin(theta1)*.99 
    c0 = safe_cos(theta0)
    c1 = safe_cos(theta1)
    t0 = np.log((1.0+s0) / c0)
    t1 = np.log((1.0+s1) / c1)
    
    N = [cy-ymin, cx-xmin, ymax-cy, xmax-cx] * (t1-t0)
    N = np.clip(-N,0,100000)+1
    
    k = np.random.choice([0,1,2,3],p=N/np.sum(N))
    
    u = t0[k]+np.random.rand()*(t1[k] - t0[k])
    cyaw = 2.0 * np.arctan((np.exp(u)-1.0)/(np.exp(u)+1.0)) + .5 * k * np.pi
    cpitch = -45 + np.random.rand()*90

    return [cy,cx,cz],cyaw*180/np.pi,cpitch

def rand_cam2():
    altitude = -1.
    while(altitude < 0.):
        randx = np.random.rand()*7892
        randy = np.random.rand()*6878
        altitude = get_height2(randx, randy)
    
    cx,cy = pix_to_coord(randx,randy)
    
    zmin = 50
    zmax = 30000
    cz = altitude + 1. / (1./zmax+np.random.rand()*(1. / zmin - 1. / zmax))
    #cz = altitude + np.random.rand()*1500
    
    thetas = [0.,90.,180.,270.,360.]
    
    N = np.abs([(xmax-cx)*(ymax-cy),(cx-xmin)*(ymax-cy),(cx-xmin)*(cy-ymin),(xmax-cx)*(cy-ymin)])
    
    k = np.random.choice([0,1,2,3],p=N/np.sum(N))
    
    cyaw = thetas[k] + np.random.rand()*(thetas[k+1]-thetas[k])
    cpitch = np.clip(np.random.normal(0,15),-45,45)

    return [cy,cx,cz],cyaw,cpitch

def rand_cam3():
    found = False
    while not found:
        altitude = -1.
        while(altitude < 0.):
            randx = np.random.rand()*7892
            randy = np.random.rand()*6878
            altitude = get_height2(randx, randy)
        
        cx,cy = pix_to_coord(randx,randy)
        
        zmin = 30
        zmax = 3000
        cz = altitude + 1. / (1./zmax+np.random.rand()*(1. / zmin - 1. / zmax))
        found = True
        for i in range(-15,15):
            for j in range(-15,15):
                ecart = cz-get_height2(randx+i,randy+j)
                if (ecart*ecart+4*i*i+4*j*j<30*30):
                    found=False
    
    
    thetas = [0.,90.,180.,270.,360.]
    
    N = np.abs([(xmax-cx)*(ymax-cy),(cx-xmin)*(ymax-cy),(cx-xmin)*(cy-ymin),(xmax-cx)*(cy-ymin)])
    
    k = np.random.choice([0,1,2,3],p=N/np.sum(N))
    
    cyaw = thetas[k] + np.random.rand()*(thetas[k+1]-thetas[k])
    cpitch = np.clip(np.random.normal(0,15),-45,45)

    return [cy,cx,cz],cyaw,cpitch

def check_camera(cx,cy,cz,cyaw,cpitch):
    yaw = cyaw*np.pi/180
    pitch = cpitch*np.pi/180
    pitch_factor= np.tan(pitch)
    cvec = [np.sin(yaw),np.cos(yaw)]
    for i in range(xmin,xmax,100):
        for j in range(ymin,ymax,100):
            direc = [i-cx,j-cy]
            angle = np.dot(cvec,direc)
            if angle>0:
                squared_norm = (direc[0]*direc[0]+direc[1]*direc[1])
                if angle*angle>0.85*squared_norm:
                    min_height = cz + pitch_factor*np.sqrt(squared_norm)
                    if get_height2(i,j)>min_height:
                        return True
    return False

       
           
            
            
            