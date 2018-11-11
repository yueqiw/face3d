''' 3d morphable model example
3dmm parameters --> mesh 
fitting: 2d image + 3dmm -> 3d face
'''
import os, sys
import subprocess
import numpy as np
import math
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')

# --- 2. generate face mesh: vertices(represent shape) & colors(represent texture)
sp = bfm.get_shape_para('random')
ep = bfm.get_exp_para('random')
vertices = bfm.generate_vertices(sp, ep)

tp = bfm.get_tex_para('random')
colors = bfm.generate_colors(tp)
colors = np.minimum(np.maximum(colors, 0), 1)

# --- 3. transform vertices to proper position
s = 8e-04
angles = [10, 30, 20]
t = [0, 0, 0]
transformed_vertices = bfm.transform(vertices, s, angles, t)
projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

# --- 4. render(3d obj --> 2d image)
# set prop of rendering
h = w = 256; c = 3
image_vertices = mesh.transform.to_image(projected_vertices, h, w)
image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

print('pose, groudtruth: \n', s, angles[0], angles[1], angles[2], t[0], t[1])

save_folder = 'results/3dmm_transform'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

io.imsave('{}/generated.jpg'.format(save_folder), image)


# -------------------------------------
## Transform

def transform_test(vertices, obj, camera, colors, triangles, h = 256, w = 256):
	'''
	Args:
		obj: dict contains obj transform paras
		camera: dict contains camera paras
	'''
	R = mesh.transform.angle2matrix(obj['angles'])
	transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
	
	if camera['proj_type'] == 'orthographic':
		projected_vertices = transformed_vertices
		image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	else:

		## world space to camera space. (Look at camera.) 
		camera_vertices = mesh.transform.lookat_camera(transformed_vertices, camera['eye'], camera['at'], camera['up'])
		## camera space to image space. (Projection) if orth project, omit
		projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
		## to image coords(position in image)
		image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)

	rendering = mesh.render.render_colors(image_vertices, triangles, colors, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 1)
	return rendering

# --------- load mesh data
# C = sio.loadmat('Data/example1.mat')
# vertices = C['vertices']; 
# global colors
# global triangles
# colors = C['colors']; 
triangles = bfm.triangles
colors = colors/np.max(colors)
# move center to [0,0,0]
vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]

# save folder
save_folder = 'results/3dmm_transform'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
transform_jpg_folder = 'results/3dmm_transform/jpg'
if not os.path.exists(transform_jpg_folder):
    os.mkdir(transform_jpg_folder)
options = '-delay 10 -loop 0 -layers optimize' # gif options. need ImageMagick installed.

# ---- start
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size

## 1. fix camera model(stadard camera& orth proj). change obj position.
camera['proj_type'] = 'orthographic'
# scale
for factor in np.arange(0.5, 1.2, 0.1):
	obj['s'] = scale_init*factor
	obj['angles'] = [0, 0, 0]
	obj['t'] = [0, 0, 0]
	image = transform_test(vertices, obj, camera, colors, triangles) 
	io.imsave('{}/1_1_{:.2f}.jpg'.format(transform_jpg_folder, factor), image)

# angles
for i in range(3):
	for angle in np.arange(-50, 51, 10):
		obj['s'] = scale_init
		obj['angles'] = [0, 0, 0]
		obj['angles'][i] = angle
		obj['t'] = [0, 0, 0]
		image = transform_test(vertices, obj, camera, colors, triangles) 
		io.imsave('{}/1_2_{}_{}.jpg'.format(transform_jpg_folder, i, angle), image)
subprocess.call('convert {} {}/1_*.jpg {}'.format(options, transform_jpg_folder, save_folder + '/obj.gif'), shell=True)

## 2. fix obj position(center=[0,0,0], front pose). change camera position&direction, using perspective projection(fovy fixed)
obj['s'] = scale_init
obj['angles'] = [0, 0, 0]
obj['t'] = [0, 0, 0]
# obj: center at [0,0,0]. size:200

camera['proj_type'] = 'perspective'
camera['at'] = [0, 0, 0]
camera['near'] = 1000
camera['far'] = -100
# eye position
camera['fovy'] = 30
camera['up'] = [0, 1, 0] #
# z-axis: eye from far to near, looking at the center of face
for p in np.arange(500, 250-1, -40): # 0.5m->0.25m
	camera['eye'] = [0, 0, p]  # stay in front of face
	image = transform_test(vertices, obj, camera, colors, triangles) 
	io.imsave('{}/2_eye_1_{}.jpg'.format(transform_jpg_folder, 1000-p), image)

# y-axis: eye from down to up, looking at the center of face
for p in np.arange(-300, 301, 60): # up 0.3m -> down 0.3m
	camera['eye'] = [0, p, 250] # stay 0.25m far
	image = transform_test(vertices, obj, camera, colors, triangles) 
	io.imsave('{}/2_eye_2_{}.jpg'.format(transform_jpg_folder, p/6), image)

# x-axis: eye from left to right, looking at the center of face
for p in np.arange(-300, 301, 60): # left 0.3m -> right 0.3m
	camera['eye'] = [p, 0, 250] # stay 0.25m far
	image = transform_test(vertices, obj, camera, colors, triangles) 
	io.imsave('{}/2_eye_3_{}.jpg'.format(transform_jpg_folder, -p/6), image)

# up direction
camera['eye'] = [0, 0, 250] # stay in front
for p in np.arange(-50, 51, 10):
	world_up = np.array([0, 1, 0]) # default direction
	z = np.deg2rad(p)
	Rz=np.array([[math.cos(z), -math.sin(z), 0],
                 [math.sin(z),  math.cos(z), 0],
                 [     0,       0, 1]])
	up = Rz.dot(world_up[:, np.newaxis]) # rotate up direction
	# note that: rotating up direction is opposite to rotating obj
	# just imagine: rotating camera 20 degree clockwise, is equal to keeping camera fixed and rotating obj 20 degree anticlockwise.
	camera['up'] = np.squeeze(up)
	image = transform_test(vertices, obj, camera, colors, triangles) 
	io.imsave('{}/2_eye_4_{}.jpg'.format(transform_jpg_folder, -p), image)

subprocess.call('convert {} {}/2_*.jpg {}'.format(options, transform_jpg_folder, save_folder + '/camera.gif'), shell=True)

# -- delete jpg files
# print('gifs have been generated, now delete jpgs')
# subprocess.call('rm {}/*.jpg'.format(transform_jpg_folder), shell=True)