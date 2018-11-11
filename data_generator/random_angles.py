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

import warnings
warnings.filterwarnings("ignore")

# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
# --- 1. load model
bfm = MorphabelModel('../examples/Data/BFM/Out/BFM.mat')

# save folder
save_folder = '3dmm_transform_angle'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# --- 2. generate face mesh: vertices(represent shape) & colors(represent texture)
sp = bfm.get_shape_para('random')
ep = bfm.get_exp_para('random')
vertices = bfm.generate_vertices(sp, ep)

tp = bfm.get_tex_para('random')
colors = bfm.generate_colors(tp)
colors = np.minimum(np.maximum(colors, 0), 1)

# --- 3. transform vertices to proper position
s = 8e-04
angles = [0, 0, 0]
t = [0, 0, 0]
transformed_vertices = bfm.transform(vertices, s, angles, t)
projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

# --- 4. render(3d obj --> 2d image)
# set prop of rendering
h = w = 256; c = 3
image_vertices = mesh.transform.to_image(projected_vertices, h, w)
image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

io.imsave('{}/face_0.jpg'.format(save_folder), image)

angles = [10, 30, 20]
transformed_vertices = bfm.transform(vertices, s, angles, t)
projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection

image_vertices = mesh.transform.to_image(projected_vertices, h, w)
image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

io.imsave('{}/face_1.jpg'.format(save_folder), image)


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


triangles = bfm.triangles
colors = colors/np.max(colors)
# move center to [0,0,0]
vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]


angle_axis_folder = os.path.join(save_folder, 'angle_axis')
random_axis_folder = os.path.join(save_folder, 'angle_random')

subprocess.call('rm -rf {}/'.format(angle_axis_folder), shell=True)
subprocess.call('rm -rf {}/'.format(random_axis_folder), shell=True)

if not os.path.exists(angle_axis_folder):
	os.mkdir(angle_axis_folder)

if not os.path.exists(random_axis_folder):
	os.mkdir(random_axis_folder)

	
options = '-delay 10 -loop 0 -layers optimize' # gif options. need ImageMagick installed.

# ---- start
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size

## 1. fix camera model(stadard camera& orth proj). change obj position.
camera['proj_type'] = 'orthographic'

# angles
for i in range(3):
	for angle in np.arange(-50, 51, 10):
		obj['s'] = scale_init
		obj['angles'] = [0, 0, 0]
		obj['angles'][i] = angle
		obj['t'] = [0, 0, 0]
		image = transform_test(vertices, obj, camera, colors, triangles) 
		io.imsave('{}/angle_axis_{}_{}_{}.jpg'.format(angle_axis_folder, *obj['angles']), image)
subprocess.call('convert {} {}/*.jpg {}'.format(options, angle_axis_folder, \
            save_folder + '/angle_axis.gif'), shell=True)

# angles
rand_angles = np.random.randint(-50, 51, size=(30,3))
for i in range(30):
    obj['s'] = scale_init
    obj['angles'] = rand_angles[i,:]
    obj['t'] = [0, 0, 0]
    image = transform_test(vertices, obj, camera, colors, triangles) 
    io.imsave('{}/random_axis_{}_{}_{}.jpg'.format(random_axis_folder, *obj['angles']), image)
subprocess.call('convert {} {}/*.jpg {}'.format(options, random_axis_folder, \
            save_folder + '/random_axis.gif'), shell=True)

print("Done!")