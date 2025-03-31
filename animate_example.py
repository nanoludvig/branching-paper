import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys
from vispy import app

timer = app.Timer(interval=0.016)
iterator = 0

def interactive_animate(folder, polar='pcp', view_style='arcball', polarity_style='balls'):
	global timer

	# Storage lists
	x_lst = []
	polar_pos_lst = []
	rod_start_lst = []
	rod_end_lst = []
	color_lst = []

	# Load data
	data = np.load(folder, allow_pickle=True)
	print(f'Number of variables saved: {len(data)}')
	x_data, p_data, q_data = data[:3]
	type_data = data[3]

	for i in range(len(x_data)):
		x = x_data[i]
		p = p_data[i]
		q = q_data[i]
		tipe = type_data[i] if type_data is not None else None

		# Select polarity vector
		if polar == 'AB':
			polar_dir = -p
		elif polar == 'pcp':
			polar_dir = -q
		else:
			raise ValueError("polar must be 'AB' or 'pcp'")

		# --- Handle polarity visualization style ---
		if polarity_style == 'balls':
			polar_pos = x + 0.2 * polar_dir
			if tipe is not None:
				mesenchyme_idx = np.where(tipe == 2)[0]
				polar_pos[mesenchyme_idx] = x[mesenchyme_idx]
			polar_pos_lst.append(polar_pos)

		elif polarity_style == 'rods':
			if polar == 'AB':
				polar_dir *= 2
			rod_start = x + 0.8 * p - polar_dir
			rod_end = rod_start + 0.8 * p + polar_dir
			if tipe is not None:
				mesenchyme_idx = np.where(tipe == 2)[0]
				rod_start[mesenchyme_idx] = x[mesenchyme_idx]
				rod_end[mesenchyme_idx] = x[mesenchyme_idx]
			rod_start_lst.append(rod_start)
			rod_end_lst.append(rod_end)

		# Color assignment
		if tipe is not None:
			custom_colormap = {
				0: [0, 0.5, 0, 1],     # Green
				1: [1, 0.4, 0.4, 1],   # Pink
				2: [1, 0, 0, 1],       # Red
			}
			colors = np.array([custom_colormap[val] for val in tipe])
			color_lst.append(colors)

		x_lst.append(x)

	number_of_frames = len(x_lst)
	print(f'Number of frames: {number_of_frames}')
	print(f'Final number of cells: {len(x)}')

	particle_size = 2.5

	# Create canvas and view
	canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
	view = canvas.central_widget.add_view()
	


	# Static positions
	scatter_x = visuals.Markers(scaling=True, alpha=10, spherical=True)
	view.add(scatter_x)

	if polarity_style == 'balls':
		scatter_pol = visuals.Markers(scaling=True, alpha=10, spherical=True)
		color_x = np.ones([len(x_lst[0]), 4])
		color_x[np.where(type_data[0] == 2)[0]] = [1, 0, 0, 1]
		scatter_pol.set_data(polar_pos_lst[0], edge_width=0, face_color=color_x, size=particle_size)
		view.add(scatter_pol)

	elif polarity_style == 'rods':
		line_pos = np.stack([rod_start_lst[0], rod_end_lst[0]], axis=1).reshape(-1, 3)
		color_x = np.zeros([len(x_lst[0]), 3])
		color_x = np.concatenate((color_x, np.ones((len(x_lst[0]), 1))), axis=1)
		line_color = color_x.repeat(2, axis=0)
		rods = visuals.Line(pos=line_pos, color=line_color, connect='segments', method='gl', width=10)
		view.add(rods)

	# Set initial positions
	scatter_x.set_data(x_lst[0], edge_width=0, face_color=color_lst[0], size=particle_size)

	# Animation function
	iterator = 0
	def update(ev):
		nonlocal iterator
		iterator = (iterator + 1) % number_of_frames
		x = x_lst[iterator]
		tipe = type_data[iterator]
		

		color_frame = color_lst[iterator]
		scatter_x.set_data(x, edge_width=0, face_color=color_frame, size=particle_size)
		if polarity_style == 'balls':
			color_x = np.ones([len(x), 4])
			color_x[np.where(tipe == 2)[0]] = [1, 0, 0, 1]
			scatter_pol.set_data(polar_pos_lst[iterator], edge_width=0, face_color=color_x, size=particle_size)
		elif polarity_style == 'rods':
			start = rod_start_lst[iterator]
			end = rod_end_lst[iterator]
			line_pos = np.stack([start, end], axis=1).reshape(-1, 3)
			color_x = np.zeros([len(x), 3])
			# concatenate a 4th dimension of the color_x array with 1s
			color_x = np.concatenate((color_x, np.ones((len(x), 1))), axis=1)
			line_color = color_x.repeat(2, axis=0)
			rods.set_data(pos=line_pos, color=line_color)

	# Controls
	@canvas.connect
	def on_key_press(event):
		nonlocal iterator
		if event.text == ' ':
			if timer.running:
				timer.stop()
			else:
				timer.connect(update)
				timer.start()
		elif event.text == 'r':
			iterator -= 51
			update(1)
		elif event.text == 't':
			iterator += 49
			update(1)
		elif event.text == ',':
			iterator -= 2
			update(1)
		elif event.text == '.':
			update(1)
		elif event.text == 'b':
			iterator = -2
			update(1)
		elif event.text == 'k':
			iterator = -1
			update(1)

	# Camera style
	view.camera = 'fly' if view_style == 'fly' else 'arcball'

	# Start
	timer.connect(update)
	timer.start()

	if sys.flags.interactive != 1:
		vispy.app.run()
