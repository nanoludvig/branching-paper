import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree
import pickle
import os




class Polar:
	def __init__(self, device='cpu', dtype=torch.float, init_k=100,
				callback=None):
		self.device = device
		self.dtype = dtype
		self.k = init_k
		self.true_neighbour_max = init_k//2
		self.d = None
		self.idx = None
		self.callback = callback

	@staticmethod
	def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf, workers=-1):
		tree = cKDTree(x)
		d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=workers)
		return d[:, 1:], idx[:, 1:]

	def find_true_neighbours(self, d, dx):
		with torch.no_grad():
			z_masks = []
			i0 = 0
			batch_size = 250
			i1 = batch_size
			while True:
				if i0 >= dx.shape[0]:
					break
				# ?
				n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
				# ??
				n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]

				z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0
				z_masks.append(z_mask)

				if i1 > dx.shape[0]:
					break
				i0 = i1
				i1 += batch_size
		z_mask = torch.cat(z_masks, dim=0)
		return z_mask
	
	
	def potential(self, x, p, q, idx, d, lam, beta1, cell_type, alpha, potential):
		# Find neighbours
		full_n_list = x[idx]

		dx = x[:, None, :] - full_n_list
		z_mask = self.find_true_neighbours(d, dx)

		# Minimize size of z_mask and reorder idx and dx
		sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)

		z_mask = torch.gather(z_mask, 1, sort_idx)
		dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
		idx = torch.gather(idx, 1, sort_idx)

		m = torch.max(torch.sum(z_mask, dim=1)) + 1

		z_mask = z_mask[:, :m]
		dx = dx[:, :m]
		idx = idx[:, :m]

		# Normalise dx
		d = torch.sqrt(torch.sum(dx**2, dim=2))
		dx = dx / d[:, :, None]



		# Calculate S
		pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
		pj = p[idx]
		qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
		qj = q[idx]

		aj = alpha[idx]
		ai = alpha[:,None,:].expand(p.shape[0], idx.shape[1],1)

		"Now we calculate the G_vecctors"
		neighbors_types = cell_type[idx]  # Shape: (num_particles, max_neighbors)

		# Expand particle_types to match the shape of neighbors_types
		particle_types_expanded = cell_type.unsqueeze(1).expand(-1, idx.shape[1])

		# Get the lambda values for each pair of particle types
		lam_values = lam[particle_types_expanded, neighbors_types]  # Shape: (num_particles, max_neighbors

		g_beta = beta1.clone()
		# All other than type 1 cells have beta value 0
		g_beta[cell_type != 1] = 0.0
		# Gather the beta values of the neighbors
		beta_neighbors = g_beta[idx]  # Shape: (no_of_particles, no_of_neighbors)

		# Mask to filter out neighbors that are not type 1 
		neighbor_mask = (cell_type[idx] == 1)  # Shape: (no_of_particles, no_of_neighbors)

		# Expand beta to match the shape of beta_neighbors for broadcasting
		beta_i = beta1.unsqueeze(1).expand_as(beta_neighbors)  # Shape: (no_of_particles, no_of_neighbors)

		# Compute the difference
		beta_diff = beta_neighbors - beta_i  # Shape: (no_of_particles, no_of_neighbors)

		beta_diff = beta_diff*neighbor_mask

		beta_diff = beta_diff*z_mask.float()

		# Add a new dimension to beta_diff to make it broadcastable with dx
		beta_diff = beta_diff.unsqueeze(-1)  # Shape: (no_of_particles, no_of_neighbors, 1)

		# Compute the element-wise product
		product = dx * beta_diff  # Shape: (no_of_particles, no_of_neighbors, 3)

		G_vectors = product.sum(dim=1)  # Shape: (no_of_particles, 3)
		
		with torch.no_grad():
			# Sum over the neighbors dimension
			

			G_vectors = G_vectors/torch.norm(G_vectors, dim=1, keepdim=True)  
			inf_mask = torch.isinf(G_vectors)
			nan_mask = torch.isnan(G_vectors)
			mask = torch.any(inf_mask | nan_mask, dim=1)

			# Set corresponding rows to zero in the copy
			G_vectors[mask] = 0
			G_vectors[cell_type != 1] = 0
		


		Vij = potential(x, d, dx, lam_values, pi, pj, qi, qj, ai, aj)
		S4 = torch.sum(torch.cross(q, G_vectors, dim=1) * torch.cross(q, G_vectors, dim=1), dim=1)


		V = torch.sum(z_mask.float() * Vij)
		V -= torch.sum(S4*lam[1, 1, 4])



		return V, int(m), idx, z_mask


	def init_simulation(self, dt, lam, p, q, x, cell_type):
		assert len(x) == len(p)
		assert len(x) == len(q)

		sqrt_dt = np.sqrt(dt)

	
		x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
		p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
		q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)
		cell_type = torch.tensor(cell_type, dtype=torch.int, device=self.device)
		beta1 = torch.zeros((len(x),1), dtype=self.dtype, device=self.device) # beta1 determines the cell type (ie. wnt induces or not)
		beta2 = torch.zeros((len(x),1), dtype=self.dtype, device=self.device) # beta2 determines signaling strength for division
		alpha = torch.zeros((len(x),1), dtype=self.dtype, device=self.device) # alpha determines wedging
		lam = torch.tensor(lam, dtype=self.dtype, device=self.device)

		idx = None
		z_mask = None

		
		return lam, p, q, sqrt_dt, x, beta1, beta2, cell_type, alpha, idx, z_mask

	def update_k(self, true_neighbour_max, tstep):
		k = self.k
		fraction = true_neighbour_max / k
		if fraction < 0.25:
			k = int(0.75 * k)
		elif fraction > 0.75:
			k = int(1.5 * k)
		n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])
		self.k = k
		return k, n_update

	def time_step(self, dt, eta, lam, beta1, beta2, p, q, cell_type, a0, alpha, diffusion_constant, prefactor, sqrt_dt, tstep, x, idx, z_mask, mes_div_time, potential):

		beta1, beta2, mesenchyme_idx = self.diffuse_mesenchyme_and_update_beta(x, cell_type, diffusion_rate=diffusion_constant)



		division = False

		# Let the mesenchyme cell relax for 100 timesteps without signaling
		if tstep<100:
			wnt_threshold = 1
			cell_type = self.cell_type_determination(cell_type, beta1, wnt_threshold)

		# Now we introduce the wnt threshold and the isotropic alpha and cell division
		elif (tstep>=100):
			wnt_threshold = 0.0001
			cell_type = self.cell_type_determination(cell_type, beta1, wnt_threshold)
			division, x, p, q, beta1, beta2, cell_type, alpha, mesenchyme_idx = self.cell_division(x, p, q, beta1, beta2, cell_type, alpha, mesenchyme_idx, prefactor, dt)
			alpha[cell_type == 1] = a0
			alpha[cell_type != 1] = 0.0

			# Let the mesenchyme cell divide every mes_div_time timesteps
			if tstep % mes_div_time == 0:
				division, x, p, q, beta1, beta2, cell_type, alpha, mesenchyme_idx = self.cell_division(x, p, q, beta1, beta2, cell_type,  alpha, mesenchyme_idx, prefactor, dt, mes_div=True)
				
		
		







		k, n_update = self.update_k(self.true_neighbour_max, tstep)
		k = min(k, len(x) - 1)
		# Find potential neighbours
		if division or tstep % n_update == 0 or self.idx is None or x.shape[0]!=idx.shape[0] :
			d, idx = self.find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
			self.idx = torch.tensor(idx, dtype=torch.long, device=self.device)
			self.d = torch.tensor(d, dtype=self.dtype, device=self.device)

		idx = self.idx
		d = self.d






		# Normalise p, q
		with torch.no_grad():
			p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None]
			q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]

		

		# Calculate potential
		V, self.true_neighbour_max, idx, z_mask = self.potential(x, p, q, idx, d, lam, beta1, cell_type, alpha, potential=potential)

		# Backpropagation
		V.backward()

		with torch.no_grad():
			x += -x.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt
			p += -p.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt
			q += -q.grad * dt + eta * torch.empty(*x.shape, dtype=self.dtype, device=self.device).normal_() * sqrt_dt

			x = self.upward_bias(x, mesenchyme_idx, upward_bias_strength=0.001)
			if len(mesenchyme_idx) > 1:
				x = self.center_of_mass_repulsion(x, mesenchyme_idx, x_cm_repulsion_strength=0.0005)
				x = self.repulsion_from_neighbours(x, mesenchyme_idx, repulsion_strength=0.001)


			if self.callback is not None:
				self.callback(tstep * dt, x, p, q, lam)
		




		# Zero gradients
		x.grad.zero_()
		p.grad.zero_()
		q.grad.zero_()



		return x, p, q, beta1, beta2, cell_type, alpha, idx, z_mask 

	def simulation(self, x, p, q, lam, cell_type, a0, diffusion_constant, prefactor, time_steps, eta, mes_div_time, potential, yield_every=1, dt=0.1):
		lam, p, q, sqrt_dt, x, beta1, beta2, cell_type, alpha, idx, z_mask = self.init_simulation(dt, lam, p, q, x, cell_type)
		tstep = 0


		while True:
			tstep += 1
			x, p, q, beta1, beta2, cell_type, alpha, idx, z_mask = self.time_step(dt, eta, lam, beta1, beta2, p, q, cell_type, a0, alpha, diffusion_constant , prefactor, sqrt_dt, tstep, x, idx, z_mask, mes_div_time, potential=potential)
						
			if tstep % yield_every == 0:
				xx = x.detach().to("cpu").numpy().copy()
				pp = p.detach().to("cpu").numpy().copy()
				qq = q.detach().to("cpu").numpy().copy()
				cc = cell_type.detach().to("cpu").numpy().copy()
				yield xx, pp, qq, cc, tstep
			
			if tstep >= time_steps:
				xx = x.detach().to("cpu").numpy().copy()
				pp = p.detach().to("cpu").numpy().copy()
				qq = q.detach().to("cpu").numpy().copy()
				cc = cell_type.detach().to("cpu").numpy().copy()
				yield xx, pp, qq, cc, tstep
				break



	@staticmethod
	def cell_division(x, p, q, beta1, beta2, cell_type, alpha, mesenchyme_idx, prefactor, dt, mes_div=False):

		# Return if there is no signaling
		if torch.sum(beta2) < 1e-8:
			return False, x, p, q, beta1, beta2, cell_type, alpha, mesenchyme_idx
		
		division = False
		# make it impossible for mesenchyme cells to divide (unless mes_div is True)
		beta2[mesenchyme_idx]=0.0
		d_prob = (beta2 * dt)*prefactor
		if mes_div: # if mes_div is True, then mesenchyme cells must divide
			d_prob[mesenchyme_idx]=1.0

		# flip coins
		draw = torch.empty_like(beta2).uniform_()
		# find successes
		events = draw < d_prob
		numdiv = torch.sum(events)


		if numdiv > 0:
			with torch.no_grad():
				division = True
				# find cells that will divide
				idxs = torch.nonzero(events)[:, 0]
				x0 = x[idxs, :]
				p0 = p[idxs,:] 
				q0 = torch.empty_like(x0).normal_()
				q0 /= torch.sqrt(torch.sum(q0**2, dim=1))[:, None]
				type = cell_type[idxs]
				b01 = beta1[idxs] 
				b02 = beta2[idxs]
				a0_array = alpha[idxs]
				# move the new cells according to gradient of the mesenchyme field
				move = torch.empty_like(x0).normal_()
				move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

				# place new cells
				x0 = x0 + move

				# append new cell data to the system state
				x = torch.cat((x, x0))
				p = torch.cat((p, p0))
				q = torch.cat((q, q0))
				beta1 = torch.cat((beta1, b01))
				beta2 = torch.cat((beta2, b02))
				cell_type = torch.cat((cell_type, type))
				alpha = torch.cat((alpha, a0_array))
				new_indices = torch.arange(x.size(0) - numdiv, x.size(0), device=x.device)
				new_mesenchyme_idx = new_indices[cell_type[new_indices] == 2]
				mesenchyme_idx = torch.cat((mesenchyme_idx, new_mesenchyme_idx))


		x.requires_grad = True
		p.requires_grad = True
		q.requires_grad = True

		return division, x, p, q, beta1, beta2, cell_type, alpha, mesenchyme_idx

	@staticmethod
	def diffuse_mesenchyme_and_update_beta(x, cell_type, diffusion_rate):
		mesenchyme_idx = torch.nonzero(cell_type == 2)[:, 0]
		if mesenchyme_idx.numel() < 1:
			return torch.zeros_like(cell_type), torch.zeros_like(cell_type), None
		with torch.no_grad():
			rij = x[:, None, :] - x[mesenchyme_idx,:]
			d2 = torch.sum(rij**2, dim=2)
			d2 = torch.sqrt(d2)
			gauss1 = torch.exp(-d2 / diffusion_rate[0])
			beta1 = torch.sum(gauss1, dim=1)
			gauss2 = torch.exp(-d2 / diffusion_rate[1]) 
			beta2 = torch.sum(gauss2, dim=1)
		return beta1, beta2, mesenchyme_idx
	

	def cell_type_determination(self, cell_type, beta1, wnt_threshold):
		cell_type[(cell_type == 1) & (beta1 < wnt_threshold)] = 0
		cell_type[(cell_type == 0) & (beta1 >= wnt_threshold)] = 1

		return cell_type
	
	def upward_bias(self, x, mesenchyme_idx, upward_bias_strength=0.001):
		# Define the upward bias vector
		x[mesenchyme_idx] += torch.tensor([upward_bias_strength, 0.0, 0.0], device=self.device, dtype=self.dtype)
		return x
	
	def center_of_mass_repulsion(self, x,mesenchyme_idx, x_cm_repulsion_strength=0.0005):
		x_cm = torch.mean(x[mesenchyme_idx], dim=0)
		x_cm_repulsion_direction = x[mesenchyme_idx]-x_cm
		x_norm_repulsion = x_cm_repulsion_direction/torch.norm(x_cm_repulsion_direction, dim=1, keepdim=True)
		x[mesenchyme_idx] += x_cm_repulsion_strength*x_norm_repulsion
		return x
	
	def repulsion_from_neighbours(self, x, mesenchyme_idx, repulsion_strength=0.001):
		positions = x[mesenchyme_idx]
		# Compute pairwise direction vectors
		pairwise_vectors = positions.unsqueeze(1) - positions.unsqueeze(0)  # Shape: (N, N, 3)
		# Normalize direction vectors
		pairwise_distances = torch.norm(pairwise_vectors, dim=-1)  # Shape: (N, N)
		diagonal_mask = torch.eye(len(mesenchyme_idx), device=pairwise_distances.device).bool()
		pairwise_distances[diagonal_mask] = float('inf')  # Set diagonal entries to infinity
		normalized_vectors = pairwise_vectors / pairwise_distances.unsqueeze(-1)
		# Compute weights as the inverse of distances
		weights = 1.0 / pairwise_distances  # Shpe: (N, N)
		weights[torch.arange(len(mesenchyme_idx)), torch.arange(len(mesenchyme_idx))] = 0  # No self-weight
		# Apply weights to normalized direction vectors
		weighted_vectors = normalized_vectors * weights.unsqueeze(-1)  # Shape: (N, N, 3)
		# Sum weighted vectors for each cell
		bias_vectors = torch.sum(weighted_vectors, dim=1)  # Shape: (N, 3)
		# Normalize the bias vectors
		normalized_bias_vectors = bias_vectors / torch.norm(bias_vectors, dim=-1, keepdim=True)
		# Apply the bias to cell positions
		x[mesenchyme_idx] += repulsion_strength * normalized_bias_vectors
		return x



def save(tup, name, save_dir):
	print(f'saving {save_dir}/sim_{name}.npy')
	with open(f'{save_dir}/sim_{name}.npy', 'wb') as f:
		pickle.dump(tup, f)




def pot(x, d, dx, lam_values, pi, pj, qi, qj, ai, aj):
	alphamean = (ai+aj)*0.5

	with torch.no_grad(): # tube-tube rejection
		no_fuse_mask = (torch.sum(pi * pj , dim = 2) < 0.0)*(torch.sum(-dx * pj , dim = 2) < 0.0)
		sum_of_fuse = torch.sum(no_fuse_mask)
		lam_values[no_fuse_mask] = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0], device=lam_values.device, dtype=lam_values.dtype)
	
	pi = pi - alphamean*dx
	pj = pj + alphamean*dx
	pi = pi/torch.sqrt(torch.sum(pi ** 2, dim=2))[:, :, None]
	pj = pj/torch.sqrt(torch.sum(pj ** 2, dim=2))[:, :, None]
	S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
	S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
	S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)
	S2, S3 = torch.abs(S2), torch.abs(S3)
	S = lam_values[...,0] + lam_values[...,1] * S1 + lam_values[...,2] * S2 + lam_values[...,3] * S3 
	Vij = (torch.exp(-d) - S * torch.exp(-d/5))

	return Vij


def load_initial_structure(file_path):
	data = np.load(file_path, allow_pickle=True)
	x, p, q = data
	cell_type = np.array([0]*len(x))
	return x, p, q, cell_type


def add_mesenchyme(x, p, q, cell_type, idx=0, distance_to_epithelial_cells=4.5):
	# Generate a mesenchyme cell right above the epithelial cell with index `idx`
	x_point = x[idx] + p[idx] * distance_to_epithelial_cells

	# Initialize polarity vectors `p_point` and `q_point` for new mesenchyme cells (will not be used)
	p_point = np.ones((1, 3))
	q_point = np.ones((1, 3))
		
	# Append new mesenchyme cells to the existing structure
	x = np.concatenate((x, x_point))
	p = np.concatenate((p, p_point))
	q = np.concatenate((q, q_point))
	cell_type = np.concatenate((cell_type, np.array([2])))
	
	return x, p, q, cell_type






def main(output_folder_path, sim_name, time_steps, x, p, q, lam, cell_type, a0, diffusion_constant, prefactor, mes_div_time, eta, yield_every, save_every, potential):
	
	try:
		os.makedirs(f'{output_folder_path}')
	except OSError:
		pass


	sim = Polar(device="cuda", init_k=50) # device: cpu or cuda
	runner = sim.simulation(x, p, q, lam, cell_type, a0, diffusion_constant, prefactor, time_steps=time_steps, eta=eta, mes_div_time = mes_div_time, yield_every=yield_every, potential=potential)

	# Running the simulation
	data = []  # For storing data
	
	print('Starting')

	

	with open(__file__) as f:
		s = f.read()
	with open(f'{output_folder_path}/sim_{sim_name}.py', 'w') as f:
		f.write(s)

	


	for xx, pp, qq, cc, tstep in runner:
		mes_idx = np.where(cc == 2)
		data.append((xx, pp, qq, cc))
		xx_data, pp_data, qq_data, cc_data = zip(*data)
		print(f'Running {tstep} of {time_steps}    ({len(xx) - len(mes_idx[0])} cells) + ({len(mes_idx[0])} mesenchyme cells) + ({np.count_nonzero(cc == 1)} wnt cells)')
		

		if tstep % save_every == 0:
			print(f'Saving at timestep {tstep}')
			xx_data, pp_data, qq_data, cc_data = zip(*data)
			save((xx_data, pp_data, qq_data, cc_data), sim_name, output_folder_path)




diffusion_constant = [1, 2] # diffusion constant for cell_type determination and division respectively
prefactor = 0.1 # prefactor for division signaling

file_path = 'initial_tube.npy' # initial tube
eta = 0.001 # noise
a0 = 0.5 # degree of isotropic wedging
yield_every = 100 # keep every 100 time steps (recommended for intermediate sized simulations)
save_every = 1000 # save the yielded data every 1000 time steps 
idx = [0] # index of the epithelial cell to which the mesenchyme cell is added (0 for initial tube)
mes_div_time = 11000 # mesenchyme cell division frequency (in time steps)
number_of_generations = 2 # number if generations that mesenchyme cells will divide (simulation stops after this many generations)

time_steps = mes_div_time*number_of_generations # total number of time steps for the simulation





x, p, q, cell_type = load_initial_structure(file_path) # load the initial structure
x, p, q, cell_type = add_mesenchyme(x, p, q, cell_type, idx) # add the mesenchyme cell


output_folder_path = f'test_output'  # folder to save the simulation data
sim_name = f'test2' 				 # name of the simulation





lam1, lam2, lam3, lam4 = 0.5, 0.4, 0.1, 0.3  # coupling strengths for different interactions

# eq. distance between mesenchyme cells
eq_dis = 10.0
lam22_1 = 5*np.exp(-eq_dis*4/5)


lam_00 = np.array([0.0, lam1, lam2, lam3, 0.0]) #  cell type 0-0 (normal epitheilial cells)
lam_01 = np.array([0.0, lam1, lam2, lam3, 0.0]) #  cell type 0-1 (normal epitheilial cell and wnt induced cell)
lam_11 = np.array([0.0, 0.5, 0.4, 0.1, lam4]) #  cell type 1-1 (wnt induced cells)
lam_02 = np.array([0.1, 0.0,  0.0, 0.0, 0.0]) #  cell type 0-2 (normal epitheilial cell and mesenchyme cell)
lam_12 = np.array([0.2, 0.0,  0.0, 0.0, 0.0]) # cell type 1-2 (wnt induced cell and mesenchyme cell)
lam_22 = np.array([lam22_1, 0.0,  0.0, 0.0, 0.0]) # cell type 2-2 (mesenchyme cells)



# the complete interaction tensor
lam = np.array([[lam_00, lam_01, lam_02], 
				[lam_01, lam_11, lam_12], 
				[lam_02, lam_12, lam_22]])


# run the simulation
main(output_folder_path, sim_name, time_steps, x, p, q, lam, cell_type, a0, diffusion_constant, prefactor, mes_div_time, eta, yield_every, save_every, potential=pot)