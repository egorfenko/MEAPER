# Copyright (c) 2026 Egor Fenko. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for full license information.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, NonlinearConstraint
import time
import math

# FOURIER FEATURE
class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim, num_freqs=10):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        freq_bands = 2 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer('freq_bands', freq_bands)
        self.out_dim = in_dim * (1 + 2 * num_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * math.pi))
            out.append(torch.cos(x * freq * math.pi))
        return torch.cat(out, dim=-1)

# IKNET
# Used GELU instead of RELU because it works better 
# with physics as a differentiable function
class IKNet(nn.Module):
    def __init__(self, cond_dim=31, num_variables=8, hidden_dim=16, m_components=10, num_freqs=4):
        super().__init__()
        self.num_variables = num_variables
        self.m = m_components
        
        self.fourier_enc = FourierFeatureEncoding(in_dim=cond_dim, num_freqs=num_freqs)
        fourier_cond_dim = self.fourier_enc.out_dim
        
        self.hypernet = nn.Sequential(
            nn.Linear(fourier_cond_dim, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.BatchNorm1d(256)
        )
        
        self.gk_shapes = []
        total_params_per_gk = []
        
        for k in range(num_variables):
            in_dim = k 
            s1 = (in_dim, hidden_dim); b1 = hidden_dim
            s2 = (hidden_dim, hidden_dim); b2 = hidden_dim
            s3 = (hidden_dim, 3 * m_components); b3 = 3 * m_components 
            
            self.gk_shapes.append([s1, b1, s2, b2, s3, b3])
            params = (in_dim * hidden_dim) + hidden_dim + \
                     (hidden_dim * hidden_dim) + hidden_dim + \
                     (hidden_dim * 3 * m_components) + (3 * m_components)
            total_params_per_gk.append(params)
            
        self.weight_projections = nn.ModuleList([
            nn.Linear(256, p) for p in total_params_per_gk
        ])
        
        for proj in self.weight_projections:
            nn.init.uniform_(proj.weight, -1e-4, 1e-4)
            nn.init.zeros_(proj.bias)

    def forward_gk(self, x_prev, weights, shapes):
        w1_size, b1_size, w2_size, b2_size, w3_size, b3_size = shapes
        idx = 0
        
        if w1_size[0] == 0:
            b1 = weights[:, idx : idx+b1_size]; idx += b1_size
            h = b1 
        else:
            w1 = weights[:, idx : idx+w1_size[0]*w1_size[1]].view(-1, w1_size[1], w1_size[0]); idx += w1_size[0]*w1_size[1]
            b1 = weights[:, idx : idx+b1_size]; idx += b1_size
            h = x_prev.unsqueeze(2)
            h = torch.bmm(w1, h).squeeze(2) + b1
            
        h = F.gelu(h)
        h = h.unsqueeze(2)
        w2 = weights[:, idx : idx+w2_size[0]*w2_size[1]].view(-1, w2_size[1], w2_size[0]); idx += w2_size[0]*w2_size[1]
        b2 = weights[:, idx : idx+b2_size]; idx += b2_size
        h = torch.bmm(w2, h).squeeze(2) + b2
        h = F.gelu(h)
        
        h = h.unsqueeze(2)
        w3 = weights[:, idx : idx+w3_size[0]*w3_size[1]].view(-1, w3_size[1], w3_size[0]); idx += w3_size[0]*w3_size[1]
        b3 = weights[:, idx : idx+b3_size]
        h = torch.bmm(w3, h).squeeze(2) + b3
        
        return h

    def forward(self, condition, y_true=None):
        batch_size = condition.size(0)
        
        cond_encoded = self.fourier_enc(condition)
        h_cond = self.hypernet(cond_encoded)
        
        gmm_params = []
        sampled_y = []
        
        for k in range(self.num_variables):
            theta_k = self.weight_projections[k](h_cond)
            
            if k == 0:
                y_prev = torch.empty((batch_size, 0), device=condition.device)
            else:
                y_prev = y_true[:, :k] if y_true is not None else torch.stack(sampled_y, dim=1)
                
            out_k = self.forward_gk(y_prev, theta_k, self.gk_shapes[k])
            
            priors = F.softmax(out_k[:, :self.m], dim=-1) 
            means = out_k[:, self.m : 2*self.m]
            stds = F.softplus(out_k[:, 2*self.m :]) + 1e-5
            
            gmm_params.append((priors, means, stds))
            
            if y_true is None:
                idx = torch.multinomial(priors, 1)
                selected_mean = torch.gather(means, 1, idx)
                selected_std = torch.gather(stds, 1, idx)
                sample = torch.normal(selected_mean, selected_std)
                sampled_y.append(sample.squeeze(1))
                
        return gmm_params if y_true is not None else (torch.stack(sampled_y, dim=1), gmm_params)

# Custom GAUSS-NEWTON SOLVER (Kinematics from predicted diagonals)
class GaussNewtonSolver:
    def __init__(self, eps=1e-12):
        self.eps = eps

    def calculate_new_location(self, prev_layer, angles, diagonals):
        theta, phi = np.split(angles, 2)
        disp = np.column_stack((diagonals * np.sin(phi) * np.cos(theta), 
                                diagonals * np.sin(phi) * np.sin(theta), 
                                diagonals * np.cos(phi)))
        return prev_layer + disp, np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)

    def jacobian(self, loc, diag, s_phi, c_phi, s_theta, c_theta):
        N = loc.shape[0]; J = np.zeros((N, 2 * N))
        u = (loc - np.roll(loc, -1, axis=0)) / (np.linalg.norm(loc - np.roll(loc, -1, axis=0), axis=1)[:, None] + self.eps)
        dp_dtheta = (diag * s_phi)[:, None] * np.column_stack((-s_theta, c_theta, np.zeros(N)))
        dp_dphi = diag[:, None] * np.column_stack((c_phi * c_theta, c_phi * s_theta, -s_phi))
        
        for j in range(N):
            jp = (j + 1) % N  
            J[j, j], J[j, N+j] = np.dot(u[j], dp_dtheta[j]), np.dot(u[j], dp_dphi[j])
            J[j, jp], J[j, N+jp] = -np.dot(u[j], dp_dtheta[jp]), -np.dot(u[j], dp_dphi[jp])
        return J

    def solve(self, prev_layer, init_angles, diag, targ_dist, acc=1e-4, damp=1e-6, max_iter=75, lr=0.05):
        angles = init_angles.copy()
        for _ in range(max_iter):
            loc, s_p, c_p, s_t, c_t = self.calculate_new_location(prev_layer, angles, diag)
            r = np.linalg.norm(loc - np.roll(loc, -1, axis=0), axis=1) - targ_dist
            if 0.5 * np.sum(r**2) < acc: break
            J = self.jacobian(loc, diag, s_p, c_p, s_t, c_t)
            angles += lr * np.linalg.solve(J.T @ J + damp * np.eye(2 * len(diag)), -J.T @ r)
        return self.calculate_new_location(prev_layer, angles, diag)[0], angles

# SLSQP POLISHER OF COORDINATES
class CoordinateOptimizer:
    @staticmethod
    def optimize_layer_1(L0, target_centroid, target_normal, rough_nodes, nn_diags, nn_edges):
        def objective(x):
            L1 = x.reshape(4, 3)
            diags = np.linalg.norm(L1 - L0, axis=1)
            edges = np.linalg.norm(L1 - np.roll(L1, -1, axis=0), axis=1)
            
            E1 = L1 - np.roll(L1, -1, axis=0)
            E0 = L0 - np.roll(L0, -1, axis=0)
            cross_2d = E1[:, 0] * E0[:, 1] - E1[:, 1] * E0[:, 0]
            twist_penalty = np.sum(cross_2d**2)
            
            v1 = L1[1] - L1[0]
            v2 = L1[3] - L1[0]
            cross = np.cross(v1, v2)
            norm_val = np.linalg.norm(cross)
            l_normal = cross / (norm_val + 1e-8)
            orientation_penalty = np.sum((l_normal - target_normal)**2)
            
            return np.sum((diags - nn_diags)**2) + np.sum((edges - nn_edges)**2) + (twist_penalty * 2.0) + (orientation_penalty * 5.0)

        def centroid_constraint(x):
            L1 = x.reshape(4, 3)
            return np.mean(L1, axis=0) - target_centroid

        def length_constraints(x):
            L1 = x.reshape(4, 3)
            diags = np.linalg.norm(L1 - L0, axis=1)
            edges = np.linalg.norm(L1 - np.roll(L1, -1, axis=0), axis=1)
            return np.concatenate((diags, edges))

        constraints = [
            {'type': 'eq', 'fun': centroid_constraint},
            NonlinearConstraint(length_constraints, 5, 8)
        ]

        x0 = rough_nodes.flatten()
        result = minimize(objective, x0, method='SLSQP', constraints=constraints, options={'ftol': 1e-3, 'disp': False})

        perfect_L1 = result.x.reshape(4, 3)
        perfect_diags = np.linalg.norm(perfect_L1 - L0, axis=1)
        perfect_edges = np.linalg.norm(perfect_L1 - np.roll(perfect_L1, -1, axis=0), axis=1)
        return perfect_L1, perfect_diags, perfect_edges

    @staticmethod
    def optimize_layer_2(L1, target_L2_centroid, target_normal, rough_nodes, nn_diags):
        def objective(x):
            L2 = x.reshape(4, 3)
            diags = np.linalg.norm(L2 - L1, axis=1)
            
            E2 = L2 - np.roll(L2, -1, axis=0)
            E1 = L1 - np.roll(L1, -1, axis=0)
            cross_2d = E2[:, 0] * E1[:, 1] - E2[:, 1] * E1[:, 0]
            twist_penalty = 0.1 * np.sum(cross_2d**2)
            
            v1 = L2[1] - L2[0]
            v2 = L2[3] - L2[0]
            cross = np.cross(v1, v2)
            norm_val = np.linalg.norm(cross)
            l_normal = cross / (norm_val + 1e-8)
            orientation_penalty = np.sum((l_normal - target_normal)**2)
            
            return np.sum((diags - nn_diags)**2) + (twist_penalty * 2.0) + (orientation_penalty * 5.0)

        def square_and_centroid_constraints(x):
            L2 = x.reshape(4, 3)
            centroid_err = np.mean(L2, axis=0) - target_L2_centroid
            sides_sq = np.sum((L2 - np.roll(L2, -1, axis=0))**2, axis=1)
            side_err = sides_sq - 4.0
            diag1_sq = np.sum((L2[0] - L2[2])**2)
            diag2_sq = np.sum((L2[1] - L2[3])**2)
            cross_err = np.array([diag1_sq - 8.0, diag2_sq - 8.0])
            return np.concatenate((centroid_err, side_err, cross_err))

        def vertical_length_constraints(x):
            L2 = x.reshape(4, 3)
            return np.linalg.norm(L2 - L1, axis=1)

        constraints = [
            {'type': 'eq', 'fun': square_and_centroid_constraints},
            NonlinearConstraint(vertical_length_constraints, 5, 8)
        ]

        x0 = rough_nodes.flatten()
        result = minimize(objective, x0, method='SLSQP', constraints=constraints, options={'ftol': 1e-6, 'disp': False, 'maxiter': 250})

        perfect_L2 = result.x.reshape(4, 3)
        perfect_diags = np.linalg.norm(perfect_L2 - L1, axis=1)
        perfect_edges = np.array([2.0, 2.0, 2.0, 2.0])
        return perfect_L2, perfect_diags, perfect_edges

def compute_conditional_vectors(X_norm, x_mean, x_std):
    X_phys = (X_norm * x_std) + x_mean
    v_ee = X_phys[:, 19:22]
    v_norm = X_phys[:, 22:25]
    
    u_ee = v_ee / (torch.linalg.norm(v_ee, dim=-1, keepdim=True) + 1e-8)
    u_ee = torch.where(u_ee[:, 2:3] < 0, -u_ee, u_ee)
    u_norm = v_norm / (torch.linalg.norm(v_norm, dim=-1, keepdim=True) + 1e-8)
    u_norm = torch.where(u_norm[:, 2:3] < 0, -u_norm, u_norm)
    return u_ee, u_norm

# IK PIPELINE
class ActuatorPredictorApp:
    def __init__(self, model_path, json_params_path, batch_size=10, num_freqs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size 
        
        self.model = IKNet(
            cond_dim=31, 
            num_variables=8, 
            hidden_dim=16, 
            m_components=15,
            num_freqs=num_freqs
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.solver = GaussNewtonSolver()
        self.optimizer = CoordinateOptimizer()
        
        # Load parameters from JSON instead of HDF5
        with open(json_params_path, 'r') as f:
            stats = json.load(f)
            self.x_mean = np.array(stats['X_mean'], dtype=np.float64)
            self.x_std = np.array(stats['X_std'], dtype=np.float64)
            self.y_mean = np.array(stats['Y_mean'], dtype=np.float64)
            self.y_std = np.array(stats['Y_std'], dtype=np.float64)
            
        self.spatial_idx = [i for i in range(25) if i not in [12, 13, 14, 22, 23, 24]]

    def predict_next_layer(self, base_nodes, c_cent, c_norm, c_edges, t_cent, t_norm, layer_idx=1, target_edge_range=None, use_slsqp=True):
        X_raw = np.hstack([np.tile(np.hstack([(base_nodes - c_cent).flatten(), c_norm, c_edges]), (self.batch_size, 1)), 
                            (t_cent + np.random.normal(0, 0.0005, (self.batch_size, 3))) - c_cent, 
                            np.tile(t_norm, (self.batch_size, 1))]).astype(np.float64)
        
        X_norm = X_raw.copy()
        X_norm[:, self.spatial_idx] = (X_raw[:, self.spatial_idx] - self.x_mean[self.spatial_idx]) / self.x_std[self.spatial_idx]
        
        x_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        u_ee, u_norm = compute_conditional_vectors(x_tensor, torch.tensor(self.x_mean, device=self.device, dtype=torch.float32), torch.tensor(self.x_std, device=self.device, dtype=torch.float32))
        full_condition = torch.cat([x_tensor, u_ee, u_norm], dim=-1)

        t_nn_start = time.perf_counter()
        with torch.no_grad():
            model_out = self.model(full_condition)
            Y_preds = (model_out[0].cpu().numpy() * self.y_std) + self.y_mean
        time_nn = time.perf_counter() - t_nn_start
        
        if target_edge_range is not None:
            min_edge, max_edge = target_edge_range
            pred_edges_batch = Y_preds[:, 4:8]
            valid_mask = np.all((pred_edges_batch >= min_edge) & (pred_edges_batch <= max_edge), axis=1)
            
            if np.any(valid_mask):
                Y_final = np.mean(Y_preds[valid_mask], axis=0)
            else:
                penalty = np.sum(np.maximum(0, min_edge - pred_edges_batch) + np.maximum(0, pred_edges_batch - max_edge), axis=1)
                Y_final = Y_preds[np.argmin(penalty)]
        else:
            Y_final = np.mean(Y_preds, axis=0)
            
        pred_diags, pred_edges = Y_final[0:4], Y_final[4:8]

        side = np.mean(c_edges) if np.mean(c_edges) > 0 else 1.0
        angle_guess = np.arcsin(side / (2 * np.mean(pred_diags))) if 2 * np.mean(pred_diags) > side else np.pi/6
        init_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4, angle_guess, angle_guess, angle_guess, angle_guess])
        
        t_gn_start = time.perf_counter()
        rough_nodes, _ = self.solver.solve(base_nodes, init_angles, pred_diags, pred_edges)
        time_gn = time.perf_counter() - t_gn_start
        
        t_opt_start = time.perf_counter()
        if use_slsqp:
            if layer_idx == 1:
                perfect_nodes, perfect_diags, perfect_edges = self.optimizer.optimize_layer_1(
                    base_nodes, t_cent, t_norm, rough_nodes, pred_diags, pred_edges
                )
            else:
                perfect_nodes, perfect_diags, perfect_edges = self.optimizer.optimize_layer_2(
                    base_nodes, t_cent, t_norm, rough_nodes, pred_diags
                )
        else:
            perfect_nodes = rough_nodes
            perfect_diags = pred_diags
            perfect_edges = pred_edges
        time_opt = time.perf_counter() - t_opt_start
        
        disp = perfect_nodes - base_nodes
        phi = np.arccos(disp[:, 2] / perfect_diags)
        theta = np.arctan2(disp[:, 1], disp[:, 0])

        return {
            "predicted_diagonals": perfect_diags, 
            "predicted_target_edges": perfect_edges,
            "next_layer_nodes": perfect_nodes,
            "convergence_angles": np.concatenate((theta, phi)),
            "timings": {"nn": time_nn * 1000, "gn": time_gn * 1000, "opt": time_opt * 1000}
        }

# INTERACTIVE VISUALIZER
class ActuatorVisualizer:
    def __init__(self, predictor):
        self.predictor = predictor
        
        self.base_nodes = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]], dtype=np.float64)
        self.curr_centroid = np.mean(self.base_nodes, axis=0)
        self.curr_normal = np.array([0.0, 0.0, 1.0])
        self.curr_edges = [np.linalg.norm(self.base_nodes[(i+1)%4] - self.base_nodes[i]) for i in range(4)]
        
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.30)

        ax_x = plt.axes([0.2, 0.20, 0.65, 0.03])
        ax_y = plt.axes([0.2, 0.15, 0.65, 0.03])
        ax_z = plt.axes([0.2, 0.10, 0.65, 0.03])
        
        ax_pitch = plt.axes([0.2, 0.05, 0.30, 0.03])
        ax_yaw = plt.axes([0.55, 0.05, 0.30, 0.03])
        
        #ax_check = plt.axes([0.80, 0.85, 0.15, 0.05])
        # self.toggle_slsqp = CheckButtons(ax_check, ['SLSQP Polish'], [True])

        self.slider_x = Slider(ax_x, 'Target 2 X', -2.0, 2.0, valinit=0.0)
        self.slider_y = Slider(ax_y, 'Target 2 Y', -2.0, 2.0, valinit=0.0)
        self.slider_z = Slider(ax_z, 'Target 2 Z', 9.5, 11.5, valinit=9.0)
        
        self.slider_pitch = Slider(ax_pitch, 'EE Pitch Dev (°)', -89.0, 89.0, valinit=0.0)
        self.slider_yaw = Slider(ax_yaw, 'EE Yaw Dev (°)', -180.0, 180.0, valinit=0.0)

        self.slider_x.on_changed(self.update)
        self.slider_y.on_changed(self.update)
        self.slider_z.on_changed(self.update)
        self.slider_pitch.on_changed(self.update)
        self.slider_yaw.on_changed(self.update)
        # self.toggle_slsqp.on_clicked(self.update)
        
        self.update(None)
        plt.show()

    def get_layer_properties(self, nodes):
        centroid = np.mean(nodes, axis=0)
        edges = np.array([np.linalg.norm(nodes[(i+1)%len(nodes)] - nodes[i]) for i in range(len(nodes))])
        v1 = nodes[1] - nodes[0]
        v2 = nodes[3] - nodes[0]
        normal = np.cross(v1, v2)
        norm_val = np.linalg.norm(normal)
        normal = normal / norm_val if norm_val > 1e-8 else np.array([0.0, 0.0, 1.0])
        return centroid, normal, edges

    def enforce_angle_limits(self, target_vec, ref_vec, min_angle=None, max_angle=None):
        t_hat = target_vec / (np.linalg.norm(target_vec) + 1e-8)
        r_hat = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)
        angle = np.arccos(np.clip(np.dot(t_hat, r_hat), -1.0, 1.0))
        
        clamped_angle = angle
        if min_angle is not None and angle < min_angle:
            clamped_angle = min_angle
        if max_angle is not None and angle > max_angle:
            clamped_angle = max_angle
            
        if clamped_angle != angle:
            axis = np.cross(r_hat, t_hat)
            axis_norm = np.linalg.norm(axis)
            
            if axis_norm < 1e-8:
                axis = np.array([1.0, 0.0, 0.0]) if abs(r_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            else:
                axis = axis / axis_norm
            
            return r_hat * np.cos(clamped_angle) + np.cross(axis, r_hat) * np.sin(clamped_angle)
        return t_hat

    def update(self, val):
        tx = self.slider_x.val
        ty = self.slider_y.val
        tz = self.slider_z.val
        # use_slsqp = self.toggle_slsqp.get_status()[0]
        use_slsqp = True
        target_pos_2 = np.array([tx, ty, tz])

        t1_z = tz / 2  
        r = np.hypot(tx, ty)
        if r > 0.56:
            t1_x = 0.56 * (tx / r)
            t1_y = 0.56 * (ty / r)
        else:
            t1_x = tx
            t1_y = ty
            
        target_pos_1 = np.array([t1_x, t1_y, t1_z])
        norm_val_1 = np.linalg.norm(target_pos_1)
        target_normal_1 = target_pos_1 / norm_val_1 if norm_val_1 > 0 else np.array([0.0, 0.0, 1.0])

        v_L0_L2 = target_pos_2 - self.curr_centroid
        v_L0_L2_norm = v_L0_L2 / (np.linalg.norm(v_L0_L2) + 1e-8)
        
        base_pitch = np.arccos(np.clip(v_L0_L2_norm[2], -1.0, 1.0))
        base_yaw = np.arctan2(v_L0_L2_norm[1], v_L0_L2_norm[0])

        pitch_dev = np.radians(self.slider_pitch.val)
        yaw_dev = np.radians(self.slider_yaw.val)
        
        final_pitch = base_pitch + pitch_dev
        final_yaw = base_yaw + yaw_dev

        target_normal_2_raw = np.array([
            np.sin(final_pitch) * np.cos(final_yaw),
            np.sin(final_pitch) * np.sin(final_yaw),
            np.cos(final_pitch)
        ])

        v_02 = target_pos_2 - self.curr_centroid
        target_normal_2 = self.enforce_angle_limits(
            target_normal_2_raw, 
            v_02, 
            max_angle=np.radians(40.0) 
        )

        res_1 = self.predictor.predict_next_layer(
            self.base_nodes, self.curr_centroid, self.curr_normal, 
            self.curr_edges, target_pos_1, target_normal_1,
            layer_idx=1, target_edge_range=(5.5, 7.5), use_slsqp=use_slsqp
        )
        nodes_1 = res_1["next_layer_nodes"]

        centroid_1, normal_1, edges_1 = self.get_layer_properties(nodes_1)

        v_12 = target_pos_2 - centroid_1
        v_12_hat = v_12 / (np.linalg.norm(v_12) + 1e-8)
        tilt_v12_rad = np.arccos(np.clip(np.dot(v_12_hat, normal_1), -1.0, 1.0))
        
        if np.degrees(tilt_v12_rad) > 20.0:
            target_normal_2 = self.enforce_angle_limits(
                target_normal_2, 
                normal_1, 
                min_angle=np.radians(20.0)
            )

        res_2 = self.predictor.predict_next_layer(
            nodes_1, centroid_1, normal_1, edges_1, 
            target_pos_2, target_normal_2,
            layer_idx=2, target_edge_range=(1.9, 2.1), use_slsqp=use_slsqp
        )
        nodes_2 = res_2["next_layer_nodes"]
        calc_centroid_2 = np.mean(nodes_2, axis=0)
        
        pos_accuracy = np.linalg.norm(calc_centroid_2 - target_pos_2)
        
        total_nn = res_1["timings"]["nn"] + res_2["timings"]["nn"]
        total_gn = res_1["timings"]["gn"] + res_2["timings"]["gn"]
        total_opt = res_1["timings"]["opt"] + res_2["timings"]["opt"]

        self.ax.clear()
        
        self.ax.scatter(self.base_nodes[:, 0], self.base_nodes[:, 1], self.base_nodes[:, 2], c='k', label='Base')
        self.ax.scatter(nodes_1[:, 0], nodes_1[:, 1], nodes_1[:, 2], c='b', label='Layer 1')
        self.ax.scatter(nodes_2[:, 0], nodes_2[:, 1], nodes_2[:, 2], c='r', label='Layer 2')
        
        for i in range(4):
            next_i = (i + 1) % 4
            self.ax.plot([self.base_nodes[i, 0], self.base_nodes[next_i, 0]], 
                         [self.base_nodes[i, 1], self.base_nodes[next_i, 1]], 
                         [self.base_nodes[i, 2], self.base_nodes[next_i, 2]], 'k-')
            self.ax.plot([nodes_1[i, 0], nodes_1[next_i, 0]], 
                         [nodes_1[i, 1], nodes_1[next_i, 1]], 
                         [nodes_1[i, 2], nodes_1[next_i, 2]], 'b-')
            self.ax.plot([nodes_2[i, 0], nodes_2[next_i, 0]], 
                         [nodes_2[i, 1], nodes_2[next_i, 1]], 
                         [nodes_2[i, 2], nodes_2[next_i, 2]], 'r-')
            
            # Inter-layer diagonal struts
            self.ax.plot([self.base_nodes[i, 0], nodes_1[i, 0]], 
                         [self.base_nodes[i, 1], nodes_1[i, 1]], 
                         [self.base_nodes[i, 2], nodes_1[i, 2]], 'g--')
            self.ax.plot([nodes_1[i, 0], nodes_2[i, 0]], 
                         [nodes_1[i, 1], nodes_2[i, 1]], 
                         [nodes_1[i, 2], nodes_2[i, 2]], 'm--')

        self.ax.quiver(target_pos_2[0], target_pos_2[1], target_pos_2[2], 
                    target_normal_2[0], target_normal_2[1], target_normal_2[2], 
                    length=1.5, color='g', label='Target EE Normal')

        info_str = (#f"L1 PROJECTED TARGET: [{t1_x:5.2f}, {t1_y:5.2f}, {t1_z:5.2f}]\n"
                    #f"L1 CALC. CENTROID:   [{centroid_1[0]:5.2f}, {centroid_1[1]:5.2f}, {centroid_1[2]:5.2f}]\n"
                    #f"L2 SLIDER TARGET:    [{tx:5.2f}, {ty:5.2f}, {tz:5.2f}]\n"
                    #f"L2 CALC. CENTROID:   [{calc_centroid_2[0]:5.2f}, {calc_centroid_2[1]:5.2f}, {calc_centroid_2[2]:5.2f}]\n"
                    f"L2 POS. ERROR:    {pos_accuracy:5.4f} units\n" 
                    f"EXECUTION TIMES (Total for both layers)\n"
                    f"NN: {total_nn:6.2f} ms | GN: {total_gn:6.2f} ms | SLSQP Opt: {total_opt:6.2f} ms")
        
        self.ax.text2D(0.05, 0.95, info_str, transform=self.ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.ax.set_xlim([-4, 4])
        self.ax.set_ylim([-4, 4])
        self.ax.set_zlim([0, 15])
        
        self.ax.set_box_aspect((6, 6, 15))
        self.ax.legend()
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    predictor_app = ActuatorPredictorApp("model.pth", "stats.json", batch_size=10, num_freqs=10)
    visualizer = ActuatorVisualizer(predictor_app)