import argparse
import os
import random

import numpy as np
import torch
import trimesh as tm
from plotly import graph_objects as go

import utils.visualize_plotly

vv

# set random seeds
np.seterr(all='raise')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from utils.HandModel import HandModel
from utils.Losses import FCLoss
from utils.ObjectModel import DeepSDFModel, SphereModel
from utils.PenetrationModel import PenetrationModel
from utils.PhysicsGuide import PhysicsGuide

# prepare models
if args.obj_model == 'bottle':
    object_model = DeepSDFModel(
        predict_normal=False, 
        state_dict_path="data/DeepSDF/2000.pth",
        code_path = 'data/DeepSDF/Reconstructions/2000/Codes/ShapeNetCore.v2/02876657',
        mesh_path = 'data/DeepSDF/Reconstructions/2000/Meshes/ShapeNetCore.v2/02876657')
    object_code, object_idx = object_model.get_obj_code_random(args.batch_size)
#! easier mode for the shpere to check the feasiblity
#! because the deepsdf model will need to more code.
elif args.obj_model == 'sphere':
    object_model = SphereModel()
    object_code = torch.rand(args.batch_size, 1, device='cuda', dtype=torch.float) * 0.2 + 0.1
else:
    raise NotImplementedError()

#! Create HandModel using ManoLayer
hand_model = HandModel(
    n_handcode=45,
    root_rot_mode='ortho6d', 
    robust_rot=False,
    flat_hand_mean=False,
    mano_path=args.mano_path, 
    n_contact=args.n_contact)

#! Create FC Loss Model: FC has three terms 
fc_loss_model = FCLoss(object_model=object_model)

penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)
#! Physics Modelling
#! Create the Model of Physics Guide for the Modeling.
physics_guide = PhysicsGuide(hand_model, object_model, penetration_model, fc_loss_model, args)

accept_history = []

if args.hand_model == 'mano_fingertip':
    num_points = hand_model.num_fingertips
elif args.hand_model == 'mano':
    num_points = hand_model.num_points
else:
    raise NotImplementedError()

z = torch.normal(0, 1, [args.batch_size, hand_model.code_length], device='cuda', dtype=torch.float32, requires_grad=True)
contact_point_indices = torch.randint(0, hand_model.num_points, [args.batch_size, args.n_contact], device='cuda', dtype=torch.long)

#! optimize hand pose and contact map using physics guidance
energy, grad, verbose_energy = physics_guide.initialize(object_code, z, contact_point_indices)
linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
for physics_step in range(args.max_physics):
    energy, grad, z, contact_point_indices, verbose_energy = physics_guide.optimize(energy, grad, object_code, z, contact_point_indices, verbose_energy)
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
    accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
    _accept = accept.sum().detach().cpu().numpy()
    accept_history.append(_accept)
    if physics_step % 100 == 0:
        print('optimize', physics_step, _accept)

for refinement_step in range(args.max_refine):
    energy, grad, z, contact_point_indices, verbose_energy = physics_guide.refine(energy, grad, object_code, z, contact_point_indices, verbose_energy)
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
    accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
    _accept = accept.sum().detach().cpu().numpy()
    accept_history.append(_accept)
    if refinement_step % 100 == 0:
        print('refine', refinement_step, _accept)


os.makedirs('%s/%s-%s-%d-%d'%(args.output_dir, args.hand_model, args.obj_model, args.n_contact, args.batch_size), exist_ok=True)

for a in torch.where(accept)[0]:
    a = a.detach().cpu().numpy()
    hand_verts = physics_guide.hand_model.get_vertices(z)[a].detach().cpu().numpy()
    hand_faces = physics_guide.hand_model.faces
    if args.obj_model == "sphere":
        sphere = tm.primitives.Sphere(radius=object_code[a].detach().cpu().numpy())
        fig = go.Figure([utils.visualize_plotly.plot_hand(hand_verts, hand_faces), utils.visualize_plotly.plot_obj(sphere)])
    else:
        mesh = object_model.get_obj_mesh(object_idx[[a]].detach().cpu().numpy())
        fig = go.Figure([utils.visualize_plotly.plot_hand(hand_verts, hand_faces), utils.visualize_plotly.plot_obj(mesh)])
    fig.write_html('%s/%s-%s-%d-%d/fig-%d.html'%(args.output_dir, args.hand_model, args.obj_model, args.n_contact, args.batch_size, a))

