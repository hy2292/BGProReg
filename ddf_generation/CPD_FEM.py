import numpy as np, pandas as pd, sys, os, SimpleITK as sitk
from tqdm import tqdm
from functools import partial
import meshio
import copy
import open3d as o3
from probreg import cpd

#Custom functions
from lib.processing.preprocessing import (MI_sitk, DSC_sitk, print_metric,timer,  get_gradient_features, transform_sitk)
from lib.processing.meshing import get_surface_mesh, quality, get_DDF, get_FEM_displacements, PlotCallback

#Meshing config
PLOT= False
SPACING= [0.8]*3
MESH_MR= True
MESH_US= True
QUICK_MODE= True #Much quicker meshing (but not as smooth)

#Registration config
USE_CUDA= True #Use CUDA for CPD registration (much faster)
SAVE= True #Should we save the outputs?
REGISTER_RIGID= False #Perform rigid CPD registration before non-rigid (better True)
USE_FEM= True #Interpolate internal displacement field using FEM (requires FeBIO)
fem_material= dict(material='neo-Hookean', density=str(1.), E=str(5000.), v=str(0.49)) #FEM material properties

#Use CUDA for CPD speedup?
if USE_CUDA:
    import cupy as cp
    to_cpu, to_array, float32 = cp.asnumpy, cp.asarray, cp.float32
    to_numpy= lambda a: to_array(to_cpu(a)).get()
else:
    to_cpu, to_array, float32= lambda x: x, np.asarray, np.float32
    to_numpy= lambda a: a

#CPD + interpolation configuration. Defined as a list for doing a hyperparameter sweep
betas, lambdas, all_kernels, epsilons= [100.], [3.3], ['thin_plate_spline'], [100.]

#Create dataframe with results. Useful if we do a hyperparameter sweep
columns= ['beta', 'lambda', 'kernel', 'epsilon', 'mi_all', 'mi_in', 'mig_all', 'mig_in','dsc', 'hd95', 'abd', 'tre']

PATH_OUT= '/biopsy_prostate/processed/'  #image
PATH_MASKS= '/biopsy_prostate/mesh'
PATH_DDF= '/biopsy_prostate/DDFs'
PATH_POINT_MR = '/biopsy_prostate/points/MR'
PATH_POINT_US = '/biopsy_prostate/points/US'
PATH_POINT_MOVE = '/biopsy_prostate/displacement'

#PIDs
pids= sorted([img.split('_')[0] for img in os.listdir(PATH_OUT) if img.endswith('MR_img.nrrd')],
             key= lambda a: int(a.split('ID')[1]))
print('PIDs:', pids)

for pid in pids:
    print(pid)
    DDF_path = os.path.join(PATH_DDF,
                            f"{pid}_DDF{'_FEM' if USE_FEM else ''}{'_linear' if 'linear' in all_kernels else ''}"
                            f"{'_linear' if 'linear' in all_kernels else ''}.nrrd")
    mr_points_path = os.path.join(PATH_POINT_MR, f"{pid}_MR_points.npy")
    us_points_path = os.path.join(PATH_POINT_US, f"{pid}_US_points.npy")
    displacement_path = os.path.join(PATH_POINT_MOVE, f"{pid}_MR_displacement.npy")

    # Load all
    mr_img, mr_msk = sitk.ReadImage(os.path.join(PATH_OUT, pid + '_MR_img.nrrd')), sitk.ReadImage(os.path.join(PATH_OUT, pid + '_MR_msk.nrrd'))
    us_img, us_msk = sitk.ReadImage(os.path.join(PATH_OUT, pid + '_US_img.nrrd')), sitk.ReadImage(os.path.join(PATH_OUT, pid + '_US_msk.nrrd'))
    mr_mesh = meshio.read(os.path.join(PATH_MASKS, f"{pid}_MR_msh{'_quick' if QUICK_MODE else ''}.vtk"))
    us_mesh = meshio.read(os.path.join(PATH_MASKS, f"{pid}_US_msh{'_quick' if QUICK_MODE else ''}.vtk"))
    us_msk.SetSpacing(mr_msk.GetSpacing())

    np.save(mr_points_path, mr_mesh.points)
    np.save(us_points_path, us_mesh.points)

    # Get surface meshes from the volumetric meshes
    surf_points_i_mr, surf_points_mr, surf_mesh_pv_mr = get_surface_mesh(mr_mesh)
    surf_points_i_us, surf_points_us, surf_mesh_pv_us = get_surface_mesh(us_mesh)

    # Get a point cloud from the surface mesh
    source_mesh = o3.geometry.PointCloud(o3.utility.Vector3dVector(surf_points_mr))
    target_mesh = o3.geometry.PointCloud(o3.utility.Vector3dVector(surf_points_us))

    # Get points into GPU / CPU
    source_pt = to_array(source_mesh.points, dtype=float32)
    target_pt = to_array(target_mesh.points, dtype=float32)


    class IdentityTransform():
        transform = lambda self, a: a  # Just for plotting before CPD

    min_p, max_p = np.min(source_mesh.points, axis=0) - 7, np.max(source_mesh.points, axis=0) + 7
    plot_params = dict(to_numpy=to_numpy, x_limits=[min_p[0], max_p[0]], y_limits=[min_p[1], max_p[1]],
                       size=(8, 6), hide_axis=True, mode='2d', axis=[2, 1, 0],
                       #save_as =f"{pid}",
                       moving_last_N=0, fixed_last_N=0)
    reg_kwargs = dict(tf_type_name='nonrigid')

    # Make loops to sweep hyperparameters if needed
    for beta in betas:
        for lmd in lambdas:
            if REGISTER_RIGID:
                with tqdm(total=100, position=0, leave=True) as pbar:
                    #see /lib/meshing.py  plot_show
                    plot_callback = PlotCallback(source_pt, target_pt, plot_interval=20, **plot_params,plot_show=False)
                    plot_callback(IdentityTransform())
                    tf_param, a, b = timer(cpd.registration_cpd)(source_pt, target_pt, tf_type_name='rigid',
                                                                 callbacks=[lambda _: pbar.update(1), plot_callback],
                                                                 use_cuda=USE_CUDA, tol=5e-4, maxiter=100)
                source_pt_init = tf_param.transform(source_pt)
                with tqdm(total=100, position=0, leave=True) as pbar:
                    plot_callback = PlotCallback(source_pt_init, target_pt, plot_interval=5, **plot_params,plot_show=False)
                    tf_param2, a, b = timer(cpd.registration_cpd)(source_pt_init, target_pt,
                                                                  callbacks=[lambda _: pbar.update(1), plot_callback],
                                                                  use_cuda=USE_CUDA, beta=beta, lmd=lmd,
                                                                  tol=5e-5, maxiter=150, **reg_kwargs)
                source_pt_after = tf_param2.transform(source_pt_init)

            else:
                source_pt_init = source_pt
                with tqdm(total=100, position=0, leave=True) as pbar:
                    plot_callback = PlotCallback(source_pt_init, target_pt, plot_interval=5, **plot_params,plot_show=True)
                    plot_callback(IdentityTransform())
                    tf_param, a, b = timer(cpd.registration_cpd)(source_pt_init, target_pt,
                                                                 callbacks=[lambda _: pbar.update(1), plot_callback],
                                                                 use_cuda=USE_CUDA, beta=beta, lmd=lmd,
                                                                 tol=5e-4, maxiter=150, **reg_kwargs)
                    source_pt_after = tf_param.transform(source_pt_init)

            # Apply registration
            surf_mesh_pv_mr_after = copy.deepcopy(surf_mesh_pv_mr)
            pt_mr_after_cpu = to_cpu(source_pt_after)
            surf_mesh_pv_mr_after.points = pt_mr_after_cpu
            quality(surf_mesh_pv_mr_after, plot=False)

            # Get surface points and CPD-found deformations
            surf_pts = to_numpy(surf_mesh_pv_mr.points)
            surf_def = to_numpy(surf_mesh_pv_mr_after.points - surf_mesh_pv_mr.points)


            # Make loops to sweep hyperparameters if needed
            for kernel in all_kernels:
                for epsilon in epsilons:
                    # Predict the DDF by interpolating from the known surface points
                    # This is a slow process, taking around 40s-80s for an 8-core processor
                    USE_EPSILON = kernel not in ['thin_plate_spline', 'linear', 'cubic']
                    if USE_FEM:
                        fem_displacements = get_FEM_displacements(mr_mesh, surf_points_i_mr, surf_def, time_steps=5, **fem_material)
                        np.save(displacement_path, fem_displacements)

                        DDF_tfm = timer(get_DDF)(mr_mesh.points, fem_displacements, mr_img.GetSize(), mr_img.GetSpacing(),
                                                 kernel='linear',
                                                 neighbors=10,
                                                 epsilon=epsilon if USE_EPSILON else 1.,
                                                 smoothing=epsilon if not USE_EPSILON else 0.)
                    else:
                        DDF_tfm = timer(get_DDF)(surf_pts, surf_def, mr_img.GetSize(), mr_img.GetSpacing(),
                                                 kernel=kernel,
                                                 epsilon=epsilon if USE_EPSILON else 1.,
                                                 smoothing=epsilon if not USE_EPSILON else 0.)

                    # Obtain reverse transformation
                    DDF_tfm_img = sitk.DisplacementFieldTransform(sitk.InvertDisplacementField(DDF_tfm.GetDisplacementField(), enforceBoundaryCondition=False, ))

                    # Apply transformation to the points and to the original image
                    mr_img_after = transform_sitk(mr_img, DDF_tfm_img, sitk.sitkBSpline,extrapolate=True)
                    mr_msk_after = transform_sitk(mr_msk, DDF_tfm_img, sitk.sitkLabelGaussian)

                    dsc = print_metric(us_msk, mr_msk, mr_msk_after, metric=DSC_sitk, name='DSC (masks)')[1]

                    # Save
                    if SAVE: sitk.WriteImage(DDF_tfm.GetDisplacementField(), DDF_path, True, -1)
                    print("PIDs:",pid," Save!")