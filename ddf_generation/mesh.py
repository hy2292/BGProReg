import numpy as np, sys, os, SimpleITK as sitk
import pygalmesh
import pyvista as pv
import meshio
from utils import read_txt_file

from pathlib import Path
sys.path.append(os.path.join(Path.home(), 'plot_lib'))
from lib.processing.preprocessing import timer

PATH_OUT= '/biopsy_prostate/processed/'
PATH_MASKS= '/biopsy_prostate/mesh/'

#Meshing config
PLOT= True
SPACING= [0.8]*3
MESH_MR= True  #Should we mesh the MR masks?
MESH_US= True  #Should we mesh the US masks?
QUICK_MODE= True #Much quicker meshing (but not as smooth)


meshing_configuration= dict(voxel_size= SPACING,
                            max_facet_distance=2.0,
                            max_cell_circumradius=1.5,
                            lloyd= not QUICK_MODE, #Gloabal outside lerp
                            odt= not QUICK_MODE, #Global inside lerp
                            perturb= True,
                            exude= True, #local
                            max_edge_size_at_feature_edges= 0.0,
                            min_facet_angle= 0.0,
                            max_radius_surface_delaunay_ball= 0.0,
                            max_circumradius_edge_ratio= 0.,
                            verbose= False)

#Statistics
mr_tris, mr_tetras, mr_surface_vertices= [], [], []
us_tris, us_tetras, us_surface_vertices= [], [], []
mr_points_num = []
us_points_num = []

#PIDs
pids= sorted([img.split('_')[0] for img in os.listdir(PATH_OUT) if img.endswith('MR_img.nrrd')],
             key= lambda a: int(a.split('ID')[1]))
print('PIDs:', pids)

filelist = read_txt_file(os.path.join('/home/dasong/BGProReg/data/bad.txt'))
print(filelist)

for pid in pids:
    print(pid)
    # Load all
    mr_img, mr_msk = sitk.ReadImage(os.path.join(PATH_OUT, pid + '_MR_img.nrrd')), sitk.ReadImage(os.path.join(PATH_OUT, pid + '_MR_msk.nrrd'))
    us_img, us_msk = sitk.ReadImage(os.path.join(PATH_OUT, pid + '_US_img.nrrd')), sitk.ReadImage(os.path.join(PATH_OUT, pid + '_US_msk.nrrd'))

    # Get to numpy
    mr_msk_arr = sitk.GetArrayFromImage(mr_msk).astype(np.uint8)
    us_msk_arr = sitk.GetArrayFromImage(us_msk).astype(np.uint8)

    # Create meshes and save
    mr_mesh_name = os.path.join(PATH_MASKS, f"{pid}_MR_msh{'_quick' if QUICK_MODE else ''}.vtk")
    us_mesh_name = os.path.join(PATH_MASKS, f"{pid}_US_msh{'_quick' if QUICK_MODE else ''}.vtk")

    if MESH_MR and not os.path.exists(mr_mesh_name):
        print('Generating:', mr_mesh_name)
        mr_mesh = timer(pygalmesh.generate_from_array)(mr_msk_arr.swapaxes(0, 2), **meshing_configuration)
        print(' - MR Mesh: There are %d vertices, %d tris, and %d tetra' % (mr_mesh.points.shape[0],
                                                                            mr_mesh.cells_dict['triangle'].shape[0],
                                                                            mr_mesh.cells_dict['tetra'].shape[0]))
        mr_mesh.write(mr_mesh_name)
        mr_mesh.write(mr_mesh_name.replace('.vtk', '.stl'))
    else:
        mr_mesh = meshio.read(mr_mesh_name)

    if MESH_US and not os.path.exists(us_mesh_name):
        us_mesh = timer(pygalmesh.generate_from_array)(us_msk_arr.swapaxes(0, 2), **meshing_configuration)
        print(' - US Mesh: There are %d vertices, %d tris, and %d tetra' % (us_mesh.points.shape[0],
                                                                            us_mesh.cells_dict['triangle'].shape[0],
                                                                            us_mesh.cells_dict['tetra'].shape[0]))
        us_mesh.write(us_mesh_name)
        us_mesh.write(us_mesh_name.replace('.vtk', '.stl'))
    else:
        us_mesh = meshio.read(us_mesh_name)

    # Save mesh statistics
    mr_tris.append(mr_mesh.cells_dict['triangle'].shape[0])
    mr_tetras.append(mr_mesh.cells_dict['tetra'].shape[0])
    mr_surface_vertices.append(len(np.unique(mr_mesh.cells_dict['triangle'].flatten())))

    us_tris.append(us_mesh.cells_dict['triangle'].shape[0])
    us_tetras.append(us_mesh.cells_dict['tetra'].shape[0])
    us_surface_vertices.append(len(np.unique(us_mesh.cells_dict['triangle'].flatten())))

    # Re-read them in pyvista format:
    mr_mesh_pv = pv.read(mr_mesh_name)
    us_mesh_pv = pv.read(us_mesh_name)

    print(pid)
    mr_points_num.append(mr_mesh.points.shape[0])
    us_points_num.append(us_mesh.points.shape[0])


print("MR:",mr_points_num)
print("US:",us_points_num)



