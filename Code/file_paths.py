import os

# project working directory
root_dir = os.getcwd()

### setting code directory
code_dir = os.path.join(root_dir, 'Code')

#### setting output directory
output_root = os.path.join(root_dir, 'Outputs')

# caltech directory (paste in place of code_dir the parent directory path where you will store caltech101 dataset)
dataset_path = os.path.join(code_dir)

# data matrices root paths
data_root = os.path.join(output_root, 'data')
data_matrix_root_path = os.path.join(data_root, 'data_matrices')
lblb_sim_root_path = os.path.join(data_root, 'lblb_sim_matrices')
imgimg_sim_root_path = os.path.join(data_root, 'imgimg_sim_matrices')

# imageid-weight pairs root path
id_weight_root_path = os.path.join(output_root, 'id_weight_pairs')

# latent semantics root path
ls_root_path =  os.path.join(output_root, 'latent_semantics')