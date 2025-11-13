# BGProReg: A Biomechanically Generative Framework for Prostate MRI–TRUS Deformable Image Registration

Demo code accompanying the paper of the same title.

## Usage

**Step 1: Obtain the biomechanical prior**  
Run the `mesh.py` and `CPD_FEM.py` in `ddf_generation` to generate biomechanical deformation fields.  

**Step 2: Train the VAE-based registration network**  
Run `python main_train.py`  

**Step 3: Perform the registration**  
Use the trained model to perform MRI–TRUS registration: `python main_predict.py`
