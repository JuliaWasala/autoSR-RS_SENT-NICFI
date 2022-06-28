# CONDA ENV SETUP #

The versions of automl packages have been carefully chosen in order to ensure GPU-accelerated computing.   
Because of these strict version requirements, the environments for this research have been split in two:  
    One for image processing to create SENT-NICFI: satellite-img-env (satellite_environment.yml)  
    One for deep learning: automl-sr-rs-env (automl_environment.yml).  

# General notes: 
It is recommended to install the conda environments with a prefix: they are both large, so the user 
has to make sure that there is enough storage space on the install location  
For both environments, YAML files are provided. The package lists provided here are meant as fallbacks   
if there are issues with installing the YAML files.

# automl-sr-rs-env #
Setup for GPU usage, which is strongly recommended.  

create conda env with python=3.7   
activate env  

# NOTE: these packages are the most dependent 
conda installs:
conda install -c nvidia cudnn=8.2.1  
jupyter

pip installs:
tensorflow==2.5.2  
keras-tuner==1.0.4  
git clone git@github.com:JuliaWasala/autokeras.git  
cd autokeras  
git checkout -t origin/1.0.16.post1 
install -e .  
tensorflow-addons==0.15.0  
tensorflow-datasets==4.5.0  
Pillow  
seaborn  
pandas  
tqdm  
tifffile  
wheel  

# satellite-img-env #
Not GPU-dependent

scikit-image
shapely
sentinelloader
sentinelsat
pandas
argparse
osgeo
tqdm