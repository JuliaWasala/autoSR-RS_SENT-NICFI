import os
from distutils.dir_util import copy_tree

from downsample import downsample

if __name__ == "__main__":

    # DOWNLOAD ORIGINAL DATA FROM http://patreo.dcc.ufmg.br/2017/11/12/brazilian-cerrado-savanna-scenes-dataset/
    # AND PLACE IN DATADIR

    DATADIR = "/data/s1620444/automl/datasets/"
    CERRADODIR = DATADIR+"cerrado/"

    # make new folder for synthetic dataset and move images to hr folder
    # if not yet exists
    if not os.path.isdir(CERRADODIR+"hr"):
        print(f"creating {CERRADODIR}hr")
        os.makedirs(CERRADODIR, exist_ok=True)
        copy_tree(
            DATADIR+"Brazilian_Cerrado_Savana_Scenes_Dataset/images", CERRADODIR)
        os.rename(CERRADODIR+"images", CERRADODIR+"hr")

    # bicubic downsampling x2
    print("Creating lr_bicubic_2x")
    downsample(CERRADODIR, "lr_bicubic_2x")

    print("Done")
