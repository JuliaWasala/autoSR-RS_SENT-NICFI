This folder contains the code to construct the SENT-NICFI dataset.

# needed
* access to Planet NICFI dataset 
* export Planet API key as PLANET_API_KEY
* use: satellite-img-env

# contents
* download_quads.py: script to download the NICFI tiles
* get_matching_tilenames.py: script to retrieve matching Sentinel-2 images. 
    Run this script with command line argument --tilename, with tilename corresponding to the NICFI tilename. Images might download at once, or if in LTA the script has to be run again 24 hours later.
    The script produces a folder for each NICFI tile which has a list of tiles that are offline (still in LTA) and tiles that are online (path to where the downloaded tile is). Multiple Sentinel images per NICFI image can be downloaded.
* tif_processing.ipynb: notebook where the retrieved Sentinel images can be inspected and selected if there are more than 1 per NICFI tile. Contains code to move the selected images to the final location
* resize_and_split.py: resizes the Sentinel and nicfi images to integer 
pixel resolutions (in m), such that the resolution of NICFI is exactly 2x as high as sentinel. Then, splits the images into tiles (100x100 for sentinel, 200x200 for NICFI). Finally, the NICFI images are color corrected using the corresponding Sentinel image.
* util.py: contains utility functions for get_matching_tilenames.py


# how to replicate the dataset
* download NICFI quads
* download the set of Sentinel images, made available at:
    note that a list of product ids could not be provided, since some of the images are merges of multiple images (this is done by sentinelloader package). It is possible to run get_matching_tilenames.py & tif_processing.ipynb as well to get the exact same images.
* run resize_and_split
* this yields the following folder structure:

sent_nicfi  -   sent 
                    - < folders with names corresponding to NICFI tile ids >
                    - original: folder where selected sentinel images are moved 
                    - 10m: images from original resampled to 10m resolution
    --->            - 10m_splits: images from 10m split into 100x100 tiles -> these are the final images
                        IF RUNNING get_matching_tilenames.py:
                    - apiquery: saved queries from sentinelloader
                    - tmp: temporary files from sentinelloader
                    - products: images downloaded by sentinelloader
            -   nicfi
                    - original: downloaded images from download_quads.py
                    - 5m: images from original resampled to 5m resolution
                    - 5m_splits: images from 5m split into 200x200 tiles
    --->            - 5m_color_corrected_splits: images from 5m_splits color corrected with histogram matching sentinel images. These are the final images.