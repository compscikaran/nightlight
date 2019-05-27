# Nightlight CLI

Steps to run

1. Clone this repo 
2. Install Anaconda and Install additional dependencies by running 
>conda install tensorflow

>python -m pip install rawpy imagio exifread3. 
3. Download model files and place them in models/ folder and place test images in images/ folder
4. Run production script
>python3 production.py [image path] [output_file_name]
5. Resulting output image will appear in files folder as output_file_name.png in images directory
