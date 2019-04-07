# Low Light Photo Enhancement

Steps to run

1. Clone this repo 
2. Install dependencies by running 

>python3 -m pip install numpy tensorflow rawpy imageio
3. Download model and test images zip file and unzip them into a new folder called files ( image should be more than 1400x1400 pixels)
4. Run production script
>python3 production.py [image path]
5. Resulting output image will appear in files folder as result.jpg