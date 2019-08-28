# hBN-Hunter


This code allows for the fast detection of hexagonal Boron Nitride flakes in microscope images, suitable for the making of encapsulated graphene Van der Waals heterostructures. 

This a colour based method. On the highest, we have created a lookup table of the colour of hBN flakes based on their thickness (as assessed using AFM) and the code tries to detect flakes with the right colour. 

Note that the colours may vary with the substrate and camera sensor used. We use a Nikon DS-Fi2 microscope and exfoliate on 90nm-SiO2 which maximizes contrast. If you use a different setup it is possible to input your own color lookup tables, or compute one using the works of Blake et al. https://aip.scitation.org/doi/10.1063/1.2768624 and Jessen et al. https://www.nature.com/articles/s41598-018-23922-1 for instance and adapting them by simply replacing the graphene refractive index by hBN one, and inputing the correct RGB sensity curves for your microscope camera. This might be something I will add in the future.

To facilitate the detection, a bunch or corrections are made: illumination, noise, colour, contrast... Individual functions (might not be up to date) can be found in the eponyme folder



Two main scripts are provided. 

# CourseScan.py
The CourseScan is designed to work with low magnification images (typically 10x) as a first pass to detect potentail candidates. 

At the moment it returns a bit too many false detections. The issue is that because our look up table was made using a 100x magnification, the numerical aperture differs significantly for low magnification which in turn modifies heavily the colors. This can be improved by correcting for the NA which might be added in the future


# FineScan.py
This script is made to refine the previous detection at high magnification (typically 100x). It is fairly robust at this point and also tries to fit a given size device on the image (a simple rectangle).  Note that the latter fitting is fairly slow.
