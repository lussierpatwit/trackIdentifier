# trackIdentifier
Program utilizing opencv libraries to identify F1 tracks

This program was made as a way to teach myself the ins and outs of opencv. I attempted to merge the learning process with someting else I enjoy, Formula 1.
In essence this program takes in a file containing all 23 of the tracks on the 2021 calender, and catalouges them as black and white outlines. It then takes a photo of a track
from another file and tries to determine which of the tracks from the 2021 calender it thinks it is. As of 4.14.21 it uses the built in opencv method matchShapes() to comapre the 
input image and each of the preloaded tracks. In future versions I may try and make my own compare method to beter understand this portion of opencv. Another peice I hope to add
down the road is a more robust GUI as well as possibly training a model to better classify hand drawn tracks. 
