## SDBC: Geographical flow clustering

This project incudes the source codes and data used in the paper:

Jianbo Tang, Yuxin Zhao, Xuexi Yang, et al. Statistical and density-based clustering of geographical flows for crowd movement patterns recognition, Applied Soft Computing, 2024, 163: 111912. https://doi.org/10.1016/j.asoc.2024.111912



#### Install

You can download this code and unzip the source code and data files in your computer. For example, unzip you code into the folder "D:/flowSDBC".

The source code are implemented with Octave and Python (version >=3.8).



#### Simulated data

The simulated dataset were stored in the folder <u>'Data'</u>

In the folder 'Simulated data', 'SD1', 'SD2', 'SD3'，'SD4'，'SD5'，'SD6' and 'SD7' correspond to the seven sets of simulation data in the paper.
In each simulated data folder, the flow ID , the coordinates of the start and end points of the flow, and the original cluster to each flow belongs are described.



#### Method

The codes of the comparison methods were stored in the folder 'Comparison methods'

In the folder 'Comparison methods', we've given the code for two of the comparison methods that we've modified. They are a stepwise spatiotemporal flow clustering method and spatial flow L-function and SpatialflowL. The code of SNN_flow method can be found at link: https://doi.org/10.6084/m9.figshare.14123960. The code of a stepwise spatial-temporal flow clustering method can be found at https://github.com/susurrant/flow-clustering.



#### Usage

**1）Implementation with Octave/MATLAB** 

In the 'Octave' folder：
Running the 'flowread.m' to load the simulated flows in SDx in the 'data' folder. 
Running the 'demo_SD1.m' script to call the flowSDBC to find clusters in SD1. 
You can use 'flowplot.m' to show the flow clustering result.
Running the 'flowRDVCompute.py' to compute the RDV wrt. R values and theta values, this will generate a result file named 'RDV_data.csv' in the current folder.
Running the 'flowRDVFigure.py' will load the 'RDV_data.csv' to show the gradients of RDV wrt R and RDV wrt theta. 

Please note that the parameter setting process can be replaced by any other method you need.

**2）Implementation with Python 3**

In the 'Python' folder:

You can also find the source codes of flowSDBC clustering algorithm implemented using Python 3. The demo scripts, e.g., demo_SD1.py, demo_SD2.py,..., demo_SD7.py, show the usage of the flowSDBC clustering algorithm.



#### Demos

Clustering of SD1:

<img src="SDBC/Demo/demo_SD1.gif" alt="demo_SD1"/>

Clustering of SD2:

<img src="SDBC/Demo/demo_SD2.gif" alt="demo_SD2"/>

Clustering of SD3:

<img src="SDBC/Demo/demo_SD3.gif" alt="demo_SD3"/>

Clustering of SD4:

<img src="SDBC/Demo/demo_SD4.gif" alt="demo_SD4"/>

Clustering of SD5:

<img src="SDBC/Demo/demo_SD5.gif" alt="demo_SD5"/>

Clustering of SD6:

<img src="SDBC/Demo/demo_SD6.gif" alt="demo_SD6"/>

Clustering of SD7:

<img src="SDBC/Demo/demo_SD7.gif" alt="demo_SD7"/>
