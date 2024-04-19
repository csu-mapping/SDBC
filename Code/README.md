# flow-clustering
Statistical and density-based clustering of geographical flows for crowd movement patterns recognition

## Enviroment
Python version: 3.7 

## Simulated data
The simulated dataset were stored in the folder 'Simulated data'
In the folder 'Simulated data', 'SD1', 'SD2', 'SD3'，'SD4'，'SD5'，'SD6' and 'SD7' correspond to the seven sets of simulation data in the paper.
In each simulated data folder, the flow ID , the coordinates of the start and end points of the flow, and the original cluster to each flow belongs are described.

## Method
The codes of SDBC and the comparison methods were stored in the folder 'Code'
The codes of SDBC was stored in the folder 'SDBC'
The codes of the comparison methods were stored in the folder 'Comparison methods'

In the folder 'Comparison methods', we've only given the code for two of the comparison methods that we've modified. They are: a stepwise spatio-temporal flow clustering method and spatial flow L-function and SpatialflowL.

The reference source link for comparison methods are: 
https://doi.org/10.6084/m9.figshare.14123960



## Implement of SDBC method
Flow_clustering.py was used to detect clustering results based on SDBC method

## Implement of the stepwise spatio-temporal flow clustering method 
flowSCluster.py was used to detect clustering results based on the stepwise spatio-temporal flow clustering method 

## Implement of SpatialflowL method
Step 1: make_buffer.py was used to generate a buffer for the flows within the input study area in order to avoid edge effects.
The file 'flows_in' stores the flows within the input study area.
The file 'flows_all' stores the flows within the input study area.
The file 'flows_all' stores all flows that includes the flows within the generated buffer and the flows within the study area.

Step 2:L.py was used to estimated the spatial aggregation scale r, which is the only parameter of the SpatialflowL method based on the global L-function.

Step 3:Extract_cluster.py was used to detect clustering results based on the SpatialflowL method.

