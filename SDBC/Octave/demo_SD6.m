close all;
clear;
clc;

%% Loading flow data
[flows, flow_times, groundTruthIDX] = flowread('data/SD6.csv');

%% Parameter setting
spatial_R = 300;   % spatial distance threshold
angle_T = 30;      % direction consistency threshold (in degree)
mc_RepeatTimes = 999;   % Monte Carlo simulation tests

%% Clustering flows using flowSDBC
[IDX, seeds, flow_density, spatialneighbors] = flowSDBC(flows, flow_times, spatial_R, angle_T);

%% Significant test under CSR hypothesis
p_value = flowCSRtests(flows, IDX, mc_RepeatTimes, spatial_R);
IDX(p_value > 0.05) = 0;

%% Show clustering results
% figure;
% flowplot(flows, IDX);

%% ARI
ARI = adjusted_rand_score(IDX, groundTruthIDX);
fprintf('adjusted_rand_score = %f\n', ARI);

