function [flows, flow_times, IX] = flowread(csvfile)

%  Readme:
%    Code of the paper "Statistical and density-based clustering of 
%    geographical flows for crowd movement patterns recognition, Applied
%    Soft Computing, 2024, 163: 111912"
%    DOI: https://doi.org/10.1016/j.asoc.2024.111912
%  If you use this code, please cite the above paper, Thanks.
%  Tang Jianbo, CSU
%

flows = importdata(csvfile);
flows = flows.data;
[~, m] = size(flows);
if m==6
   % fid, ox, oy, dx, dy, label 
   flow_times = [];
   IX = flows(:, 6);
   flows = flows(:, 2:5);
elseif m==8
   % fid, ox, oy, ot, dx, dy, dt, label
   flow_times = flows(:, [4, 7]);
   IX = flows(:, 8);
   flows = flows(:, [2,3,5,6]);
else
    error('flow data format is not vaild.');
end
vmin = min([min(flows(:,1:2)); min(flows(:,3:4))]);
flows(:,1) = flows(:,1) - vmin(1);
flows(:,2) = flows(:,2) - vmin(2);
flows(:,3) = flows(:,3) - vmin(1);
flows(:,4) = flows(:,4) - vmin(2);
end % flowread()

