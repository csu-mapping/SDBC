function flowplot(flows, IDX, linewidth)

%  Readme:
%    Code of the paper "Statistical and density-based clustering of 
%    geographical flows for crowd movement patterns recognition, Applied
%    Soft Computing, 2024, 163: 111912"
%    DOI: https://doi.org/10.1016/j.asoc.2024.111912
%  If you use this code, please cite the above paper, Thanks.
%  Tang Jianbo, CSU
%

if nargin<2
    IDX = ones(size(flows,1), 1);
end
if nargin<3
    linewidth = 2;
end

n = size(flows, 1);
if length(IDX) < n
    IX = zeros(n,1);
    IX(IDX) = 1;
    IDX = IX;
    clear IX;
end

hold on;
% set(gcf, 'position', [544.1111  163.2222  772.0000  691.5556]);
IX = unique(IDX(IDX>0));
n = length(IX);
cluster_colors = 'b';
if n==1
    cluster_colors = [1 0 0]; %colors = {'k','r','b','g','y','c','m','p'};
elseif n>1
    cluster_colors = rand(n, 3); %colors = {'k','r','b','g','y','c','m','p'};
end
noise_colors = [0.7 0.7 0.7];

[X, Y] = flow_to_sequence(flows, IDX<1);
if ~isempty(X)
    plot(X, Y, '-', 'color', noise_colors, 'linewidth', linewidth);
end

for i=1:length(IX)
    [X, Y] = flow_to_sequence(flows, IDX==IX(i));
    if ~isempty(X)
        plot(X, Y, '-', 'color', cluster_colors(i,:), 'linewidth', linewidth);
    end
end
axis('equal');
xlabel('X');
ylabel('Y');
set(gca, 'fontsize', 14, 'fontweight', 'bold');
drawnow;
end % flowplot


function [X, Y] = flow_to_sequence(flows, label)
if any(label)
    n = length(label);
    if n==size(flows, 1)
        n = sum(label);
    end
    X = NaN*ones(3*n, 1);
    Y = NaN*ones(3*n, 1);
    X(1:3:end) = flows(label,1);
    X(2:3:end) = flows(label,3);
    Y(1:3:end) = flows(label,2);
    Y(2:3:end) = flows(label,4);
else
    X = [];
    Y = [];
end
end % func


