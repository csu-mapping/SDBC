function [IDX, seeds, flow_density, spatialneighbors] = flowSDBC(flows, flow_times,...
    spatial_R, angle_T)

%  Readme:
%    Code of the paper "Statistical and density-based clustering of 
%    geographical flows for crowd movement patterns recognition, Applied
%    Soft Computing, 2024, 163: 111912"
%    DOI: https://doi.org/10.1016/j.asoc.2024.111912
%  If you use this code, please cite the above paper, Thanks.
%  Tang Jianbo, CSU
%

% 参数设置
MIN_CLUSTER_MEMBER_NUMS = 5;
dist_mode = 2;
% gif_file_name = 'demo_SD7.gif';

% Step.1
n = size(flows, 1);
nnmat = false(n, n);
if ~isempty(flow_times)
    for i=1:n
        for j=(i+1):n
            [d, r] = flow_distance(flows(i,:), flows(j,:), dist_mode);
            t = (flow_times(i,1)<=flow_times(j,2)) && (flow_times(i,2)>=flow_times(j,1));
            if d<=spatial_R && r<=angle_T && t
                nnmat(i,j) = true;
                nnmat(j,i) = true;
            end
        end
    end
else
    for i=1:n
        for j=(i+1):n
            [d, r] = flow_distance(flows(i,:), flows(j,:), dist_mode);
            if d<=spatial_R && r<=angle_T
                nnmat(i,j) = true;
                nnmat(j,i) = true;
            end
        end
    end
end
flow_density = zeros(n, 1);
spatialneighbors = cell(n,1);
for i=1:n
    flow_density(i) = sum(nnmat(i,:));
    spatialneighbors{i} = find(nnmat(i,:));
end


% Step.2
npts = size(flows, 1);
data = flow_density;  
data = (data - mean(data))./std(data, 1);  % 数据标准化

lamada = mean(data) - 2*std(data,1);

K = sum(nnmat, 2);
K = sqrt( K.*(npts-K) ./ (npts-1) );
G = nnmat*data + data;
G = G./K;
seeds = find(G >= lamada);

if length(seeds)<1
    IDX = [];
    disp('   Not any clusters detected.');
    return;
end

% 寻找合并后使得Gi*最大化的V图单元组合(I,J)进行合并，值得无法合并时停止
hotspots = struct(...
    'Members',num2cell((1:npts)'), ...
    'Gi', num2cell(data),...
    'IsSeed', num2cell(zeros(npts,1)),...
    'InnerDis', cell(npts,1));

for i=1:length(seeds)
    hotspots(seeds(i)).IsSeed = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
set(gcf, 'position', [544.1111  163.2222  772.0000  691.5556]);
IDX = zeros(npts, 1);
IDX([hotspots.IsSeed]==1) = 1;
flowplot(flows, IDX);
title('Detected flow seeds','fontsize',14,'fontweight','bold');
drawnow;
% gif = csu_gif(gcf, ...
%         'filename', gif_file_name, ...
%         'delaytime', 3.0, ...
%         'loopcount', 0);
%%%%%%%%%%%%%%%%%%%%%%%

count = 0;
nclust = length(hotspots);
while nclust > 1   
    [I, J, gstat] = searching(data, nnmat, hotspots, lamada);
    if ( I < 1 ) || ( J < 1 )
        break;
    end    
    [hotspots, nnmat] = updating(nnmat, hotspots, I, J, gstat);
    nclust = length(hotspots);
    
    count = count + 1;
    %%%%%%%%%%%%%%%%%%%%%%%
    if rem(count, 5)==0
        figure(2);
        set(gcf, 'position', [544.1111  163.2222  772.0000  691.5556]);
        cla;
        hold on;
        IX = get_idx(hotspots, npts, 2);
        flowplot(flows, IX, 2);
        % pause(1);
        axis equal;
        axis off;
        title(['Clustering (iteration=',num2str(count),')'],'fontsize',14,'fontweight','bold');
        drawnow;
        % gif.add(gcf, 0.001);
    end
    %%%%%%%%%%%%%%%%%%%%%%%
end

clear  nnmat data;

% member_nums = arrayfun(@(x) length(x.Members), hotspots);
% hotspots = hotspots( member_nums>=MIN_CLUSTER_MEMBER_NUMS );

%-------------------------------
for i=1:length(hotspots)
    [dmax, ~] = cluster_inner_distance(flows, hotspots(i).Members, dist_mode);
    hotspots(i).InnerDis = dmax;
end

n = length(hotspots) + 1;
while length(hotspots) < n
    n = length(hotspots);
    hotspots = cluster_merge(flows, hotspots, dist_mode);
    
    %%%%%%%%%%%%%%%
    count = count + 1;
    figure(2);
    set(gcf, 'position', [544.1111  163.2222  772.0000  691.5556]);
    cla;
    hold on;
    IX = get_idx(hotspots, npts, 2);
    flowplot(flows, IX, 2);
    % pause(1);
    axis equal;
    axis off;
    title(['Clustering (iteration=',num2str(count),')'],'fontsize',14,'fontweight','bold');
    drawnow;
    % gif.add(gcf, 0.001);
    %%%%%%%%%%%%%%%
end
mnums = arrayfun(@(x) length(x.Members), hotspots);
hotspots = hotspots( mnums>=MIN_CLUSTER_MEMBER_NUMS );


IDX = zeros(npts,1);
mnums = zeros(npts,1);
for i=1:length(hotspots)
    IDX(hotspots(i).Members) = i;
    mnums(i) = length(hotspots(i).Members);
end

[~, I] = sort(mnums, 'descend');
is_seed = false(npts,1);
is_seed(seeds) = true;
for i=1:length(I)
    cid = I(i);
    nn = [spatialneighbors{IDX==cid}];
    ind = is_seed(nn) & IDX(nn)<1;
    IDX(nn(ind)) = cid;
end

%%%%%%%%%%%%%%%
figure(2);
set(gcf, 'position', [544.1111  163.2222  772.0000  691.5556]);
cla;
hold on;
flowplot(flows, IDX, 2);
axis equal;
axis off;
title('Clustering result','fontsize',14,'fontweight','bold');
drawnow;
% gif.add(gcf, 1.0);
%%%%%%%%%%%%%%%

clear member_nums;
end % flowSDBC


% search_min_vars
function [I, J, max_gstat] = searching(data, nnmat, hotspots, lamada)
n = length(data);
seeds = find([hotspots.IsSeed]);
clusternums = length(seeds);
I = 0;
J = 0;
max_gstat = -999999.0;

for i = 1:clusternums
    id = seeds(i);
    A  = hotspots(id).Members;
    %gstatA = hotspots(id).Gi;
    neighbor = find( nnmat(id,:) );   
    if isempty(neighbor)
        continue;
    end
    
    for j = 1:length(neighbor)
        nid = neighbor(j);
        B  = [hotspots(nid).Members];
        %gstatB = hotspots(nid).Gi;
        current_member = [A, B];
        k = length(current_member);
        
        % measuring index
        current_stat = sum( data(current_member) )/sqrt(k*(n-k)/(n-1));
        if current_stat > lamada && (current_stat > max_gstat)
            max_gstat = current_stat;
            I = id;
            J = nid;
        end
    end
end
end   % search_min_vars



% update_cluster
function [hotspots, nnmat] = updating(nnmat, hotspots, I, J, gstat)
hotspots(I).Members = [hotspots(I).Members, hotspots(J).Members];
hotspots(I).Gi = gstat;
hotspots(I).IsSeed = true;
hotspots(J) = [];

if length(J) == 1
    nnmat(I,:) = nnmat(I,:)|nnmat(J,:);
else
    nnmat(I,:) = any( nnmat([I,J],:), 1);
end

nnmat(:,I) = nnmat(I,:)';
nnmat(I,I) = false;

nnmat(J,:) = [];
nnmat(:,J) = [];

end % update_cluster

function [d, r] = flow_distance(flow_i, flow_j, dist_mode)
if nargin < 3
    dist_mode = 2;
end
df = flow_j(1:4) - flow_i(1:4);
df = df.^2;
switch dist_mode
    case 1
        d = sqrt(max([df(1)+df(2), df(3)+df(4)]));
    case 2
        d = (sqrt(df(1)+df(2))+sqrt(df(3)+df(4)))/2;
end
v1 = flow_i(3:4)-flow_i(1:2);
v2 = flow_j(3:4)-flow_j(1:2);
r = 180/pi*acos((v1(1)*v2(1)+v1(2)*v2(2))/(norm(v1)*norm(v2)+eps));
end % 


function d = fdij(flow_i, flow_j, dist_mode)
if nargin < 3
    dist_mode = 2;
end
switch dist_mode
    case 1
        df = flow_j(1:4) - flow_i(1:4);
        df = df.^2;
        d = sqrt(max([df(1)+df(2), df(3)+df(4)]));
        return
    case 2
        df = flow_j(1:4) - flow_i(1:4);
        df = df.^2;
        d = (sqrt(df(1)+df(2))+sqrt(df(3)+df(4)))/2;
        return
    otherwise
        df = flow_j(1:4) - flow_i(1:4);
        df = df.^2;
        do = df(1)+df(2);
        dd = df(3)+df(4);
        L1 = norm(flow_i(3:4)-flow_i(1:2));
        L2 = norm(flow_j(3:4)-flow_j(1:2));
        d = sqrt((do+dd)/(L1*L2+eps));
end
end %


function [dmax, davg] = cluster_inner_distance(flows, ids, dist_mode)
if nargin < 3
    dist_mode = 2;
end
    
dmax = -inf;
davg = 0;
n = length(ids);
count = 0;
for i=1:(n-1)
    flow_i = flows(ids(i),:);
    for j=(i+1):n
        flow_j = flows(ids(j),:);
        d = fdij(flow_i, flow_j, dist_mode);
        dmax = max([dmax, d]);
        davg = davg + d;
        count = count+1;
    end
end
davg = davg/count;
end


function dr = cluster_direction_similarity(flows, c1_ids, c2_ids)
v1 = mean(flows(c1_ids,:), 1);
v1 = v1(3:4)-v1(1:2);
v2 = mean(flows(c2_ids,:), 1);
v2 = v2(3:4)-v2(1:2);
dr = 180/pi*acos(sum(v1.*v2)/(norm(v1)*norm(v2)+eps));
end


function [dmin, davg] = cluster_between_distance(flows, c1_ids, c2_ids, dist_mode)
if nargin < 4
    dist_mode = 2;
end
n1 = length(c1_ids);
n2 = length(c2_ids);
dmin = inf;
davg = 0;
count = 0;
for i=1:n1
    flow_i = flows(c1_ids(i),:);
    for j=1:n2
        flow_j = flows(c2_ids(j),:);
        d = fdij(flow_i, flow_j, dist_mode);
        dmin = min([dmin, d]);
        davg = davg + d;
        count = count + 1;
    end
end
davg = davg/count;
end %



function [hotspots, st] = cluster_merge(flows, hotspots, dist_mode)
if nargin < 3
    dist_mode = 2;
end
mnums = arrayfun(@(x) length(x.Members), hotspots);
[~, ind] = sort(mnums, 'descend');
flag = false;
II = -1;
JJ = -1;
max_angle = 30; % degree
for i=1:(length(ind)-1)
    I = ind(i);
    for j=(i+1):length(ind)
        J = ind(j);
        A = hotspots(I).Members;
        B = hotspots(J).Members;
        if ~isempty(A) &&  ~isempty(B)
            [dmin, davg] = cluster_between_distance(flows, A, B, dist_mode);
            dr = cluster_direction_similarity(flows, A, B);
            if (davg <= 1.2*min([hotspots(I).InnerDis, hotspots(J).InnerDis])) && ...
                    (dr <= max_angle)
                II = I;
                JJ = J;
                flag = true;
                break;
            end
        else
            continue;
        end
    end
    if flag
        break;
    end
end
if II~=-1 && JJ~=-1
    hotspots(II).Members = unique([hotspots(II).Members, hotspots(JJ).Members]);
    [dmax, davg] = cluster_inner_distance(flows, hotspots(II).Members, dist_mode);
    hotspots(II).InnerDis = davg;
    hotspots(JJ) = [];
    st = true;
else
    st = false;
end
end %


function IDX = get_idx(hotspots, n, minpts)
IDX = zeros(n,1);
for i=1:length(hotspots)
    if length(hotspots(i).Members)>=minpts
        IDX(hotspots(i).Members) = i;
    end
end
end % 

