function ARI=adjusted_rand_score(predicted_labels, ground_truth_labels)
%ADJRAND  Adjusted rand index (consider each noise as a single cluster).
%
% Computes the adjusted Rand index to assess the quality of a clustering.
% Perfectly random clustering returns the minimum score of 0, perfect
% clustering returns the maximum score of 1.
%
% INPUTS
%    predicted_labels = the labeling as predicted by a clustering algorithm
%    ground_truth_labels = the true labeling
%
% OUTPUTS
%    adjrand = the adjusted Rand index
%
%
% Author: Tijl De Bie, february 2003.

predicted_labels = make_sorted_idx(predicted_labels);
ground_truth_labels = make_sorted_idx(ground_truth_labels);

n=length(predicted_labels);
ku=max(predicted_labels);
kv=max(ground_truth_labels);
m = zeros(ku,kv);
for i=1:n
    m(predicted_labels(i),ground_truth_labels(i)) = m(predicted_labels(i),ground_truth_labels(i)) + 1;
end
mu = sum(m,2);
mv = sum(m,1);

a = 0;
for i=1:ku
    for j=1:kv
        if m(i,j)>1
            a = a+nchoosek(m(i,j),2);
        end
    end
end

b1=0;
b2=0;
for i=1:ku
    if mu(i)>1
        b1=b1+nchoosek(mu(i),2);
    end
end
for i=1:kv
    if mv(i)>1
        b2=b2+nchoosek(mv(i),2);
    end
end

c=nchoosek(n,2);

ARI = (a-b1*b2/c)/(0.5*(b1+b2)-b1*b2/c);



function IX = make_sorted_idx(IDX)
clusterids = unique(IDX(IDX>0));
IX = zeros(size(IDX));
n = length(clusterids);
for i=1:n
    IX(IDX==clusterids(i)) = i;
end
IX(isnan(IX)) = 0;
m = sum(IX<=0);
IX(IX<=0) = (n+1):(n+m);



