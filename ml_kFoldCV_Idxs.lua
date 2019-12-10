-- Get the indexes of k-fold Cross Validation.
-- Adapated from Matlab code


require 'torch'
require 'math'

function ml_kFoldCV_Idxs(n, k, shldRandom)
--% idxss = ml_kFoldCV_Idxs(n, k, shldRandom)
--% Get the indexes of k-fold Cross Validation.
--% Suppose your training data has n examples and you want to perform k-fold CV
--% You need to divide the training data into k equal partitions (as equal as possible).
--% This function provides indexes for the partitions.
--% Inputs:
--%   n: the number of training examples.
--%   k: the number of folds of k-fold CVs.
--%   shldRandom: should we shuffle the indexes from 1:n randomly before dividing into partions?
--%       The default value is 1.
--% Outputs:
--%   idxss: a table, idxss[i] is a list of indexes for test data of the i^th fold.
    local suffledIdxs
    if shldRandom then
        suffledIdxs = torch.randperm(n);
    else
        suffledIdxs = torch.range(1,n)
    end;
    
    local q = math.floor(n/k);
    local r = n - k*q;
    local idxss = {}
    for i=1,k do
        if i<=r then
            idxss[i] = suffledIdxs[{{1+(i-1)*(q+1),i*(q+1)}}];        
        else
            idxss[i] = suffledIdxs[{{1+r+(i-1)*q,r+i*q}}];
        end;
    end;
    
    return idxss
end
