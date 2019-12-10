-- Create data loader for prediction and recognition

require 'torch'
require 'math'
require 'ml_kFoldCV_Idxs'

local myHdf5 = require 'ML_HDF5'
local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.create(dataFile, labelFile, batchSzCL, isweighted, weight)
    local self = {}
    setmetatable(self, DataLoader)

    -- construct a tensor with all the data
    print('loading data files...')    
    local alldata = myHdf5.load(dataFile)
    local allLabel = myHdf5.load(labelFile)

    print('processing data for classification')    
    local nBatchCL = 0
    local nSeqCL = 0
    self.x_batches_CL_temp = {}
    self.y_batches_CL_temp = {}
    self.z_batches_CL_temp = {}
    self.x_batches_CL = {}
    self.y_batches_CL = {}
    self.z_batches_CL = {}
    local index = 1
    local nJntCoord
    if alldata[1]:size():size() < 3 then
        nJntCoord = alldata[1]:size(1)
    else
        nJntCoord = alldata[1]:size(2)
    end
--    print(alldata)
--    print(#allLabel[1])
    for i = 1, #alldata do
        local tempNumSeq 
        local tempSeqLen 
        if alldata[i]:size():size() < 3 then
            tempNumSeq = 1
            tempSeqLen = alldata[i]:size(2)
        else
            tempNumSeq = alldata[i]:size(1)
            tempSeqLen = alldata[i]:size(3)
        end
        local tempNumBatch = math.floor(tempNumSeq/batchSzCL)
        nBatchCL = nBatchCL + tempNumBatch
        nSeqCL = nSeqCL + tempNumSeq
        local batchIdx = ml_kFoldCV_Idxs(tempNumSeq, tempNumBatch, true)

        for j = 1, tempNumBatch do
            self.x_batches_CL_temp[index] = torch.Tensor(batchSzCL, nJntCoord, tempSeqLen)
            A = self.x_batches_CL_temp[index]
            self.y_batches_CL_temp[index] = torch.Tensor(batchSzCL, tempSeqLen)
            B = self.y_batches_CL_temp[index]
            self.z_batches_CL_temp[index] = torch.Tensor(batchSzCL, tempSeqLen)
            C = self.z_batches_CL_temp[index]
            for k = 1, batchSzCL do
                A[{k,{},{}}] = alldata[i][batchIdx[j][k]]
                B[{k,{}}] = allLabel[i][batchIdx[j][k]]
                if isweighted then
                    local t1 = B[{k,{}}]
                    local l1 = torch.Tensor{1}
                    for m = 1, t1:size(1)-1 do
                        if t1[m+1] ~= t1[m] then
                            l1 = l1:cat(torch.Tensor{m+1})
                        end
                    end
                    l1 = l1:cat(torch.Tensor{t1:size(1)+1})
                    local l2 = torch.Tensor(l1:size(1)-1)
                    for m = 1, l1:size(1)-1 do
                        l2[m] = l1[m+1] - l1[m]
                    end
                    for m = 1, l2:size(1) do
                        local t1a = torch.range(1, l2[m])
                        t1a = (t1a - math.floor(l2[m]/2))*weight
                        local t1c = t1a:exp() + 1
                        local t1b = t1a:cdiv(t1c)
                        t1b = t1b:div(t1b:sum())                
                        C[{k,{l1[m], l1[m+1]-1}}] = t1b
                    end
                else
                    C[{k,{}}] = torch.ones(tempSeqLen)
                end
                
            end
            -- normalize the weight
            C = C:cdiv(C:sum(2):repeatTensor(1, tempSeqLen))
            index = index + 1
        end
    end
    
--    shuffle = torch.randperm(nBatchCL)
    shuffle = torch.range(1,nBatchCL)
    for i = 1, #self.x_batches_CL_temp do
        self.x_batches_CL[i] = self.x_batches_CL_temp[shuffle[i]]
        self.y_batches_CL[i] = self.y_batches_CL_temp[shuffle[i]]
        self.z_batches_CL[i] = self.z_batches_CL_temp[shuffle[i]]
    end
    local nClass=0
    for i = 1, #allLabel do
        local temp = allLabel[i]:max()
        if nClass < temp then
            nClass = temp
        end
    end
    self.nJntCoord = nJntCoord
    
    self.nBatchCL = nBatchCL
    self.batchSzCL = batchSzCL
    self.currentBatchCL = 0
    self.nBatchEvalCL = 0  -- number of times next_batch() called
    self.nClass = nClass
    print('data load done.')
    collectgarbage()
    print(string.format('Finish creating data loader, classification: nSeq = %d, nBatch = %d', nSeqCL, nBatchCL))
    return self
end


function DataLoader:next_batch_CL()
    self.currentBatchCL = (self.currentBatchCL % self.nBatchCL) + 1
    self.nBatchEvalCL = self.nBatchEvalCL + 1
    return self.x_batches_CL[self.currentBatchCL], self.y_batches_CL[self.currentBatchCL],self.z_batches_CL[self.currentBatchCL]
end



return DataLoader
