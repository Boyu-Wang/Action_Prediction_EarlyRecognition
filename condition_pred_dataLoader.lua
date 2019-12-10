-- Create data loader


require 'torch'
require 'math'
require 'ml_kFoldCV_Idxs'

local myHdf5 = require 'ML_HDF5'
local DataLoader = {}
DataLoader.__index = DataLoader

local function crtShortSeqs(train, label, seqLen, stepSz, labelOutPut)
    local allShortSeqs = {}
    local shftShortSeqs = {}
    local labelShortSeqs = {}
    for i = 1,#train do
      len = train[i]:size(1)
      if len >= seqLen + 1 then
        for j=1,(len-seqLen),stepSz do
          allShortSeqs[#allShortSeqs+1] = train[i][{ {j,j+seqLen-1}, {} }]:transpose(1,2)
          shftShortSeqs[#shftShortSeqs+1] = train[i][{ {j+1,j+seqLen}, {} }]:transpose(1,2)  
          if labelOutPut and label[1]:size(1) == 1 then     
            -- here labelShortSeqs is one dimension vector.        
            labelShortSeqs[#labelShortSeqs+1] = label[i][{1,{j, j+seqLen-1}}]
          elseif labelOutPut and label[1]:size(1) ~= 1  then
            -- here labelShortSeqs is nClass*seqLen tensor
            labelShortSeqs[#labelShortSeqs+1] = label[i][{{},{j, j+seqLen-1}}]:exp()
          else   
            labelShortSeqs[#labelShortSeqs+1] = label[i][{1,j}]    
          end
        end
      end
    end
    return allShortSeqs, shftShortSeqs, labelShortSeqs
end

function DataLoader.create(dataFile, labelFile, batchSz, seqLen, stepSz, labelOutPut)
    local self = {}
    setmetatable(self, DataLoader)
    -- construct a tensor with all the data
    print('loading data files...')    
    local data = myHdf5.load(dataFile) 
    local label = myHdf5.load(labelFile)

    local allShortSeqs, shftShortSeqs, labelShortSeqs = crtShortSeqs(data, label, seqLen, stepSz, labelOutPut)
        
    local nSeq = #allShortSeqs
    local nBatch = math.floor(nSeq/batchSz)
    local batchIdxss = ml_kFoldCV_Idxs(nSeq, nBatch, 1)
    local nJntCoord = allShortSeqs[1]:size(1)
    -- local nClass = label:max()
    local nClass
    if labelOutPut and label[1]:size(1)~=1 then
      nClass = label[1]:size(1)
    else
      nClass=0
      for i = 1, #label do
          local temp = label[i]:max()
          if nClass < temp then
              nClass = temp
          end
      end
    end
    local eyeMatrix = torch.eye(nClass)
    
    self.x_batches = {}
    self.y_batches = {}  
    self.l_batches = {}  
    for i = 1,nBatch do
      self.x_batches[i] = torch.Tensor(batchSz, nJntCoord, seqLen)
      A = self.x_batches[i]
      
      self.y_batches[i] = torch.Tensor(batchSz, nJntCoord, seqLen)
      B = self.y_batches[i]
      if labelOutPut then
        self.l_batches[i] = torch.Tensor(batchSz, nClass, seqLen)
      else
        self.l_batches[i] = torch.Tensor(batchSz, nClass)        
      end
      C = self.l_batches[i]

      for j=1,batchSz do -- Essentially, we throw away some data 
        A[{j,{},{}}] = allShortSeqs[batchIdxss[i][j]]
        B[{j,{},{}}] = shftShortSeqs[batchIdxss[i][j]]
        if labelOutPut and label[1]:size(1) == 1 then
          C[{j,{}}] = eyeMatrix:index(2, labelShortSeqs[batchIdxss[i][j]]:long() )
        elseif labelOutput and label[1]:size(1) ~= 1 then
          C[{j,{},{}}] = labelShortSeqs[batchIdxss[i][j]]
        else
          C[{j,{}}] = eyeMatrix[{labelShortSeqs[batchIdxss[i][j]],{}}]
        end
      end
    end
        
    self.nBatch = nBatch
    self.batchSz = batchSz
    self.seqLen = seqLen
    self.currentBatch = 0
    self.nBatchEval = 0  -- number of times next_batch() called
    self.nJntCoord = nJntCoord;
    self.nClass = nClass
    print('data load done.')
    collectgarbage()
    print(string.format('Finish creating data loader, nSeq = %d, nBatch = %d', nSeq, nBatch))
    return self
end

function DataLoader:next_batch()
    self.currentBatch = (self.currentBatch % self.nBatch) + 1
    self.nBatchEval = self.nBatchEval + 1
    return self.x_batches[self.currentBatch], self.y_batches[self.currentBatch], self.l_batches[self.currentBatch]
end

return DataLoader
