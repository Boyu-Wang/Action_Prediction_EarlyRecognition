-- jointly prediction and classification

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'mla_utils'

local LSTM = require 'MLA_LSTM'             -- LSTM timestep and utilities
local model_utils=require 'model_utils'
local myHdf5 = require 'ML_HDF5'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training LSTM for human action')
cmd:text()
cmd:text('Options')
cmd:option('-dataDir','../../data/ChaL/','data directory. Should contain the file train.h5 and test.h5')
cmd:option('-cla_modelFile','../../result/lstm_classify_init_skipCon/20_classify_init_skipCon_5_300_100_adagrad_d0.00_true_epoch451.61_0.1434.t7','filename to autosave the model (protos)')
cmd:option('-pred_modelFile','../../result/lstm_cdt_pred/cdt_det_pred_5_300_100_adagrad_d0.00_true_epoch36.50_0.0289.t7','filename to autosave the model (protos)')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-batchSzCL',100, 'batch size for classification problem')
cmd:option('-tstFile','seg_tstP_Act_whiten','file for training')
cmd:option('-tstLabelFile','seg_tstP_Act_whiten_Label.h5','files for testing')
cmd:option('-dropout',0,'parameter for dropout')
cmd:option('-optimization','adagrad','optimization method: adagrad, adadelta,sgd, rmsprop')
cmd:option('-saveTstFile','')
cmd:option('-saveTstModel','')
cmd:option('-saveDir','../../result/lstm_classify_test')
cmd:option('-ratio',0.2)
cmd:text()

if not arg[1] then
    arg = {...}
end 

-- parse input params
local opt = cmd:parse(arg)


-- preparation stuff:
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    opt.gpuid = mla_setupCuda(opt.gpuid, opt.seed)
end


local oriTstData = myHdf5.load(opt.dataDir..opt.tstFile)
local oriTstLabel = myHdf5.load(opt.dataDir..opt.tstLabelFile)

-- define model prototypes for ONE timestep, then clone them
local nJntCoord = oriTstData[1]:size(2)--tstLoader.nJntCoord -- Number of joint coordinates
local nClass=0
for i = 1, #oriTstLabel do
    local temp = oriTstLabel[i]:max()
    if nClass < temp then
        nClass = temp
    end
end
local iterInt
local randomIni = true

--load prediction model
print('loading prediction LSTM from checkpoint ' .. opt.pred_modelFile )
local pred_checkpoint = torch.load(opt.pred_modelFile)
local pred_protos = pred_checkpoint.protos
-- make sure the vocabs are the same
print('prediction rnn_size=' .. pred_checkpoint.opt.rnnSz .. ', num_layers=' .. pred_checkpoint.opt.nLayer .. ' based on the checkpoint.')
opt.pred_rnnSz = pred_checkpoint.opt.rnnSz
opt.pred_nLayer = pred_checkpoint.opt.nLayer
randomIni = false

--load classification model
print('loading classification LSTM from checkpoint ' .. opt.cla_modelFile )
local cla_checkpoint = torch.load(opt.cla_modelFile)
local cla_protos = cla_checkpoint.protos
cla_protos.prob = nn.SoftMax()
-- make sure the vocabs are the same
print('prediction rnn_size=' .. cla_checkpoint.opt.rnnSz .. ', num_layers=' .. cla_checkpoint.opt.nLayer .. ' based on the checkpoint.')
opt.cla_rnnSz = cla_checkpoint.opt.rnnSz
opt.cla_nLayer = cla_checkpoint.opt.nLayer



-- the initial state of the cell/hidden states
--local init_state_global_cla = clone_list(init_state_cla)

if opt.gpuid >=0 then 
    for k,v in pairs(pred_protos) do v:cuda() end 
    for k,v in pairs(cla_protos) do v:cuda() end
end


local cLosses = 0
local errors = 0
local nBatch2evalCL = #oriTstLabel
local init_state_cla = {}
for L=1,opt.cla_nLayer do
    local h_init_cla = torch.zeros(1, opt.cla_rnnSz)
    if opt.gpuid >=0 then 
        h_init_cla = h_init_cla:cuda() 
    end    
    table.insert(init_state_cla, h_init_cla:clone())
    table.insert(init_state_cla, h_init_cla:clone())
end

local init_state_pred = {}
for L=1,opt.pred_nLayer do
    local h_init_pred = torch.zeros(1, opt.pred_rnnSz)
    if opt.gpuid >=0 then 
        h_init_pred = h_init_pred:cuda() 
    end    
    table.insert(init_state_pred, h_init_pred:clone())
    table.insert(init_state_pred, h_init_pred:clone())
end

local gtLabel = torch.Tensor(nBatch2evalCL)
local predRawValue = torch.Tensor(nClass, nBatch2evalCL)
--local predSeqs = {}

local oneHotLabelVec = torch.eye(nClass):cuda()
local avgLabelVec = torch.Tensor(1, nClass):fill(1/nClass):cuda()

local cnt = 1;
local initLen = 20
for i = 1, nBatch2evalCL do
    print(i)
     local x = oriTstData[i]
    local y = oriTstLabel[i]
    local batchSeqLen = y:size(2)    
    if opt.gpuid >= 0 then
        x = x:float():cuda()
        y = y:float():cuda()
    end

    ------------------------- forward pass -------------------------
    
    local rnn_state_cla = {[0] = init_state_cla}
    local rnn_state_pred = {[0] = init_state_pred}
    local temp_prob
    local embeddings_pred, embeddings_cla
    local data_input
    local data_pred
    local labelVector
    local predSeq = torch.Tensor(x:size())
    labelVector = avgLabelVec
    
    local startLen = math.ceil(batchSeqLen * opt.ratio)
    if startLen < initLen then
        local tmp_embeddings
        local prev_pose = x[{{1},{}}]
        local rnn_state_pred_init = {[0] = init_state_pred}
        for t = 1, initLen-startLen do
            tmp_embeddings = pred_protos.embed:forward{prev_pose, labelVector}
            local lst = pred_protos.lstm:forward{tmp_embeddings, unpack(rnn_state_pred_init[t-1])}
            rnn_state_pred_init[t] = {}
            for i = 1, # init_state_pred do
                table.insert(rnn_state_pred_init[t], lst[i])
            end
        end
        rnn_state_pred[0] = rnn_state_pred_init[initLen-startLen]
    end
    
    data_input = x[{{1},{}}]
    predSeq[{{1,startLen+1},{}}]:copy(x[{{1, startLen+1},{}}])
    
    for t = 1, batchSeqLen do        
        -- feed into classification network
        embeddings_cla = cla_protos.embed:forward(data_input)
        cla_protos.lstm:evaluate()
        local lst = cla_protos.lstm:forward{embeddings_cla, unpack(rnn_state_cla[t-1])}
        rnn_state_cla[t] = {}
        for i=1,#init_state_cla do 
            table.insert(rnn_state_cla[t], lst[i]) 
        end
        local invEmbed_cla = cla_protos.invEmbed2label:forward(lst[#lst])
        temp_prob = cla_protos.prob:forward(invEmbed_cla)
        
        if t == batchSeqLen then
            gtLabel[i]=y[{1, 1}]
            predRawValue[{{},i}]:copy(temp_prob:float():double())
        end
        
        -- feed into prediction network  
        if t < batchSeqLen then        
--            labelVector = temp_prob
            local max_value, max_idx = torch.max(temp_prob, 2)
            labelVector = oneHotLabelVec[{{max_idx[{1,1}]}, {}}]
                         
            embeddings_pred = pred_protos.embed:forward{data_input, labelVector}
            local lst = pred_protos.lstm:forward{embeddings_pred, unpack(rnn_state_pred[t-1])}
            rnn_state_pred[t] = {}
            for i = 1, # init_state_pred do
                table.insert(rnn_state_pred[t], lst[i])
            end
            data_pred = pred_protos.invEmbed:forward(lst[#lst])         
            
            if t <= startLen then
                data_input = x[{{t+1}, {}}]
            else
                data_input = data_pred
                predSeq[{{t+1},{}}]:copy(data_pred)
            end
        end
        
    end
    
--    predSeqs[cnt] = predSeq:float():double()
--    cnt = cnt + 1
end
        
                   
local gtLabel_tosave = {}
local predRawValue_tosave = {}
gtLabel_tosave[1] = gtLabel
predRawValue_tosave[1] = predRawValue

local dim2 = torch.Tensor{#gtLabel_tosave, 1}
local dim4 = torch.Tensor{#predRawValue_tosave, 1}
--local dim5 = torch.Tensor{#predSeqs, 1}

myHdf5.save(string.format('%s/%s_%s_gtLabel.h5', opt.saveDir, opt.saveTstFile, opt.saveTstModel), gtLabel_tosave, dim2)
myHdf5.save(string.format('%s/%s_%s_predRaw.h5', opt.saveDir, opt.saveTstFile, opt.saveTstModel), predRawValue_tosave, dim4)

--myHdf5.save(string.format('%s/%s_%s_predSeq.h5', opt.saveDir, opt.saveTstFile, opt.saveTstModel), predSeqs, dim5)

        
        