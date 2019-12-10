-- Training LSTM RNN for 3D joint data to recognize and predict
-- This supports multiple layers of LSTMs


require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'mla_utils'

local BatchDataLoader = require 'cl_dataLoader_batch'
--local DataLoader = require 'cl_dataLoader'
local LSTM = require 'MLA_LSTM'             -- LSTM timestep and utilities
local model_utils=require 'model_utils'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training LSTM for human action')
cmd:text()
cmd:text('Options')
cmd:option('-dataDir','../../data/ChaL/','data directory. Should contain the file train.h5 and test.h5')
cmd:option('-rnnSz',300,'size of LSTM internal state')
cmd:option('-nIter', 1e9, 'Number of optimization iterations. Can be overriden by maxEpoch')
cmd:option('-maxEpoch',-1,'number of full passes through the training data. If not -1, override number of iterations.')
cmd:option('-modelFile','21_classify_init','filename to autosave the model (protos)')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-nLayer',3,'Number of LSTM layers')
cmd:option('-saveEvery',2000,'save every 200 steps, overwriting the existing file')
cmd:option('-printEvery',2000,'how many steps/minibatches between printing out the loss')
cmd:option('-init_from','','if not empty then use the checkpoint as initialization')
cmd:option('-checkpoint_dir', '../../result/lstm_classify_init','directory to save checkpoint files')
cmd:option('-maxLen',100, 'maximum length of an entire sequence')
cmd:option('-batchSzCL',100, 'batch size for classification problem')
cmd:option('-trFile','seg_trP_All_batch.h5','file for training')
cmd:option('-trLabelFile','seg_trP_All_batchLabel.h5','label file for training')
cmd:option('-valFile','seg_trP_All_batch.h5','file for val')
cmd:option('-valLabelFile','seg_trP_All_batchLabel.h5','label file for val')
cmd:option('-tstFile','seg_valP_All_batch.h5','files for testing')
cmd:option('-tstLabelFile','seg_valP_All_batchLabel.h5','label file for testing')
cmd:option('-dropout',0,'parameter for dropout')
cmd:option('-optimization','adagrad','optimization method: adagrad, adadelta,sgd, rmsprop')
cmd:option('-skipConnect',true, 'LSTM with skip connection')
cmd:option('-BN',false, 'LSTM with Batch Normalization')
cmd:option('-weight',0.2, 'weight parameter for different time')
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


local trLoader  = BatchDataLoader.create(opt.dataDir..opt.trFile, opt.dataDir..opt.trLabelFile, opt.batchSzCL, true, opt.weight)
local valLoader  = BatchDataLoader.create(opt.dataDir..opt.valFile, opt.dataDir..opt.valLabelFile, opt.batchSzCL, true,opt.weight)
local tstLoader  = BatchDataLoader.create(opt.dataDir..opt.tstFile, opt.dataDir..opt.tstLabelFile, opt.batchSzCL, true,opt.weight)
--local valTrLoader  = DataLoader.create(opt.dataDir..opt.valFile, opt.batchSzCL)
--local tstLoader = DataLoader.create(opt.dataDir..opt.tstFile, opt.batchSzCL)
local logName = string.format('%s_%d_%d_%d_%s_d%.2f_skip-%s_BN-%s_loss+error.log',opt.modelFile,opt.nLayer,opt.rnnSz,
    opt.batchSzCL,opt.optimization, opt.dropout, opt.skipConnect, opt.BN )
print(logName)
print(opt.optimization)
local Log = optim.Logger(logName)

-- define model prototypes for ONE timestep, then clone them
local nJntCoord = trLoader.nJntCoord -- Number of joint coordinates
local nClass = trLoader.nClass -- Number of action class
--local nClass = 2
local protos = {}
local init_state_cla = {}
local iterInt
local randomIni = true
if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from )
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    print('overwriting rnn_size=' .. checkpoint.opt.rnnSz .. ', num_layers=' .. checkpoint.opt.nLayer .. ' based on the checkpoint.')
    opt.rnnSz = checkpoint.opt.rnnSz
    opt.nLayer = checkpoint.opt.nLayer
    iterInt = checkpoint.iteration+1
    randomIni = false
    print('overwriting iteration='..iterInt)
--    print('overwriting init state')     
--    if checkpoint.rnnNet.init_state_cla[1]:size(1) ~= opt.batchSzCL then
--        for i = 1, #checkpoint.rnnNet.init_state_cla do
--            init_state_cla[i] = torch.mean(checkpoint.rnnNet.init_state_cla[i],1):repeatTensor(opt.batchSzCL,1)
--        end
--    else 
--        init_state_cla = checkpoint.rnnNet.init_state_cla
--    end 
--    print(init_state_cla)     
else
    protos.embed = nn.Linear(nJntCoord, opt.rnnSz)
    -- lstm timestep's input: {x, prev_c1, prev_h1, ..., prev_ck, prev_hk}, output: {next_c1, next_h1, ..., next_ck, next_hk}
    protos.lstm = LSTM.lstm(opt.rnnSz, opt.nLayer, opt.dropout, opt.skipConnect, opt.BN)
    protos.invEmbed2label = nn.Linear(opt.rnnSz, nClass)
    protos.classify = nn.LogSoftMax()
    protos.clcriterion = nn.ClassNLLCriterion() 
    iterInt = 1
end

local classWeight = torch.Tensor(nClass):fill(10)
classWeight[nClass] = 1;
classWeight = classWeight:div(classWeight:sum())

protos.clcriterion = nn.ClassNLLCriterion(classWeight)

local rnnNet = {}
rnnNet.protos = protos;
rnnNet.nLayer = opt.nLayer

for L=1,opt.nLayer do
    local h_init_cla = torch.zeros(opt.batchSzCL, opt.rnnSz)
    if opt.gpuid >=0 then 
        h_init_cla = h_init_cla:cuda() 
    end    
    table.insert(init_state_cla, h_init_cla:clone())
    table.insert(init_state_cla, h_init_cla:clone())
 end

-- the initial state of the cell/hidden states
local init_state_global_cla = clone_list(init_state_cla)

if opt.gpuid >=0 then 
    for k,v in pairs(protos) do v:cuda() end 
end

-- put the above things into one flattened parameters tensor
local paramsAll, grad_params_all = model_utils.combine_all_parameters(protos.embed, protos.lstm, protos.invEmbed2label)
if randomIni then
    paramsAll:uniform(-0.08, 0.08)
end
print(protos.embed:getParameters():mean())
-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.maxLen, not proto.parameters)
end


-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= paramsAll then
        paramsAll:copy(params_)
    end
    grad_params_all:zero()    

    --------------------------------- evaluate on classification problem ------------------------------
    ------------------ get minibatch -------------------
    -- x is Tensor batchSzCL*nJntCoord*seqLen, y is label batchSzCL*1
    local x, y, z = trLoader:next_batch_CL()  

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end    
    

    local batchSeqLen = x:size(3)
    local embeddings = {}            -- input embeddings
    local rnn_state_cla = {[0] = init_state_global_cla} -- internal cell states of LSTM   
    local rnn_final_output = {} 
    local temp_labels = {}           -- invEmbed label for each frame
    local current_labels = {}
    local cLoss = 0                  -- classification loss
    
    -- weight represent the weight of different time step. It should have the same length as time step 
--    local weight = torch.ones(batchSeqLen)
--    weight[-1] = 1
    local weight = z
    
    ------------------------- forward pass -------------------------
    for t = 1, batchSeqLen do
        embeddings[t] = clones.embed[t]:forward(x[{{}, {}, t}]:squeeze())         
        clones.lstm[t]:training()
        local lst = clones.lstm[t]:forward{embeddings[t], unpack(rnn_state_cla[t-1])}           
        rnn_state_cla[t] = {}
        for i=1,#init_state_cla do 
            table.insert(rnn_state_cla[t], lst[i]) 
        end
        if opt.skipConnect then
            rnn_final_output[t] = lst[#lst]
        end
        temp_labels[t] = clones.invEmbed2label[t]:forward(lst[#lst]) -- Use the last output to recognize
        current_labels[t] = protos.classify:forward(temp_labels[t])
        -- if you forward multiple data in ClassNLLCriterion, the loss is already averaged.              
        cLoss = cLoss + protos.clcriterion:forward(current_labels[t], y[{{},t}]:squeeze())       
    end 

--    local labels = current_labels[batchSeqLen]:clone()
--    local errors = 0
--    local _,pre_sorted = labels:sort(2,true)
--    local pre = pre_sorted[{{},1}]
--    for seqNum = 1, x:size(1) do
--        local gtnum = y[seqNum]:squeeze()
--        local p = pre[seqNum]
--        if p ~= gtnum then
--            errors = errors + 1
--        end
--    end
--    errors = errors / x:size(1)
--    print(string.format('error on current batch %.2f, batch length %d',errors, batchSeqLen))
    
    ------------------------- backward pass -------------------------
    -- complete reverse order of the above
    local drnn_state
    if opt.skipConnect then
        local drnn_state_tmp = clone_list(init_state_cla, true)  
        table.insert(drnn_state_tmp,torch.zeros(opt.batchSzCL, opt.rnnSz):cuda())    
        drnn_state = {[batchSeqLen] = clone_list(drnn_state_tmp, true)} -- true also zeros the clones 
    else
        drnn_state = {[batchSeqLen] = clone_list(init_state_cla, true)} -- true also zeros the clones 
    end
       
    for t= batchSeqLen,1,-1 do
        local weight_temp = weight[{{},t}]:repeatTensor(1,nClass):float():cuda()
        local doutput_t = protos.clcriterion:backward(current_labels[t], y[{{},t}]:squeeze()):cmul(weight_temp)
        local dl_dlabel = protos.classify:backward(temp_labels[t], doutput_t)
        local dtoph_t
        if opt.skipConnect then
            dtoph_t = clones.invEmbed2label[t]:backward(rnn_final_output[t], dl_dlabel)
        else
            dtoph_t = clones.invEmbed2label[t]:backward(rnn_state_cla[t][#rnn_state_cla[t]], dl_dlabel)
        end
        
        -- Two cases for dloss/dhtop_t 
        --  1. toph_T is only used once, sent to the invEmbed (but not to the next LSTM time step)
        --  2. toph_t is used twice, for the invEmbed and the next step. To obey the  
        --     multivariate chain rule, we add them.
        -- In both cases, we add them, assuming initially the derivative vector is zero
        drnn_state[t][#drnn_state[t]]:add(dtoph_t)
        local dlst = clones.lstm[t]:backward({embeddings[t], unpack(rnn_state_cla[t-1])}, drnn_state[t])       
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- the next elements are derivative of c_1,h_1, ..., c_top, h_top
                drnn_state[t-1][k-1] = v
            end
        end
        if opt.skipConnect then 
            drnn_state[t-1][#dlst] = torch.zeros(opt.batchSzCL, opt.rnnSz):cuda()
        end
        local dembedding_t = dlst[1]       
        clones.embed[t]:backward(x[{{}, {}, t}]:squeeze(), dembedding_t)
    end

    -- transfer final state to initial state (BPTT)
--    init_state_global_cla = clone_list(rnn_state_cla[#rnn_state_cla])  


    -- end of the batch

--    grad_params_all
    cLoss = cLoss

    -- clip gradient element-wise 
    grad_params_all:clamp(-5, 5)

    return cLoss, grad_params_all
end

local function getLoss(Loader)
    local cLosses = 0
    local errors = 0
    local nBatch2evalCL = Loader.nBatchCL
    local init_state_eval_cla = init_state_global_cla
    local cntFrm = 0
    local cnt = 0
    for i = 1, nBatch2evalCL do
        local x, y = Loader:next_batch_CL()
        if opt.gpuid >= 0 then
            x = x:float():cuda()
            y = y:float():cuda()
        end
        local batchSeqLen = x:size(3)
        cntFrm = cntFrm + batchSeqLen
        cnt = cnt + x:size(1)
        
        local weight = torch.ones(batchSeqLen)
--        weight[-1] = 1
        
        local embeddings = {}            -- input embeddings
        local rnn_state_cla = {[0] = init_state_global_cla} -- internal cell states of LSTM    
        local temp_labels = {}           -- invEmbed label for each frame
        local current_labels = {}
        local cLoss = 0                  -- classification loss
        ------------------------- forward pass -------------------------
        for t = 1, batchSeqLen do
            embeddings[t] = clones.embed[t]:forward(x[{{}, {}, t}]:squeeze())         
            clones.lstm[t]:evaluate()
            local lst = clones.lstm[t]:forward{embeddings[t], unpack(rnn_state_cla[t-1])}           
            rnn_state_cla[t] = {}
            for i=1,#init_state_global_cla do 
                table.insert(rnn_state_cla[t], lst[i]) 
            end
            temp_labels[t] = clones.invEmbed2label[t]:forward(lst[#lst]) -- Use the last h to recognize
            current_labels[t] = protos.classify:forward(temp_labels[t])
            -- if you forward multiple data in ClassNLLCriterion, the loss is already averaged.
            cLoss = cLoss + weight[t]*protos.clcriterion:forward(current_labels[t], y[{{},t}]:squeeze())   
        end
        local labels = current_labels[batchSeqLen]
        cLosses = cLosses + cLoss
        
        local _,pre_sorted = labels:sort(2,true)
        local pre = pre_sorted[{{},1}]
        for seqNum = 1, x:size(1) do
            local gtnum = y[{seqNum,batchSeqLen}]
            local p = pre[seqNum]
            if p ~= gtnum then
                errors = errors + 1
            end
        end
        
    end
    -- cLoss already averaged 
    -- cLosses averaged by #of batch
    cLosses = cLosses/nBatch2evalCL
    errors = errors/cnt    
    return cLosses, errors   
end

-- optimization stuff
local adagrad_optim_state = {learningRate = 1e-3}
local rmsprop_optim_state = {learningRate = 1e-2, alpha = 0.95}
local sgd_optim_state = {learningRate = 1e-3, learningRateDecay = 1e-7, momentum = 0.95, weightDecay = 1e-6 }
local adadelta_optim_state = {learningRate = 1e-3, decay = 1e-6, epsilon = 1e-7}
local nIter
if opt.maxEpoch >= 1 then
    nIter = opt.maxEpoch * trLoader.nBatchCL
else
    nIter = opt.nIter
end
print('Finish setting up. Start optimization now. The number of epoch is ' .. nIter/trLoader.nBatchCL..'. The number of iterations is '..nIter)

local optIter
local startTimer = torch.Timer()

local nIterEpoch = trLoader.nBatchCL
local trErrors = 0
local valErrors = 0
local trCLosses = 0
local valCLosses = 0 


Log:setNames({'Iter', 'Epoch', 'loss', 'trLoss', 'trError','valLoss', 'valError','time/iter','elpaseT'})
for optIter = iterInt, nIter do 
    local epoch = optIter / trLoader.nBatchCL
    local timer = torch.Timer()
    local loss
--    truntrLosses, truntrErrors = getLossOnTrainData()
--    trCLosses, trErrors = getLossOnValiData(valTrLoader)
--    valCLosses, valErrors = getLossOnValiData(tstLoader)
    if opt.optimization == 'adagrad' then
        _, loss = optim.adagrad(feval, paramsAll, adagrad_optim_state)
    elseif opt.optimization == 'rmsprop' then
        _, loss = optim.rmsprop(feval, paramsAll, rmsprop_optim_state)
    elseif opt.optimization == 'sgd' then
        _, loss = optim.sgd(feval, paramsAll, sgd_optim_state)
    elseif opt.optimization == 'adadelta' then
        _, loss = optim.adadelta(feval, paramsAll, adadelta_optim_state)          
    end
    local time = timer:time().real        
    if optIter % opt.printEvery == 0 then 
        trCLosses, trErrors = getLoss(valLoader)
        valCLosses, valErrors = getLoss(tstLoader)
        local eTime = startTimer:time().real
        print(string.format("Iter:%d/%d, Epoch:%.1f, loss = %5.3e, trLoss = %5.3e, trError = %5.3e, valLoss = %5.3e, valError = %5.3e, gradnorm = %5.3e, time/iter = %3.0es, elpaseT: %.0fs",
            optIter, nIter, epoch, loss[1], trCLosses,trErrors, valCLosses,valErrors, grad_params_all:norm(), time, eTime))
        --        Log:add{['Iter'] = optIter, ['Epoch']=epoch, ['trLoss']=trCLosses, ['trError']=trErrors, ['valLoss']=valCLosses,
        --                ['valError']=valErrors, ['time/iter']=time/nIterEpoch, ['elpaseT']=eTime }
        Log:add{optIter, epoch, loss[1], trCLosses,trErrors, valCLosses,valErrors,
            time, eTime }          
--        Log:plot('truncateBatchError', 'trError', 'valError')     
    end
    if optIter % opt.saveEvery == 0 then  
        -- save model
--        rnnNet.init_state_cla = {}
--        for i=1,#init_state_global_cla do
--            rnnNet.init_state_cla[i] = init_state_global_cla[i]
--        end        
--        torch.save(string.format('%s_%d_%d_%d_%s_d%.2f.t7',opt.modelFile,opt.nLayer,opt.rnnSz,opt.batchSzCL,opt.optimization, opt.dropout),
--            rnnNet)

        local savefile = string.format('%s/%s_%.1f_%d_%d_%d_%s_d%.2f_skip-%s_BN-%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.modelFile, opt.weight,
            opt.nLayer,opt.rnnSz,opt.batchSzCL,opt.optimization, opt.dropout, opt.skipConnect, opt.BN, epoch, valErrors)

        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_errors = trErrors
        checkpoint.val_errors = valErrors
        checkpoint.optIter = optIter
        checkpoint.epoch = epoch
        checkpoint.iteration = optIter
        checkpoint.rnnNet = rnnNet
        torch.save(savefile, checkpoint)               
    end
end  

