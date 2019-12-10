
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'mla_utils'
-- require 'GaussianCriterion'

local DataLoader = require 'condition_pred_dataLoader'
local LSTM = require 'MLA_LSTM'             -- LSTM timestep and utilities
local model_utils=require 'model_utils'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training LSTM for human action')
cmd:text()
cmd:text('Options')
cmd:option('-dataDir','../../data/ChaL/','data directory. Should contain the file train.h5 and test.h5')
cmd:option('-trFile','seg_valP_Act_whiten.h5','file for training')
cmd:option('-trLabelFile','seg_valP_Act_whiten_Label.h5','Label file for training')
cmd:option('-valFile','seg_tstP_Act_whiten.h5','file for validation')
cmd:option('-valLabelFile','seg_tstP_Act_whiten_Label.h5','label file for validation')
cmd:option('-tstFile','seg_tstP_Act_whiten.h5','file for testing')
cmd:option('-tstLabelFile','seg_tstP_Act_whiten_Label.h5','label file for testing')
cmd:option('-seqLen',20,'number of timesteps to unroll to')
cmd:option('-seedLen', 0, 'number timesteps before computing the loss, for training. Default seqLen/2')
cmd:option('-rnnSz',300,'size of LSTM internal state')
cmd:option('-stepSz',1,'Step size for creating small training subsequences of length seqLen from long sequences.')
cmd:option('-batchSz',100,'number of sequences to train on in parallel')
cmd:option('-nIter', 1e9, 'Number of optimization iterations. Can be overriden by maxEpoch')
cmd:option('-maxEpoch',-1,'number of full passes through the training data. If not -1, override number of iterations.')
cmd:option('-modelFile','cdt_det_pred','filename to autosave the model (protos)')
cmd:option('-checkpoint_dir', '../../result/lstm_cdt_pred','directory to save checkpoint files')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-nLayer',3,'Number of LSTM layers')
cmd:option('-saveEvery',2000,'save every 1000 steps, overwriting the existing file')
cmd:option('-printEvery',2000,'how many steps/minibatches between printing out the loss')
cmd:option('-startModel','','if non empty, start the model from the file')
cmd:option('-dropout',0,'parameter for dropout')
cmd:option('-optimization','adagrad','optimization method: adagrad, adadelta,sgd, rmsprop')
cmd:option('-skipConnect',true, 'LSTM with skip connection')
cmd:text()

if not arg[1] then
  arg = {...}
end 

-- parse input params
local opt = cmd:parse(arg)

if opt.seedLen == -1 then
    opt.seedLen = math.floor(opt.seqLen/2)
end


-- preparation stuff:
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    opt.gpuid = mla_setupCuda(opt.gpuid, opt.seed)
end

local trLoader  = DataLoader.create(opt.dataDir..opt.trFile, opt.dataDir..opt.trLabelFile, opt.batchSz, opt.seqLen, opt.stepSz)
local valLoader = DataLoader.create(opt.dataDir..opt.valFile, opt.dataDir..opt.valLabelFile, opt.batchSz, opt.seqLen, opt.stepSz)
local tstLoader = DataLoader.create(opt.dataDir..opt.tstFile, opt.dataDir..opt.tstLabelFile, opt.batchSz, opt.seqLen, opt.stepSz)
logName = string.format('%s_%d_%d_%d_%s_d%.2f_loss+error.log',opt.modelFile,opt.nLayer,opt.rnnSz,
    opt.batchSz,opt.optimization, opt.dropout)
print(logName)
print(opt.optimization)
local Log = optim.Logger(logName)

-- define model prototypes for ONE timestep, then clone them
local nJntCoord = trLoader.nJntCoord -- Number of joint coordinates
local nClass = trLoader.nClass -- Number of action class
local protos = {}
local init_state = {}
local iterInt
local randomIni = true

if string.len(opt.startModel) > 0 then
    print('loading model from checkpoint '..opt.startModel)
    local checkpoint = torch.load(opt.startModel)
    protos = checkpoint.protos  
    print('overwriting num_layers = ' .. checkpoint.opt.nLayer .. ' based on the checkpoint.')
    print('overwriting rnn_size = ' .. checkpoint.opt.rnnSz .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnnSz
    opt.nLayer = checkpoint.opt.nLayer
    iterInt = checkpoint.iteration + 1
    randomIni = false
    print('overwriting iteration='..iterInt)    
    print('overwriting init state')
    if checkpoint.rnnNet.init_state[1]:size(1) ~= opt.batchSz then
        for i = 1, #checkpoint.rnnNet.init_state do
            init_state[i] = torch.mean(checkpoint.rnnNet.init_state[i],1):repeatTensor(opt.batchSz,1)
        end
    else 
        init_state = checkpoint.rnnNet.init_state
    end    
--    for i = 1, #init_state do
--        init_state[i] = init_state[i]:repeatTensor(opt.batchSz, 1)
--    end
else    
    -- embedding layer is a linear layer whose input has 2 elements, one is Joints, another is class vector
    local function createEmbed(nJntCoord, nClass, rnnSz)
        local inputs = {}
        inputs[1] = nn.Identity()()
        inputs[2] = nn.Identity()()
        local concatenation = nn.JoinTable(2)(inputs)
        local output = nn.Linear(nJntCoord+nClass, opt.rnnSz)(concatenation)
        return nn.gModule(inputs, {output})
    end
    -- protos.embed = nn.Linear(nJntCoord+nClass, opt.rnnSz)
    protos.embed = createEmbed(nJntCoord, nClass, opt.rnnSz)
--    print(protos.embed:getParameters():size())
    -- lstm timestep's input: {x, prev_c1, prev_h1, ..., prev_ck, prev_hk}, output: {next_c1, next_h1, ..., next_ck, next_hk}
    protos.lstm = LSTM.lstm(opt.rnnSz, opt.nLayer, opt.dropout, opt.skipConnect) 
--    print(protos.lstm:getParameters():size())   
    protos.invEmbed = nn.Linear(opt.rnnSz, nJntCoord)
    -- protos.invEmbed1 = nn.Sequential():add(nn.Linear(opt.rnnSz, nJntCoord))
    -- protos.invEmbed2 = nn.Sequential():add(nn.Linear(opt.rnnSz, nJntCoord)):add(nn.Exp())
--    print(protos.invEmbed:getParameters():size())
    -- protos.criterion = nn.GaussianCriterion()
    protos.criterion = nn.MSECriterion()
    --protos.criterion = nn.AbsCriterion()
    iterInt = 1
    -- initialize init state with zeros
    for L=1,opt.nLayer do
        local h_init = torch.zeros(opt.batchSz, opt.rnnSz)
        if opt.gpuid >=0 then h_init = h_init:cuda() end    
        table.insert(init_state, h_init:clone())
        table.insert(init_state, h_init:clone())
    end
    
end

local rnnNet = {}
rnnNet.protos = protos;
rnnNet.nLayer = opt.nLayer


local init_state_global = clone_list(init_state)
if opt.gpuid >=0 then 
    for k,v in pairs(protos) do v:cuda() end 
end

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.lstm, protos.invEmbed1, protos.invEmbed2)
if randomIni then
    params:uniform(-0.08, 0.08)
end
print(params:size())
-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.seqLen, not proto.parameters)
end


-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()
    
    ------------------ get minibatch -------------------
    -- z is the label vector
    -- for input, concatenate x and z together
    local x, y, z = trLoader:next_batch() -- assume x is Tensor batchSz*nJntCoord*seqLen, y is shifted tensor by 1 time frame
    
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        z = z:float():cuda()
    end    

    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local rnn_state = {[0] = init_state_global} -- internal cell states of LSTM    
    -- local means = {}           -- invEmbed outputs
    -- local variance = {}
    local predPose = {}
    local loss = 0
    local rnn_final_output = {}

    for t=1,opt.seqLen do
        embeddings[t] = clones.embed[t]:forward{x[{{}, {}, t}]:squeeze(), z}
        clones.lstm[t]:training()
        local lst = clones.lstm[t]:forward{embeddings[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do 
            table.insert(rnn_state[t], lst[i]) 
        end 
        if opt.skipConnect then
            rnn_final_output[t] = lst[#lst]
        end

        if t > opt.seedLen then
            predPose[t] = clones.invEmbed[t]:forward(lst[#lst]) -- Use the last h to predict
            -- means[t] = clones.invEmbed1[t]:forward(lst[#lst]) -- Use the last h to predict
            -- variance[t] = clones.invEmbed2[t]:forward(lst[#lst]) -- Use the last h to predict
            -- loss = loss + clones.criterion[t]:forward(means[t], variance[t], y[{{}, {}, t}]:squeeze())
            loss = loss + clones.criterion[t]:forward(predPose[t], y[{{}, {}, t}]:squeeze())
        end
    end
        

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local drnn_state
    if opt.skipConnect then
        local drnn_state_tmp = clone_list(init_state, true)  
        table.insert(drnn_state_tmp,torch.zeros(opt.batchSz, opt.rnnSz):cuda())    
        drnn_state = {[opt.seqLen] = clone_list(drnn_state_tmp, true)} -- true also zeros the clones 
    else
        drnn_state = {[opt.seqLen] = clone_list(init_state_cla, true)} -- true also zeros the clones 
    end
    -- local drnn_state = {[opt.seqLen] = clone_list(init_state, true)} -- true also zeros the clones    
    
    for t=opt.seqLen,1,-1 do
        -- backprop through loss, and invEmbed/linear
        local doutput_mean, doutput_var
        if t > opt.seedLen then            
            -- doutput_mean, doutput_var = clones.criterion[t]:backward(means[t],variance[t], y[{{}, {}, t}]:squeeze())
            doutput = clones.criterion[t]:backward(predPose[t], y[{{}, {}, t}]:squeeze())
            -- local dtoph_t1
            -- local dtoph_t2
            local dtoph_t
            if opt.skipConnect then
                -- dtoph_t1 = clones.invEmbed1[t]:backward(rnn_final_output[t], doutput_mean)
                -- dtoph_t2 = clones.invEmbed2[t]:backward(rnn_final_output[t], doutput_var)
                dtoph_t = clones.invEmbed[t]:backward(rnn_final_output[t], doutput)
            else
                dtoph_t = clones.invEmbed[t]:backward(rnn_state[t][#rnn_state[t]], doutput)
                -- dtoph_t1 = clones.invEmbed1[t]:backward(rnn_state[t][#rnn_state[t]], doutput_mean)
                -- dtoph_t2 = clones.invEmbed2[t]:backward(rnn_state[t][#rnn_state[t]], doutput_var)
            end
            -- local dtoph_t1 = clones.invEmbed1[t]:backward(rnn_state[t][#rnn_state[t]], doutput_mean)
            -- local dtoph_t2 = clones.invEmbed2[t]:backward(rnn_state[t][#rnn_state[t]], doutput_var)
            -- local dtoph_t = dtoph_t1 + dtoph_t2
            -- Two cases for dloss/dhtop_t 
            --  1. toph_T is only used once, sent to the invEmbed (but not to the next LSTM time step)
            --  2. toph_t is used twice, for the invEmbed and the next step. To obey the  
            --     multivariate chain rule, we add them.
            -- In both cases, we add them, assuming initially the derivative vector is zero
            drnn_state[t][#drnn_state[t]]:add(dtoph_t)
        end                
        
        local dlst = clones.lstm[t]:backward({embeddings[t], unpack(rnn_state[t-1])}, drnn_state[t])
        
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- the next elements are derivative of c_1,h_1, ..., c_top, h_top
                drnn_state[t-1][k-1] = v
            end
        end
        if opt.skipConnect then 
            drnn_state[t-1][#dlst] = torch.zeros(opt.batchSz, opt.rnnSz):cuda()
        end

        dembedding_t = dlst[1]        
        clones.embed[t]:backward(x[{{}, {}, t}]:squeeze(), dembedding_t)
    end

    
    -- transfer final state to initial state (BPTT)
--    init_state_global = rnn_state[#rnn_state]  
            
    -- By default, MSECriterion is avarage over all elements. So the loss is normalized by batch size and 3*nJoint already
    -- We just need to normalize by the sequence length    
    loss = loss/(opt.seqLen - opt.seedLen) 
    grad_params:div(opt.seqLen - opt.seedLen)

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

-- mode = 0 use gt label to test
-- mode = 1 use average label to test
local function getLoss(Loader, mode)
    local loss = 0
    local nBatch2eval = math.min(Loader.nBatch, 10) -- maximum 10 batches
    for i=1,nBatch2eval do
        ------------------ get minibatch -------------------
        local x, y, labelBatch = Loader:next_batch() -- assume x is Tensor batchSz*nJntCoord*seqLen, y is shifted tensor by 1 time frame
        
        if mode == 1 then
            labelBatch = torch.Tensor(opt.batchSz, nClass):fill(1/nClass)
        end

        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
            labelBatch = labelBatch:float():cuda()
        end    

        ------------------- forward pass -------------------
        local rnn_state = init_state_global; -- internal cell states of LSTM
        protos.lstm:evaluate()           
        for t=1,opt.seqLen do   
            local embedding = protos.embed:forward{x[{{}, {}, t}]:squeeze(), labelBatch}
            local lst = protos.lstm:forward{embedding, unpack(rnn_state)}
            rnn_state = {}
            for i=1,#init_state_global do 
                table.insert(rnn_state, lst[i]) 
            end
            if t > opt.seedLen then
                local predPose = protos.invEmbed:forward(lst[#lst]) -- Use the last h to predict
                -- local means = protos.invEmbed1:forward(lst[#lst]) -- Use the last h to predict
                -- local var = protos.invEmbed2:forward(lst[#lst]) -- Use the last h to predict
                -- loss = loss + protos.criterion:forward(means, var, y[{{}, {}, t}]:squeeze())
                loss = loss + protos.criterion:forward(predPose, y[{{}, {}, t}]:squeeze())
            end
        end
    end    
    loss = loss/nBatch2eval/(opt.seqLen - opt.seedLen)  
    return loss
end

-- optimization stuff
local adagrad_optim_state = {learningRate = 1e-3}
local rmsprop_optim_state = {learningRate = 1e-3, alpha = 0.95}
local sgd_optim_state = {learningRate = 1e-3, learningRateDecay = 1e-7, momentum = 0.95, weightDecay = 1e-6 }
local adadelta_optim_state = {learningRate = 1e-3, decay = 1e-6, epsilon = 1e-7}
local nIter
if opt.maxEpoch >= 1 then
    nIter = opt.maxEpoch * trLoader.nBatch
else
    nIter = opt.nIter
end
print('Finish setting up. Start optimization now. The number of epoch is ' .. nIter/trLoader.nBatch..'. The number of iterations is '..nIter)

local optIter
local startTimer = torch.Timer()

local nIterEpoch = trLoader.nBatch
local trLosses 
local valLosses_gt
 local valLosses_avg
Log:setNames({'Iter', 'Epoch', 'loss', 'trLoss', 'valLoss(gt)', 'valLoss(avg)', 'time/iter','elpaseT'})
for optIter = iterInt, nIter do 
    local epoch = optIter / trLoader.nBatch
    local timer = torch.Timer()
    local loss
    if opt.optimization == 'adagrad' then
        _, loss = optim.adagrad(feval, params, adagrad_optim_state)
    elseif opt.optimization == 'rmsprop' then
        _, loss = optim.rmsprop(feval, params, rmsprop_optim_state)
    elseif opt.optimization == 'sgd' then
        _, loss = optim.sgd(feval, params, sgd_optim_state)
    elseif opt.optimization == 'adadelta' then
        _, loss = optim.adadelta(feval, params, adadelta_optim_state)          
    end
    local time = timer:time().real   
    
    if optIter % opt.printEvery == 0 then 
        trLosses = getLoss(valLoader, 0)
        valLosses_gt = getLoss(tstLoader, 0)
        valLosses_avg = getLoss(tstLoader, 1)
        local eTime = startTimer:time().real
        print(string.format("Iter:%d/%d, Epoch:%.1f, loss = %5.3e, trLoss = %5.3e, valLoss_gt = %5.3e, valLoss_avg = %5.3e,  gradnorm = %5.3e, time/iter = %3.0es, elpaseT: %.0fs",
            optIter, nIter, epoch, loss[1], trLosses, valLosses_gt, valLosses_avg, grad_params:norm(), time, eTime))
        --        Log:add{['Iter'] = optIter, ['Epoch']=epoch, ['trLoss']=trCLosses, ['trError']=trErrors, ['valLoss']=valCLosses,
        --                ['valError']=valErrors, ['time/iter']=time/nIterEpoch, ['elpaseT']=eTime }
        Log:add{optIter, epoch, loss[1], trLosses, valLosses_gt, valLosses_avg,
            time, eTime }        
    end
    if optIter % opt.saveEvery == 0 then  
        -- save model
        rnnNet.init_state = {}
        for i=1,#init_state_global do
            rnnNet.init_state[i] = init_state_global[i]
        end        
        -- torch.save(string.format('%s_%d_%d_%d_%s_d%.2f.t7',opt.modelFile,opt.nLayer,opt.rnnSz,opt.batchSz,opt.optimization, opt.dropout),
            -- rnnNet)

        local savefile = string.format('%s/%s_%d_%d_%d_%s_d%.2f_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.modelFile,
            opt.nLayer,opt.rnnSz,opt.batchSz,opt.optimization, opt.dropout, opt.skipConnect, epoch, valLosses_gt)

        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.optIter = optIter
        checkpoint.epoch = epoch
        checkpoint.iteration = optIter
        checkpoint.rnnNet = rnnNet
        torch.save(savefile, checkpoint)               
    end
end    
