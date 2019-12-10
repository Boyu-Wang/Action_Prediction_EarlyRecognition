-- LSTM architecture with skip connections: input connect to all hidden state and all hidden state to output
-- modified based on MLA_LSTM.lua
-- By: Boyu Wang (boywang@cs.stonybrook.edu)



local LSTM = {}
-- according to RECURRENT NEURAL NETWORK REGULARIZATION (https://arxiv.org/abs/1409.2329)
-- dropout is only applied on vertical connections (from one layer to another layer)
-- not on horizontal connections (within recurrent layer)
-- 
function LSTM.lstm(rnnSz, n, dropout, skipConnect, isBN, forgetBias)        
        
    dropout = dropout or 0
    skipConnect = skipConnect or true
    isBN = isBN or false
    forgetBias = forgetBias or 1
    
    -- there will be 2*n+1 inputs
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- embedding of x
    for L = 1,n do
        table.insert(inputs, nn.Identity()()) -- prev_c[L]
        table.insert(inputs, nn.Identity()()) -- prev_h[L]
    end

    local x
    local outputs = {}
    for L = 1,n do
        -- c,h from previos timesteps
        local prev_h = inputs[L*2+1]
        local prev_c = inputs[L*2]
        -- the input to this layer
        if L == 1 then 
            x = inputs[1] -- input 
        else 
            if skipConnect then
                x = nn.JoinTable(2)({inputs[1], outputs[(L-1)*2]}) -- concatenation of input and h value from lower level                
            else
                x = outputs[(L-1)*2],2 -- h value from lower level
            end
            if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
        end
        -- evaluate the input sums at once for efficiency
        local i2h
        if not skipConnect or L == 1 then
            i2h = nn.Linear(rnnSz, 4 * rnnSz)(x)
        else
            i2h = nn.Linear(2*rnnSz, 4 * rnnSz)(x)
        end
        local h2h = nn.Linear(rnnSz, 4 * rnnSz)(prev_h)
        if dropout > 0 then
            i2h = nn.Dropout(dropout)(i2h)
--            h2h = nn.Dropout(dropout)(h2h)
        end
        if isBN then
            i2h = nn.BatchNormalization(4 * rnnSz, 1e-5, 0.1, true)(i2h)
            h2h = nn.BatchNormalization(4 * rnnSz, 1e-5, 0.1, true)(h2h)
        end
        
        local all_input_sums = nn.CAddTable()({i2h, h2h})
        -- decode the gates
--        local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnnSz)(all_input_sums)
--        sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
        local in_gate = nn.Narrow(2, 1, rnnSz)(all_input_sums)
        in_gate = nn.Sigmoid()(in_gate)
        
        -- forget bias is set to 1, to make sure forget gate is always open
        local forget_gate = nn.Narrow(2, rnnSz + 1, rnnSz)(all_input_sums)
        forget_gate = nn.Sigmoid()(nn.AddConstant(forgetBias)(forget_gate))
        local out_gate = nn.Narrow(2, 2 * rnnSz + 1, rnnSz)(all_input_sums)
        out_gate = nn.Sigmoid()(out_gate)
        -- decode the write inputs
        local in_transform = nn.Narrow(2, 3 * rnnSz + 1, rnnSz)(all_input_sums)
        in_transform = nn.Tanh()(in_transform)
        -- perform the LSTM update
        local next_c           = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate,     in_transform})
        })
        
        -- seems to have faster convergence rate without this normalization, if you have skip connection
--        if isBN then
--            next_c = nn.BatchNormalization(rnnSz, 1e-5, 0.1, true)(next_c)
--        end
        -- gated cells form the output
        local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})        
        
        table.insert(outputs, next_c)
        table.insert(outputs, next_h)
    end
    
    -- connect all hidden states to final output
    if skipConnect then
        local allH = {}
        for L = 1,n do
            table.insert(allH, outputs[2*L])
        end
        local finalOut
        if n > 1 then
            finalOut = nn.Linear(n*rnnSz, rnnSz)(nn.JoinTable(2)(allH))
        else
            finalOut = nn.Linear(n*rnnSz, rnnSz)(outputs[2])
        end
        table.insert(outputs, finalOut) 
    end
   
    return nn.gModule(inputs, outputs)
end

return LSTM

