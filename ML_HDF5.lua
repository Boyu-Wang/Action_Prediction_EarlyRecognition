-- Utility functions to save and load hdf5 files


require 'torch'
require 'math'
require 'hdf5'

ML_HDF5 = {}
ML_HDF5.__index = ML_HDF5


function ML_HDF5.load(dataFile)
    local myFile = hdf5.open(dataFile, 'r')
    local dims = myFile:read('/meta'):all()
    dims = dims:squeeze()
    local cp = torch.cumprod(dims)    
    local n = cp[-1]
    
    local cData = {}
    for i =1,n do
        cData[i] = myFile:read(string.format('/%d', i)):all()
    end
        
    myFile:close()
    return cData
end

function ML_HDF5.save(outFile, cData, dims)
    local myFile = hdf5.open(outFile, 'w')
    dims = dims:squeeze()
    local cp = torch.cumprod(dims)
    
    local n = cp[-1]
    
    if #cData ~= n then
        print("Error: the number of elements in cData must match dims")
        print(string.format("#cData: %d, n: %d\n", #cData, n))
        return
    end 
    
    myFile:write('/meta', dims)
    
    for i =1,n do
        myFile:write(string.format('/%d', i), cData[i])
    end
        
    myFile:close()    
end

return ML_HDF5
