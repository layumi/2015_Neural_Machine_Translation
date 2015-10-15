require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local WordSplitLMMinibatchLoader = require 'util.WordSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local GRU_cond = require 'model.GRU_cond'
local RNN = require 'model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
<<<<<<< HEAD
cmd:option('-rnn_size', 256, 'size of LSTM internal state')
=======
cmd:option('-rnn_size', 1024, 'size of LSTM internal state')
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'gru_cond', 'lstm,gru,rnn or gru_cond')
-- optimization
cmd:option('-learning_rate',1e-4,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','gru_cond','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',3,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = WordSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, split_sizes)
local vocab_size_en = loader.vocab_size_en  -- the number of distinct characters
local vocab_size_fr = loader.vocab_size_fr
opt.seq_length1 = loader.num_layers1
opt.seq_length2 = loader.num_layers2
local vocab = loader.vocab_mapping
print('vocab size en: ' .. vocab_size_en)
print('vocab size fr: ' .. vocab_size_fr)

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    for c,i in pairs(checkpoint.vocab) do 
        if not vocab[c] == i then 
            vocab_compatible = false
        end
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'gru_cond' then
        protos.rnn = GRU.gru(vocab_size_en, opt.rnn_size, opt.num_layers,opt.dropout) 
<<<<<<< HEAD
        protos.rnn_cond = GRU_cond.gru(vocab_size_fr, opt.rnn_size, opt.num_layers, opt.dropout) 
=======
        protos.rnn_cond = GRU_cond.gru(vocab_size_fr, opt.rnn_size, opt.num_layers, opt.dropout)  -- my model rnn_size=1024
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
        print('set up model successively')
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    end
    protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size) --batch_size=1
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
<<<<<<< HEAD
params, grad_params = model_utils.combine_all_parameters(protos.rnn,protos.rnn_cond)
=======
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

<<<<<<< HEAD
print('number of parameters in the gru model: ' .. params:nElement())

=======
print('number of parameters in the model: ' .. params:nElement())
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
--

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print(name)
    if name=='rnn' then 
       print('cloning ' .. 'encoder')
       clones[name] = model_utils.clone_many_times(proto, opt.seq_length1, not proto.parameters)
    else
      print('cloning ' .. 'decoder')
      clones[name] = model_utils.clone_many_times(proto, opt.seq_length2, not proto.parameters)
    end
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
<<<<<<< HEAD
        local x1, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x1 = x1:float():cuda()
            y = y:float():cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            x1 = x1:cl()
=======
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            x = x:cl()
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
            y = y:cl()
        end
        -- forward pass
        for t=1,opt.seq_length1 do
            clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
<<<<<<< HEAD
            local lst = clones.rnn[t]:forward{x1[{{}, t}],mask1, unpack(rnn_state[t-1])} 
=======
            local lst = clones.rnn[t]:forward{x1[{{}, t}], unpack(rnn_state[t-1])} 
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
            context = lst
        end
<<<<<<< HEAD
        --print(context)
=======
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
        for t=1,opt.seq_length2 do
            clones.rnn_cond[t]:training() 
            if t==1 then 
                tmp=1 
            else tmp = predictions[t-1]
            end
<<<<<<< HEAD
            local lst2 = clones.rnn_cond[t]:forward{tmp,context,mask2,unpack(rnn_state_cond[t-1])}
=======
            local lst2 = clones.rnn_cond[t]:forward{tmp,context, unpack(rnn_state_cond[t-1])}
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
            rnn_state_cond[t] = {}
            for i=1,#init_state do table.insert(rnn_state_cond[t], lst2[i]) end -- extract the state, without output
            predictions[t] = lst2[#lst2] -- last element is the prediction
            loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
        end
        -- carry over lstm state
        --rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length2 / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
local init_state_global_cond = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
<<<<<<< HEAD
    local x1,x2,mask1,mask2,y = loader:next_batch(1)
=======
    local x1,x2, y = loader:next_batch(1)
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x1 = x1:float():cuda()
        --x2 = x2:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x1 = x1:cl()
        --x2 = x2:cl()
        y = y:cl()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local rnn_state_cond = {[0] = init_state_global_cond}
    local predictions = {}           -- softmax outputs
    local loss = 0
<<<<<<< HEAD
    mask1_t={}
    mask2_t={}
    predict_word={}
    for t=1,opt.seq_length1 do
        if mask1[1][t]==1 then 
                tmp_mask = torch.DoubleTensor(1,opt.rnn_size):zero()+1 
        else tmp_mask = torch.DoubleTensor(1,opt.rnn_size):zero()
        end
        table.insert(mask1_t,tmp_mask)
        clones.rnn[t]:training() 
        local lst = clones.rnn[t]:forward{x1[{{}, t}],mask1_t[t], unpack(rnn_state[t-1])} 
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end 
        context = lst
    end
    
    print('finish encoding!')
    for t=1,opt.seq_length2 do
        clones.rnn_cond[t]:training() 
        if t==1 then 
                tmp = torch.IntTensor(1):zero()+1 
        else tmp = y[{{}, t-1}]
        end
        if mask2[1][t]==1 then 
                tmp_mask = torch.DoubleTensor(1,opt.rnn_size):zero()+1 
        else tmp_mask = torch.DoubleTensor(1,opt.rnn_size):zero()
        end
        table.insert(mask2_t,tmp_mask)
        table.insert(predict_word,tmp)
        local lst2 = clones.rnn_cond[t]:forward{predict_word[t],mask2_t[t],context,unpack(rnn_state_cond[t-1])}
        rnn_state_cond[t] = {}
        for i=1,#init_state do table.insert(rnn_state_cond[t], lst2[i]) end
        predictions[t] = lst2[#lst2] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])*mask2[1][t]
    end
    print('finish decoding!')
    loss = loss / opt.seq_length2
    print(loss)
    ------------------ backward pass -------------------
    local drnn_state2 = {[opt.seq_length2] = clone_list(init_state, true)} -- true also zeros the clones
    local drnn_state1 = {[opt.seq_length1] = clone_list(init_state, true)} 
    for t=opt.seq_length2,1,-1 do
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state2[t], doutput_t)
        --print('drnn',drnn_state2[t])
        local dlst2 = clones.rnn_cond[t]:backward({predict_word[t],mask2_t[t],context,unpack(rnn_state_cond[t-1])}, drnn_state2[t])
        drnn_state2[t-1] = {}
        for k,v in pairs(dlst2) do
            if k > 1 then -- mask 1  context 2
=======
    for t=1,opt.seq_length1 do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x1[{{}, t}], unpack(rnn_state[t-1])} 
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        context = rnn_state[t]
    end
    --print(context)
    print('finish encoding!')
    for t=1,opt.seq_length2 do
        clones.rnn_cond[t]:training() 
        --if t==1 then 
                --tmp = torch.IntTensor(1):zero()+1 
       -- else tmp = prev_word
            -- print('prev_word:',prev_word)
       -- end
        print('hehe')
        local lst2 = clones.rnn_cond[t]:forward{torch.IntTensor(1):zero()+1 ,context,unpack(rnn_state_cond[t-1])}
        rnn_state_cond[t] = {}
        for i=1,#init_state do table.insert(rnn_state_cond[t], lst2[i]) end -- extract the state, without output
        predictions[t] = lst2[#lst2] -- last element is the prediction
        --print(predictions[t])
        local probs = torch.exp(predictions[t]):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_word = torch.multinomial(probs:float(), 1):resize(1):float()
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    print('finish decoding!')
    loss = loss / opt.seq_length2
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state2 = {[opt.seq_length2] = clone_list(init_state, true)} -- true also zeros the clones
    local drnn_state1 = {[opt.seq_length1] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length2,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state2[t], doutput_t)
        print("haha")
        local dlst = clones.rnn_cond[t]:backward({x1[{{}, t}]}, drnn_state2[t])
        drnn_state2[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- k==2 is gradient on context
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
                drnn_state2[t-1][k-1] = v
            end
        end
    end
<<<<<<< HEAD
    --graph.dot(clones.rnn_cond.fg, 'rnn_cond')
    print('finish bp decoding!')
   for t=opt.seq_length2,1,-1 do
        local flag=1
        for tt=opt.seq_length1,1,-1 do
          local dlst1 = clones.rnn[tt]:backward({x1[{{}, tt}],mask1_t[tt],nn.Reshape(1,opt.rnn_size):forward(unpack(rnn_state[tt-1]))}, drnn_state2[t-1][2])  -- transpose rnn_state
        end
    end
    print('Congratulation! Finish bp encoding!')
=======
    print('finish bp decoding!')
--[[
    for t=opt.seq_length1,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state1[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x1[{{}, t}], unpack(rnn_state[t-1])}, drnn_state1[t])
        drnn_state1[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state1[t-1][k-1] = v
            end
        end
    end
    print('finish bp encoding!')
--]]
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    init_state_global_cond = rnn_state_cond[#rnn_state_cond]
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
<<<<<<< HEAD
    if loss[1] > loss0 * 100 then
=======
    if loss[1] > loss0 * 3 then
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
        print('loss is exploding, aborting.')
        break -- halt
    end
end

