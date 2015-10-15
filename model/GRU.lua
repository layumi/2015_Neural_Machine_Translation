
local GRU = {}

<<<<<<< HEAD
function GRU.gru(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 
  -- claim inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- mask
=======
--[[
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
function GRU.gru(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local outputs = {}
<<<<<<< HEAD
  local mask = inputs[2]
  for L = 1,n do
    local prev_h = inputs[L+2]
=======
  for L = 1,n do

    local prev_h = inputs[L+1]
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
    -- the input to this layer  embeding!!
    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
<<<<<<< HEAD
    local next_h_w = nn.CAddTable()({zh, zhm1})
    local h1 = nn.CMulTable()({mask, next_h_w})
    local h2 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(mask)), prev_h})
    local next_h = nn.CAddTable()({h1,h2})
    table.insert(outputs, next_h)
  end
-- set up the decoder
 -- local top_h = outputs[#outputs]
 -- if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
 -- local proj = nn.Linear(rnn_size, input_size)(top_h)
 -- local logsoft = nn.LogSoftMax()(proj)
 -- table.insert(outputs, logsoft)
  
=======
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

--  local proj = nn.Linear(rnn_size, input_size)(top_h)
--  local logsoft = nn.LogSoftMax()(proj)
--  table.insert(outputs, logsoft)

>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
  return nn.gModule(inputs, outputs)
end

return GRU
