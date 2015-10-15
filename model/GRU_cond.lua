
local GRU_cond = {}

<<<<<<< HEAD
function GRU_cond.gru(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  local inputs = {}
  --gru-cond
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- mask
=======
--[[
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
function GRU_cond.gru(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  --gru-cond
  table.insert(inputs, nn.Identity()()) -- x
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
  table.insert(inputs, nn.Identity()()) -- context
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  function new_input_sum2(insize, xv, hv, c) -- wx+v*_h
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    local c2h = nn.Linear(rnn_size, rnn_size)(c)  --should be modified later
    return nn.CAddTable()({i2h, h2h ,c2h})
  end

  local x, input_size_L
  local outputs = {}

  for L = 1,n do
<<<<<<< HEAD
    local prev_h = inputs[L+3]
    local context = inputs[3]
    local mask = inputs[2]
=======
    local prev_h = inputs[L+2]
    local context = inputs[L+1]
>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
    -- the input to this layer
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
    -- c_size pre rnn_size
    local update_gate = nn.Sigmoid()(new_input_sum2(input_size_L, x, prev_h, context))
    local reset_gate = nn.Sigmoid()(new_input_sum2(input_size_L, x, prev_h,context))
    -- reset_gate = w*x+v*_h+w*c
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    -- gated_hidden = r*_h 
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    -- p2 = w*r*_h+b
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    -- p1 = w*x+b
    local p3 = nn.Linear(rnn_size, rnn_size)(context)
    -- p3 = w*c+b
    local p12 = nn.CAddTable()({p1,p2})
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p12,p3}))
    -- h~ = tanh(p2+p1+p3)  
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
<<<<<<< HEAD
    local next_h_w = nn.CAddTable()({zh, zhm1})
    local h1 = nn.CMulTable()({mask, next_h_w})
    local h2 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(mask)), prev_h})
    local next_h = nn.CAddTable()({h1,h2})
=======
    local next_h = nn.CAddTable()({zh, zhm1})

>>>>>>> c11a875ca3a70aa7c3030ab2f71f08b685e0e5bc
    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return GRU_cond
