require 'nn'
require 'nngraph'
local inputs = {}
table.insert(inputs,nn.Identity()())
table.insert(inputs,nn.Identity()())
table.insert(inputs,nn.Identity()())
local input = inputs[1]
local prev_c = inputs[2]
local prev_h = inputs[3]
local i2h = nn.Linear(input_size,4*rnn_size)(input)
local h2h = nn.Linear(rnn_size,4*rnn_size)(prev_h)
local preactivations = nn.CAddTable()({i2h,h2h})
