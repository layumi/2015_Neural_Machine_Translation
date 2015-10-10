
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local WordSplitLMMinibatchLoader = {}
WordSplitLMMinibatchLoader.__index = WordSplitLMMinibatchLoader

function WordSplitLMMinibatchLoader.create(data_dir, batch_size, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, WordSplitLMMinibatchLoader)

    local input_file = path.join(data_dir, 'en.txt')
    local output_file = path.join(data_dir, 'fr.txt')
    local vocab_en_file = path.join(data_dir, 'vocab_en.t7')
    local tensor_en_file = path.join(data_dir, 'data_en.t7')
    local vocab_fr_file = path.join(data_dir, 'vocab_fr.t7')
    local tensor_fr_file = path.join(data_dir, 'data_fr.t7')
    local map_en_file = path.join(data_dir, 'map_en.t7')
    local map_fr_file = path.join(data_dir, 'map_fr.t7')
    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_fr_file) or path.exists(vocab_en_file) or path.exists(tensor_en_file) or path.exists(tensor_fr_file)) then
        -- prepro files do not exist, generate them
        print('vocab_fr.t7 and data_fr.t7 and data_en.t7 and vocab_en.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local output_attr = lfs.attributes(output_file)
        local vocab_en_attr = lfs.attributes(vocab_en_file)
        local tensor_en_attr = lfs.attributes(tensor_en_file)
        local vocab_fr_attr = lfs.attributes(vocab_fr_file)
        local tensor_fr_attr = lfs.attributes(tensor_fr_file)
        if input_attr.modification > vocab_en_attr.modification or input_attr.modification > tensor_en_attr.modification then
            print('vocab_en.t7 or data_en.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
        if output_attr.modification > vocab_fr_attr.modification or output_attr.modification > tensor_fr_attr.modification then
            print('vocab_fr.t7 or data_fr.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    run_prepro = true  --force to rerun
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        en_batch,en_max_len=WordSplitLMMinibatchLoader.text_to_tensor(input_file, vocab_en_file, tensor_en_file, map_en_file)
        print('one-time setup: preprocessing output text file ' .. input_file .. '...')
        fr_batch,fr_max_len=WordSplitLMMinibatchLoader.text_to_tensor(output_file, vocab_fr_file, tensor_fr_file, map_fr_file)
    end
    self.ntrain = en_batch -- en_batch and fr_batch should be same
    self.num_layers1 = en_max_len
    self.num_layers2 = fr_max_len
    print('loading data files...')
    local data_en = torch.load(tensor_en_file)
    self.vocab_mapping_en = torch.load(vocab_en_file)
    local data_fr = torch.load(tensor_fr_file)
    self.vocab_mapping_fr = torch.load(vocab_fr_file)

    -- count vocab
    self.vocab_size_en = 0
    for _ in pairs(self.vocab_mapping_en) do 
        self.vocab_size_en = self.vocab_size_en + 1 
    end
    print('self.vocab_size_en',self.vocab_size_en)
    self.vocab_size_fr = 0
    for _ in pairs(self.vocab_mapping_fr) do 
        self.vocab_size_fr = self.vocab_size_fr + 1 
    end
    print('self.vocab_size_fr',self.vocab_size_fr)
    print('reshaping tensor...')
    self.batch_size = batch_size

    local ydata = data_fr:clone()
    ydata:sub(1,-2):copy(data_fr:sub(2,-1))
    ydata[-1] = data_fr[1]
    self.x1_batches = data_en:view(batch_size, -1):split(en_max_len, 2)  -- #rows = #batches
    self.nbatches = #self.x1_batches
    self.x2_batches = data_fr:view(batch_size, -1):split(fr_max_len, 2)  -- #rows = #batches
    self.y_batches = data_fr:view(batch_size, -1):split(fr_max_len, 2)  --

-- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))

    collectgarbage()
    return self
end

function WordSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function WordSplitLMMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x1_batches[ix], self.x2_batches[ix], self.y_batches[ix]
end

-- *** STATIC method ***
function WordSplitLMMinibatchLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile,out_mapfile)
    local timer = torch.Timer()
    local maxlen = 10000  --max word range
    print('loading text file...')
    local rawdata
    local tot_len = 0
    print(in_textfile)
    f = io.open(in_textfile, "r")

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all words(char) to a set
    local unordered = {}
    rawdata = f:read('*l') --read line
    local word_counter=0
    local max_str_len=0
    local sentence_counter=0
    repeat
        word_counter=0
        for word in rawdata:gmatch'.- ' do
            if not unordered[word] then unordered[word] = true end
            word_counter = word_counter+1
        end
        if word_counter>max_str_len then max_str_len=word_counter end
        sentence_counter = sentence_counter+1
        tot_len = tot_len + word_counter
        rawdata = f:read('*l') --read line
    until not rawdata
    f:close()
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for word in pairs(unordered) do 
          ordered[#ordered + 1] = word
          if #ordered == maxlen then
              break
          end 
    end
    table.sort(ordered)
    -- invert `ordered` to create the word->int mapping
    local vocab_mapping = {}
    for i, word in ipairs(ordered) do
        vocab_mapping[word] = i+1   -- 1 for low frenquency or not exist word
    end
    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = torch.IntTensor(sentence_counter,max_str_len+1):zero()+1 -- store it into 2D first, then rearrange
    local map = torch.IntTensor(sentence_counter,max_str_len+1) :zero()+1
    print('how many sentences',sentence_counter)
    print('max_sen_lens',max_str_len)
    f = io.open(in_textfile, "r")
    local currlen = 1 --sentence counter
    local counter = 1
    rawdata = f:read('*l') --read line
    repeat
        counter = 1
        for word in rawdata:gmatch'.- ' do
            --print (char,vocab_mapping[char])
            if vocab_mapping[word]~=nil then
                data[currlen][counter] = vocab_mapping[word] -- lua has no string indexing using []
            end
            map[currlen][counter]=1
            counter = counter + 1
        end
        currlen = currlen + 1
        rawdata = f:read('*l')
    until not rawdata
    f:close()

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
    print('saving ' .. out_mapfile)
    torch.save(out_mapfile, map)
    return sentence_counter,max_str_len+1
end

return WordSplitLMMinibatchLoader

