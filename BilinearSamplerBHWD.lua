local BilinearSamplerBHWD, parent = torch.class('nn.BilinearSamplerBHWD', 'nn.Module')
--just a test
--[[
   BilinearSamplerBHWD() :
   BilinearSamplerBHWD:updateOutput({inputImages, grids})
   BilinearSamplerBHWD:updateGradInput({inputImages, grids}, gradOutput)

   BilinearSamplerBHWD will perform bilinear sampling of the input images according to the
   normalized coordinates provided in the grid. Output will be of same size as the grids,
   with as many features as the input images.

   - inputImages has to be in BHWD layout

   - grids have to be in BHWD layout, with dim(D)=2
   - grids contains, for each sample (first dim), the normalized coordinates of the output wrt the input sample
      - first coordinate is Y coordinate, second is X
      - normalized coordinates : (-1,-1) points to top left, (-1,1) points to top right
      - if the normalized coordinates fall outside of the image, then output will be filled with zeros
]]

function BilinearSamplerBHWD:__init()
   parent.__init(self)
   self.gradInput={}
end

function BilinearSamplerBHWD:check(input, gradOutput)
   local inputImages = input[1]
	 local grids = input[2]
   local masks = input[3]
   local canvas = input[4]

   assert(inputImages:isContiguous(), 'Input images have to be contiguous')
   assert(inputImages:nDimension()==4)
   assert(canvas:isContiguous(), 'Canvas images have to be contiguous')
   assert(canvas:nDimension()==4)
   assert(grids:nDimension()==4)
   assert(inputImages:size(1)==grids:size(1)) -- batch
   assert(inputImages:size(1)==canvas:size(1)) -- batch
   assert(grids:size(4)==2) -- coordinates

   if gradOutput then
      assert(grids:size(1)==gradOutput:size(1))
      assert(grids:size(2)==gradOutput:size(2))
      assert(grids:size(3)==gradOutput:size(3))
   end
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

-- Different from the original version, we have three inputs here:
-- input[1]: input images
-- input[2]: grid
-- input[3]: canvas images
function BilinearSamplerBHWD:updateOutput(input)
   local _inputImages = input[1]
   local _grids = input[2]
   local _masks = input[3]
   local _canvas = input[4]

   local inputImages, grids, masks, canvas
   if _inputImages:nDimension()==3 then
      inputImages = addOuterDim(_inputImages)
      grids = addOuterDim(_grids)
      masks = addOuterDim(_masks)
   else
      inputImages = _inputImages
      grids = _grids
      masks = _masks
      canvas = _canvas
   end

   local input = {inputImages, grids, masks, canvas}

   self:check(input)

   self.output:resize(inputImages:size(1), grids:size(2), grids:size(3), inputImages:size(4))

   -- Note: instead of set all value be zero, we copy the value from canvas
   self.output:copy(canvas)

   inputImages.nn.BilinearSamplerBHWD_updateOutput(self, inputImages, grids, masks, self.output)

   if _inputImages:nDimension()==3 then
      self.output=self.output:select(1,1)
   end

   return self.output
end

function BilinearSamplerBHWD:updateGradInput(_input, _gradOutput)
  local _inputImages = _input[1]
  local _grids = _input[2]
  local _masks = _input[3]
  local _canvas = _input[4]

  local inputImages, grids, masks, canvas, gradOutput
  if _inputImages:nDimension()==3 then
    inputImages = addOuterDim(_inputImages)
    grids = addOuterDim(_grids)
    masks = addOuterDim(_masks)
    gradOutput = addOuterDim(_gradOutput)
  else
    inputImages = _inputImages
    grids = _grids
    masks = _masks
    gradOutput = _gradOutput
    canvas = _canvas
  end

  local input = {inputImages, grids, masks, canvas}

  self:check(input, gradOutput)
	for i=1,#input do
    self.gradInput[i] = self.gradInput[i] or input[1].new()
    self.gradInput[i]:resizeAs(input[i]):zero()
  end

  local gradInputImages = self.gradInput[1]
  local gradGrids = self.gradInput[2]
  local gradMasks = self.gradInput[3]
  local gradCanvas = self.gradInput[4]

  inputImages.nn.BilinearSamplerBHWD_updateGradInput(self, inputImages, grids, masks, canvas, gradInputImages, gradGrids, gradMasks, gradCanvas, gradOutput)

  if _gradOutput:nDimension()==3 then
    self.gradInput[1]=self.gradInput[1]:select(1,1)
    self.gradInput[2]=self.gradInput[2]:select(1,1)
    self.gradInput[3]=self.gradInput[3]:select(1,1)
    self.gradInput[4]=self.gradInput[4]:select(1,1)
  end

  return self.gradInput
end
