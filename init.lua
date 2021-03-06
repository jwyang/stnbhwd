require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libstnlr'
if withCuda then
   require 'libcustnlr'
end

require('stnlr.AffineTransformMatrixGeneratorLR')
require('stnlr.AffineGridGeneratorBHWDLR')
require('stnlr.BilinearSamplerBHWDLR')

require('stnlr.test')

return nn
