flags = ['-fPIC', '-std=c++0x', '-Wall', '-pthread', '-ggdb', '-O2', '-I', '.', '-x', 'c++', '-I', './dmlc-core/include', '-I', './include', '-I', './mshadow', '-DDMLC_USE_CXX11=1', '-DMXNET_USE_CUDA=1']

def FlagsForFile(filename):
    return { 'flags' : flags, 'do_cache': True }
