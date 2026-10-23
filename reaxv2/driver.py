import ctypes as C

class NativeInt8:
    def __init__(self,binary,configs=None):
        self.configs = {} if configs is None else configs
        self.cuda=C.CDLL('libcuda.so.1')
        self.image=C.create_string_buffer(binary.read_bytes())
        self.module=C.c_void_p()
        self.check(self.cuda.cuModuleLoadData(C.byref(self.module),self.image))
        self.functions={}
    @staticmethod
    def check(status):
        if status:raise RuntimeError('CUDA driver status '+str(status))
