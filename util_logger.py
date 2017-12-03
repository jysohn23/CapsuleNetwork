import io

class custom_logger:
    def __init__(self,file_name,):
        self.fid_loss = io.open(file_name+'_loss.txt','ab')
        self.fid_param = io.open(file_name+'_param.txt','ab')

    def print_loss(self,input_str):
        self.fid_loss.writelines(input_str)

    def print_paramn(self,param):
        self.fid_param.writelines(param)

    def close_all(self):
        self.fid_loss.close()
        self.fid_param.close()