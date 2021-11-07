import pickle
import os

class OutputData():
    def __init__(self, op_dir='../assets'):
        self.nn_os = os
        self.nn_os.chdir(op_dir)
        self.op_d = self.nn_os.getcwd()
        self.data_dict = self.get_data()
        self.file_list = self.get_file_list()
        # print(self.op_d)
        # print(self.file_handles)

    def get_data(self):
        dd = {}
        for fname in os.listdir(self.nn_os.getcwd()):
            # print(fname, type(fname))
            f = os.path.join(self.nn_os.getcwd(), fname)
            # print(f)
            if 'Output' in fname:
                if os.path.isfile(fname):
                    obj = open(fname, 'rb')
                    dic = pickle.load(obj)
                    # print(fname, dic)
                    dd[fname] = dic

        return dd

    def get_file_list(self):
        l = []
        for fname in os.listdir(self.nn_os.getcwd()):
            l.append(fname)

        return l