from multiprocessing import Queue
import time
import os

import parsl
from parsl import python_app, bash_app
from parsl.executors import ThreadPoolExecutor
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.config import Config
from parsl.data_provider.files import File
from concurrent.futures import Future

import mpi_method_server_methods2

class MpiMethodServer:

    """ If load_default = True, we load the default methods list from a separate methods file
        else the user must pass a list of methods via methods_list kwarg.
    """
    def __init__(self, input_queue, value_server, methods_list=None, load_default=True):
        self.input_queue   = input_queue
        self.value_server  = value_server
        self.task_list     = []
        self.methods_table = {} # Dict maps {func_name : func}

        if load_default is True:
            for method in mpi_method_server_methods2.methods_list:
                self.add_method(method)
        else:
            for method in methods_list:
                self.add_method(method)  

    # Python function's name can be accessed as a string via __name__
    def add_method(self, method):
        self.methods_table[method.__name__] = method

    def launch_method(self, method_name, *args, **kwargs):
        if method_name in self.methods_table:
            val = self.methods_table[method_name](*args, **kwargs)
            return val
        else:
            print(f"Requested method : {method_name} is not loaded")

    # We listen on a Python multiprocessing Queue as an example
    # we launch the application with the params that arrive over this queue
    # Listen on the input queue for params, run a task for the param, and output the result on output queue
    @python_app(executors=['local_threads'])
    def listen_and_launch(self):
        while True:
            param = self.input_queue.get()
            # Putting None onto queue seems to yield 'null'
            if param is None or param=='null':
                break
            future = self.run_application(param)
            self.task_list.append(future)
        return self.task_list


    def make_outdir(self, path):
        # Make outputs directory if it does not already exist
        if not os.path.exists(path):
            os.makedirs(path)


    # Calls a function (remotely) and add result to output queue
    def run_application(self, i):
        print(f"Run_application called with {i}")
        outdir = 'outputs'
        self.make_outdir(outdir)
        x = self.launch_method('simulate', i, delay=1 + int(i) % 2, outputs=[File(f'{outdir}/simulate_{i}.out')])
        y = self.launch_method('output_result', self.value_server, i, inputs=[x.outputs[0]])
        return y


    def main_loop(self):
        m = self.listen_and_launch(self)
        print(m.result())
        for task in self.task_list:
            current = task
            print('Task:',current)
            while True:
                x = current.result()
                if isinstance(x, Future):
                    current = x
                else:
                    break
        dfk = parsl.dfk()
        dfk.wait_for_current_tasks()
