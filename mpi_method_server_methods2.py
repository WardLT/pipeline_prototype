import parsl
from parsl import python_app, bash_app
from parsl.executors import ThreadPoolExecutor
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.config import Config
from parsl.data_provider.files import File
from concurrent.futures import Future

# Simulate will run some commands on bash, this can be made to run MPI applications via aprun
# Here, simulate will put a random number from range(0-32767) into the output file.
# ************ NOTE: Seems that a bash_app cannot take a "self", so I need to remove
@bash_app(executors=['theta_mpi_launcher'])
def simulate(params, delay=1, outputs=[], stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    return f'''sleep {delay};
    echo "Running at ", $PWD
    echo "Running some serious MPI application"
    set -x
    echo "aprun mpi_application {params} -o {outputs[0]}"
    echo $RANDOM > {outputs[0]}
    '''

# Output the param and output kv pair on the output queue.
# This app runs on the Parsl local side on threads.
#@python_app(executors=['local_threads'])
@bash_app(executors=['theta_mpi_launcher'])
def output_result(value_server, param, inputs=[]):
    with open(inputs[0]) as f:
        simulated_output = int(f.readline().strip())
        print(f"Outputting {param} : {simulated_output}")
        value_server.set(param, simulated_output)
    return param, simulated_output

@bash_app(executors=['theta_mpi_launcher'])
def NWChem_app(params):
    return f'aprun NWChem {params}'

methods_list = [NWChem_app, output_result, simulate]
