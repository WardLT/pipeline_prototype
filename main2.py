import argparse
from multiprocessing import Queue
import redis
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

from redis_q import RedisQueue

import mpi_method_server2
import value_server

config_mac = Config(
    executors=[
        ThreadPoolExecutor(label="theta_mpi_launcher"),
        ThreadPoolExecutor(label="local_threads")
    ],
    strategy=None,
)

config = Config(
    executors=[
        HighThroughputExecutor(
            label="theta_mpi_launcher",
            # Max workers limits the concurrency exposed via mom node
            max_workers=2,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
        ),
        ThreadPoolExecutor(label="local_threads")
    ],
    strategy=None,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--redishost", default="127.0.0.1",
                        help="Address at which the redis server can be reached")
    parser.add_argument("--redisport", default="6379",
                        help="Port on which redis is available")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Count of apps to launch")
    parser.add_argument("-m", "--mac", action='store_true',
                        help="Configure for Mac")
    args = parser.parse_args()

    if args.debug:
        parsl.set_stream_logger()

    if args.mac:
        parsl.load(config_mac)
    else:
        parsl.load(config)

    print('''This program creates an "MPI Method Server" that listens on an input queue and write on an output queue:

        input_queue --> mpi_method_server --> output_queue

To send it a request, add an entry to the input queue:
     run "python3 pump.py -i -p N" where N is an integer request
To access a result, remove it from the outout queue:
     run "python3 pull.py" (blocking) or "python3 pull.py -t T" (T an integer) to time out after T seconds
''')

    # input_queue --> mpi_method_server --> output_queue

    input_queue = RedisQueue(args.redishost, port=int(args.redisport), prefix='input')
    input_queue.connect()

    value_server = value_server.ValueServer(args.redishost, port=int(args.redisport), database=1)

    mms = mpi_method_server2.MpiMethodServer(input_queue, value_server)
    mms.main_loop()

    keys = value_server.all_keys()
    print('All keys', keys)

    value_server.flush()

    print("All done")
