"""Perform GPR Active Learning where the model is trained / ran on the local thread"""

from pipeline_prototype.method_server import MethodServer
from pipeline_prototype.redis.queue import MethodServerQueues, ClientQueues
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider
from parsl.config import Config
from threading import Thread
from datetime import datetime
from parsl import python_app
from random import uniform
import numpy as np
import argparse
import logging
import hashlib
import parsl
import json
import os


# The Thinker and Doer Classes
class Thinker(Thread):
    """Tool that monitors results of simulations and calls for new ones, as appropriate"""

    def __init__(self, queues: ClientQueues, n_parallel: int = 1, n_guesses: int = 10,
                out_path: str = None):
        """
        Args:
            n_guesses (int): Number of guesses the Thinker can make
            n_parallel (int): Maximum number of functional evaluations in parallel
            queues (ClientQueues): Queues for communicating with method server
            out_path (str): Path where to write result files
        """
        super().__init__()
        self.n_guesses = n_guesses
        self.n_parallel = n_parallel
        self.queues = queues
        self.logger = logging.getLogger(self.__class__.__name__)
        if out_path is not None:
            self.output_file = open(out_path, 'w')
        else:
            self.output_file = os.devnull

    def run(self):
        """Connects to the Redis queue with the results and pulls them"""

        # Define the sampling range
        sample_range = (-32.768, 32.768)

        # Make a random guess to start
        self.logger.info(f'Starting {self.n_parallel} random guesses')
        for _ in range(self.n_parallel):
            self.queues.send_inputs(uniform(*sample_range))
        self.logger.info('Submitted initial random guesses')
        train_X = []
        train_y = []

        # Use the initial guess to train a GPR
        gpr = GaussianProcessRegressor(normalize_y=True, kernel=kernels.RBF() * kernels.ConstantKernel())
        result = self.queues.get_result()
        print(result.json(), file=self.output_file)
        self.logger.info(f'Received result: result')
        train_X.append(result.args)
        train_y.append(result.value['result'])

        # Make guesses based on expected improvement
        for _ in range(self.n_guesses - 1):
            # Update the GPR with the available training data
            gpr.fit(train_X, train_y)

            # Generate a random assortment of potential next points to sample
            sample_X = np.random.uniform(*sample_range, size=(64, 1))

            # Compute the expected improvement for each point
            pred_y, pred_std = gpr.predict(sample_X, return_std=True)
            best_so_far = np.min(train_y)
            ei = (pred_y - best_so_far) / pred_std

            # Run the sample with the highest EI
            best_ei = sample_X[np.argmax(ei), 0]
            self.queues.send_inputs(best_ei)
            self.logger.info(f'Sent new guess based on EI: {best_ei}')

            # Wait for the value to complete
            result = self.queues.get_result()
            print(result.json(), file=self.output_file)
            self.logger.info('Received value')

            # Add the value to the training set for the GPR
            train_X.append([best_ei])
            train_y.append(result.value['result'])

if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--redishost", default="127.0.0.1",
                        help="Address at which the redis server can be reached")
    parser.add_argument("--redisport", default="6379",
                        help="Port on which redis is available")
    parser.add_argument("--n_guesses", default=10, type=int,
                        help="Total number of guesses to make")
    parser.add_argument("--parallel_guesses", default=1, type=int,
                        help="Number of calculations to maintain in parallel")
    parser.add_argument("--workers", default=1, type=int,
                        help="Number of workers processes to deploy for function evaluations.")
    parser.add_argument("--mean", default=None, type=float,
                        help="Mean time for tasks to sleep between before evaluation. None indicates"
                             " that no sleep will be performed. Units: s")

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = f'{start_time.strftime("%d%b%y-%H%M%S")}-{params_hash}'
    os.makedirs(out_dir, exist_ok=True)

    # Save the run parameters to disk
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)

    # Set up the logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, 
                        handlers=[
                            logging.FileHandler(os.path.join(out_dir, 'run.log')),
                            logging.StreamHandler()
                        ])

    # Make the target function, which includes sleeping and runtime
    @python_app(executors=['htex'])
    def target_function(x, mean_sleep=None):
        # Imports for the Python function
        from random import expovariate
        from time import sleep, perf_counter
        from pipeline_experiment.targets import ackley

        # Run the target function
        start_time = perf_counter()
        if mean_sleep is not None:
            sleep_time = expovariate(mean_sleep)
            sleep(sleep_time)
        result = ackley(x)
        end_time = perf_counter()

        # Return the output and the runtime
        return {'result': result[0], 'runtime': end_time - start_time}

    # Generate the method server
    class Doer(MethodServer):
        """Class the manages running the function to be optimized"""

        def run_application(self, method, *args, **kwargs):
            return target_function(*args)

    # Write the configuration
    config = Config(
        executors=[
            HighThroughputExecutor(
                address="localhost",
                label="htex",
                max_workers=2,
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
            ),
            ThreadPoolExecutor(label="local_threads", max_threads=4)
        ],
        strategy=None,
        run_dir=os.path.join(out_dir, 'run-info')
    )
    parsl.load(config)

    # Connect to the redis server
    client_queues = ClientQueues(args.redishost, args.redisport)
    server_queues = MethodServerQueues(args.redishost, args.redisport)

    # Create the method server and task generator
    doer = Doer(server_queues)
    thinker = Thinker(client_queues, 
                      n_guesses=args.n_guesses,
                      n_parallel=args.parallel_guesses,
                      out_path=os.path.join(out_dir, 'results.json'))
    logging.info('Created the method server and task generator')

    try:
        # Launch the servers
        #  The method server is a Thread, so that it can access the Parsl DFK
        #  The task generator is a Thread, so that all debugging methods get cast to screen
        doer.start()
        thinker.start()
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        client_queues.send_kill_signal()

    # Wait for the method server to complete
    doer.join()
