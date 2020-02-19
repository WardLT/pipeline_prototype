# Proxy Application Experiments

This folder contains a set of experiments of different configurations of the `pipeline_prototype`
designed to explore the effects of architecture designs and optimization algorithm choices.

## Implementation Details

The target problem is fixed between all examples: minimizing the Ackley function.
The Ackley function is configurable eon a configurable function, where we can 
vary the difficulty of the problem by adjusting the dimensionality of the function.
The computational cost of the application is simulated using a `sleep` call within
the function.
"Computational cost" can be varied by adjusting the altering the probability
distribution from which sleep times are drawn.

The initial simulator for the computational resources available to the application will be 
mocked with a Parsl `HighThroughputExecutor` with as many worker processes as specified by the user.
The entire application will run on a single node for simplicity. 
The Ackley target function itself is inexpensive and should spend most of its time 
in a sleep state, which should make it possible to run many workers per CPU core. 

The target function and various utilities are contained within the `pipeline_experiment` module.
The module is installed by calling `pip install -e .` from this folder.

The proxy applications themselves are implemented as CLI applications.
All applications have options to control the target function difficulty, target function
cost and the size of the allocation of distributed computing resources.   
Each application will have different configuration settings, as appropriate.

### To Do List

Features which we should consider adding:

- Find out a good model for task completion times observed by Parsl
- Making the cost dependent on the input values 
  to the function, in case we seek to evaluate strategies which use
  the expected cost of a function evaluation.
- Adding a cheaper and noisier version of the function, 
  to assess &Delta;-learning or multifidelity strategies.
- Implement a truly distributed version of the application
- Implement a quantum chemistry structure optimization example

## Application Design

The full details of the applications will be included in README files 
stored in the root directory of each test.
In general, the applications will follow a Thinker/Doer model.
The "Thinker" processes are tasked with choosing which tasks for the Doer model 
to compute.  

Initial implementation of the applications will use at least two processes besides the
worker processes:

- `MethodServer` running in the main Python thread, which accepts
tasks requests and submits them to compute resources.
- `Thinker` running as a separate process, which makes task requests 
and receives results.

## Profiling Approach

Our focus will be to measure the performance of the optimization algorithm,  
the utilization of the distributed compute resources, and the walltime-to-solution
for each implementation.
We describe the strategies for assessing each characteristic below.
We will optionally run the application with detailed profiling with
use of `cProfile` to assess the computation.

### Optimization Algorithm Performance

We will measure the fitness of the result as a function of the the number of 
evaluations. 
This tracking will be enabled by using a `MethodServer` subclass which also saves
the function results to disk as they are completed.

### Resource Utilization

We will also measure how well the system is utilized by monitoring the queue
length of work on the ``MethodServer``.
The queue length will assess how well the "Thinker" processes are 
providing the ``MethodServer``(s) with suitable usable work.
The queue length will be monitored by logging messages that are written
as a task is recieved and a result sent back to the thinker
by the method server.

#### To Do List

- Does Parsl implement the ability to monitor the worker utilization?

### Time-To-Solution

We will also assess the wall time required for an application to perform 
the optimization. 
The walltime will be measured by (1) the total wall time consumed
by process and (2) the fitness of results as a function of wall-time.