import logging
import time
import uuid
from queue import PriorityQueue
from typing import List

from kubeflow.pytorchjob import PyTorchJobClient
from kubeflow.pytorchjob.constants.constants import PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, PYTORCHJOB_PLURAL
from kubernetes import client

from fltk.util.cluster.client import construct_job, ClusterManager
from fltk.util.config.base_config import BareConfig
from fltk.util.task.generator.arrival_generator import ArrivalGenerator, Arrival
from fltk.util.task.task import ArrivalTask


class Orchestrator(object):
    """
    Central component of the Federated Learning System: The Orchestrator

    The Orchestrator is in charge of the following tasks:
    - Running experiments
        - Creating and/or managing tasks
        - Keep track of progress (pending/started/failed/completed)
    - Keep track of timing

    Note that the Orchestrator does not function like a Federator, in the sense that it keeps a central model, performs
    aggregations and keeps track of Clients. For this, the KubeFlow PyTorch-Operator is used to deploy a train task as
    a V1PyTorchJob, which automatically generates the required setup in the cluster. In addition, this allows more Jobs
    to be scheduled, than that there are resources, as such, letting the Kubernetes Scheduler let decide when to run
    which containers where.
    """
    _alive = False
    # Priority queue, requires an orderable object, otherwise a Tuple[int, Any] can be used to insert.
    pending_tasks: "PriorityQueue[ArrivalTask]" = PriorityQueue()
    deployed_tasks: List[ArrivalTask] = []
    completed_tasks: List[str] = []

    def __init__(self, cluster_mgr: ClusterManager, arv_gen: ArrivalGenerator, config: BareConfig):
        self.__logger = logging.getLogger('Orchestrator')
        self.__logger.debug("Loading in-cluster configuration")
        self.__cluster_mgr = cluster_mgr
        self.__arrival_generator = arv_gen
        self._config = config

        # API to interact with the cluster.
        self.__client = PyTorchJobClient()

    def stop(self) -> None:
        """
        Stop the Orchestrator.
        @return:
        @rtype:
        """
        self.__logger.info("Received stop signal for the Orchestrator.")
        self._alive = False

    def run(self, clear: bool = True) -> None:
        """
        Main loop of the Orchestartor.
        @param clear: Boolean indicating whether a previous deployment needs to be cleaned up (i.e. lingering jobs that
        were deployed by the previous run).

        @type clear: bool
        @return: None
        @rtype: None
        """
        
        num_cores = [1, 1] 
        num_nodes = [1, 1]
        epochs = [2, 2]
        learning_rate = ['0.010', '0.005']
        batch_size = [256, 64]

        self._alive = True
        start_time = time.time()
        if clear:
            self.__clear_jobs()
        while self._alive and time.time() - start_time < self._config.get_duration():
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            #while not self.__arrival_generator.arrivals.empty():
#                task = ArrivalTask(priority=arrival.get_priority(),
#                                   id=unique_identifier,
#                                   network=arrival.get_network(),
#                                   dataset=arrival.get_dataset(),
#                                   sys_conf=arrival.get_system_config(),
#                                   param_conf=arrival.get_parameter_config())
#                self.__logger.debug(f"Arrival of: {task}")
#                self.pending_tasks.put(task)
            while (self.__arrival_generator.arrivals.empty()): 
                #do_nothing
                time.sleep(1)
            arrival: Arrival = self.__arrival_generator.arrivals.get()
            unique_identifier: uuid.UUID = uuid.uuid4()
            curr_priority = 0

            for epoch in epochs:
                for nodes in num_nodes:
                    unique_identifier: uuid.UUID = uuid.uuid4()
                    task = ArrivalTask(priority=curr_priority,
                                       id=unique_identifier,
                                       network=arrival.get_network(),
                                       dataset=arrival.get_dataset(),
                                       sys_conf=arrival.get_system_config(),
                                       param_conf=arrival.get_parameter_config())
                    task.param_conf.maxEpoch = epoch
                    task.sys_conf.dataParallelism = nodes
                    curr_priority += 1
                    self.pending_tasks.put(task)
                    self.__logger.info(f"Deploying task with : p: {task.priority}, id: {task.id}, max_epochs: {task.param_conf.maxEpoch}, parallelism: {task.sys_conf.dataParallelism}")

#            for lr in learning_rate:
#                for bs in batch_size:
#                    unique_identifier: uuid.UUID = uuid.uuid4()
#                    task = ArrivalTask(priority=curr_priority,
#                                       id=unique_identifier,
#                                       network=arrival.get_network(),
#                                       dataset=arrival.get_dataset(),
#                                       sys_conf=arrival.get_system_config(),
#                                       param_conf=arrival.get_parameter_config())
#                    task.param_conf.batchSize = bs
#                    task.param_conf.learningRate = lr
#                    curr_priority += 1
#                    self.pending_tasks.put(task)
#                    self.__logger.info(f"Deploying task with : p: {task.priority}, id: {task.id}, lr: {task.param_conf.learningRate}, bs: {task.param_conf.batchSize}")


            while not self.pending_tasks.empty():
                # Do blocking request to priority queue
                curr_task = self.pending_tasks.get()
                self.__logger.info(f"Scheduling arrival of Arrival: {curr_task.id}")
                job_to_start = construct_job(self._config, curr_task)


                # Hack to overcome limitation of KubeFlow version (Made for older version of Kubernetes)
                self.__logger.info(f"Deploying on cluster: {curr_task.id}")
                self.__client.create(job_to_start, namespace=self._config.cluster_config.namespace)
                self.deployed_tasks.append(curr_task)

                time.sleep(10)
                job_name = self.__client.get(namespace='test')['items'][0]['metadata']['name']
                self.__logger.info("Job name : " + job_name)
                self.__client.wait_for_job(name=job_name, namespace='test')

#                while self.__client.get_job_status(job_name, namespace='test') != "Succeeded": 
                    # Do nothing
#                    time.sleep(1)

                # TODO: Extend this logic in your real project, this is only meant for demo purposes
                # For now we exit the thread after scheduling a single task.

                # self.stop()
                # return

            self.stop()
            return
            self.__logger.debug("Still alive...")
            time.sleep(5)

        logging.info(f'Experiment completed, currently does not support waiting.')

    def __clear_jobs(self):
        """
        Function to clear existing jobs in the environment (i.e. old experiments/tests)
        @return: None
        @rtype: None
        """
        namespace = self._config.cluster_config.namespace
        self.__logger.info(f'Clearing old jobs in current namespace: {namespace}')

        for job in self.__client.get(namespace=self._config.cluster_config.namespace)['items']:
            job_name = job['metadata']['name']
            self.__logger.info(f'Deleting: {job_name}')
            try:
                self.__client.custom_api.delete_namespaced_custom_object(
                    PYTORCHJOB_GROUP,
                    PYTORCHJOB_VERSION,
                    namespace,
                    PYTORCHJOB_PLURAL,
                    job_name)
            except Exception as e:
                self.__logger.warning(f'Could not delete: {job_name}')
                print(e)
