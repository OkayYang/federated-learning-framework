import ray
import torch

@ray.remote
class RayClientActor:
    """
    Ray Actor wrapper for FL clients to enable distributed execution.
    """
    def __init__(self, client_class, *args, **kwargs):
        self.client = client_class(*args, **kwargs)
        self.client_id = self.client.client_id

    def local_train(self, *args, **kwargs):
        return self.client.local_train(*args, **kwargs)

    def local_evaluate(self, *args, **kwargs):
        return self.client.local_evaluate(*args, **kwargs)

    def global_evaluate(self, *args, **kwargs):
        return self.client.global_evaluate(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.client.get_weights(*args, **kwargs)

    def get_model_copy(self):
        return self.client.get_model_copy()
    
    def get_client_id(self):
        return self.client.client_id

    def set_progress_actor(self, progress_actor):
        self.client.set_progress_actor(progress_actor)
