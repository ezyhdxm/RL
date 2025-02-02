import abc # Abstract Base Class
import numpy as np

class BasePolicy(object, metaclass=abc.ABCMeta):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get a single action for the input observation
        """
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs):
        """
        Return a dictionary of logging information
        """
        raise NotImplementedError
    
    def save(self, filepath):
        """
        Save the policy to a file
        """
        raise NotImplementedError