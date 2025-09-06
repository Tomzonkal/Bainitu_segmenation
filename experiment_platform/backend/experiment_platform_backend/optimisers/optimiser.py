from logger import MLFlowLogger
class Optimiser(MLFlowLogger):
    """
    A class to represent a generic optimiser.
    """

    def __init__(self,*args,**kwargs):
        """
        Initializes the Optimiser instance.
        """
        super().__init__(*args,**kwargs)

    def optimise(self):
        """
        Placeholder method for the optimisation process.

        This method should be implemented to perform the optimisation.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")
