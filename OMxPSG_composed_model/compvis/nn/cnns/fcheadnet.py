# =============================================================================
# Network surgery for fining-tuning method 
# Sub-module of the module conv.nn
# =============================================================================

# Importing Libraries

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Creating the Class to generate the Head of the fully-connected layer

class FCHeadNet:
    """Fully Connected Head Network class. This class enable us to create \n
    the FC layer for transfer learning tasks. The FC head is putted on the top the Network.
    """
    @staticmethod
    def build(baseModel, classes, D):
        """Build FCHead function.
        Args:
            baseModel: the base model structure until the FC
            classes: int value with the number of classes
            D: int value the number of nodes for each layers in the FC.
        return headModel
        """
        # Initializing the head model to put in the top of base, then add a FC layer
        headModel = baseModel.output
        headModel = Flatten(name = "flatten")(headModel)
        headModel = Dense(D, activation=("relu"))(headModel)
        headModel = Dropout(0.5)(headModel)
        ##Adding the last layer
        if classes == 2:
           headModel = Dense(1, activation=("sigmoid"))(headModel)
           
        else:
            headModel = Dense(classes, activation=("softmax"))(headModel)
        
        # returning the model
        return headModel
    
        