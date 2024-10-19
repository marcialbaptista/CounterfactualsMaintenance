import io
import copy

class EarlyStopping():

  def __init__(self, patience=5, min_delta=0.25, restore_best_weights=True):
    """Stops training sequence of model according to parameters set to prevent overfitting.
    
    Args:
        min_delta (float): minimal amount of loss decrease to be allowed to continue
        patience (int): amount of epochs with less than min_delta loss changes after which training is stopped
        restore_best_weights (bool): if True, weights are restored to their best version before early stopping
    """
    self.patience = patience
    self.min_delta = min_delta
    self.restore_best_weights = restore_best_weights
    self.best_model = None
    self.best_loss = None
    self.counter = 0
    self.status = ""
    
  def __call__(self, model, val_loss):
    if self.best_loss == None:
      self.best_loss = val_loss
      self.best_model = copy.deepcopy(model)
    elif self.best_loss - val_loss > self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      self.best_model.load_state_dict(model.state_dict())
    elif self.best_loss - val_loss < self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.status = f"Stopped on {self.counter}"
        if self.restore_best_weights:
          model.load_state_dict(self.best_model.state_dict())
        return True
    self.status = f"{self.counter}/{self.patience}"
    return False