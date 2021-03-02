from functools import wraps

def flatten_loss(loss_func):

    @wraps(loss_func)
    def flattened_loss(yhat, y):
        batch_size = y.size()[0]
        return loss_func(yhat.view(batch_size, -1), y.view(batch_size, -1))

    return flattened_loss
