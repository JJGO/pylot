def Checkpoint(experiment, save_freq=10):
    i = 0

    def CheckpointCallback(experiment, epoch):
        nonlocal i
        i += 1
        if i % save_freq == 0:
            experiment.checkpoint()

    return CheckpointCallback
