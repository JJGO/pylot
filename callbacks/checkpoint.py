def Checkpoint(experiment, save_freq=1):
    i = 0

    def CheckpointCallback(experiment, epoch):
        nonlocal i
        i += 1
        if i % save_freq == 0:
            experiment.checkpoint(tag=f"{epoch:03d}")

    return CheckpointCallback
