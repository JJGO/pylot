def Checkpoint(experiment, save_freq=1):
    i = 0

    def CheckpointCallback(epoch):
        # TODO think about a cleaner way
        # Maybe define callbacks for both epoch start and end
        # instead of just at end?
        nonlocal i
        i += 1
        if i % save_freq == 0:
            # For resumable experiments is easier to think about
            # Checkpoint to be at the beginning of epoch rather
            # than at the end
            experiment._epoch += 1
            experiment.checkpoint(tag=f"{experiment._epoch:03d}")
            experiment._epoch -= 1

    return CheckpointCallback
