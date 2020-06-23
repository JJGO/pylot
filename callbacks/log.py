from torch import nn


def LogParameters(experiment, parameters):

    def LogParametersCallback(experiment, epoch):
        param_dict = {}
        for parameter in parameters:
            param = getattr(experiment.model, parameter)
            if isinstance(param, (nn.ParameterList, list)):
                for i, p in enumerate(param):
                    param_dict[f'{parameter}_{i}'] = p.item()
            else:
                param_dict[parameter] = param.item()
        experiment.log(**param_dict)

    return LogParametersCallback


def TqdmParameters(experiment, parameters):

    def TqdmParametersCallback(experiment, postfix):
        param_dict = {}
        for parameter in parameters:
            param = getattr(experiment.model, parameter)
            if isinstance(param, (list, nn.ParameterList)):
                for i, p in enumerate(param):
                    param_dict[f'{parameter}_{i}'] = p.item()
            else:
                param_dict[parameter] = param.item()
        postfix.update(param_dict)

    return TqdmParametersCallback
