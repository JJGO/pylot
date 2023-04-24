![PYLOT](https://raw.githubusercontent.com/JJGO/pylot/assets/pylot-logo.svg)
=====

This library is designed as a supporting library to my other research efforts in Deep Learning using PyTorch. It has many features and implementations but some of the most relevant ones:

- 🧪 **Experiment Management** - Human friendly classes to wrap the complexity of running experiments. Experiments can be loaded, providing convient access to model, data and logged results.
- 🚀 **Flexible Compilation** - Experiment options are specified via hierarchical YAML configs that are implicitly verified. No more `argparse`.
- ☎️ **Two-way Callbacks** - To modify experiments in a flexible and modular way, Pylot uses callbacks to extend the behavior of experiments.
- 💾 **Simplified I/O** - Pylot removes the pains of loading/saving data through its many OOP interfaces in `util`.
- 📊 **Results Analysis** - Result aggregation primitives such as Pareto frontiers, bootstrapping, and recursive selection are included.
- 📈 **Metrics and Losses** - Implementation of common Computer Vision metrics and losses in pure PyTorch
- 🐘 **Model/Layer/Optim ZOO** - Pylot includes implementations of popular deep learning models, building blocks and optimization algorithms, such as: Sharpness Aware Minimization (SAM), Exponential Moving Average Weights (EMA), U-Net models, Voxelmorph networks, Hypernetworks, SqueezeExcitation layers and many many more.
- ✅ **Extensive Type Validation** - Type hints are included and _enforced_ thanks to pydantic.
- 🐼 **Pandas Extensions** - Pandas DataFrame API is extended to better handle result dataframes, with methods such as `.select`, `.drop_constant`, `.unique_per_col`  or the `UnixAccessor` functionality.
- 🏎️ **Designed for Speed** - From careful I/O considerations to CUDA optimizations, pylot is painstakingly designed with efficiency and speed in mind.
