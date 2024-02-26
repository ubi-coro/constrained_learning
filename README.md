# constrained_learning
Handy package for training output constrained neural networks. As of now, this package only contains 
constrained extreme learning machines (CELM), constrained multi layer neural networks will be added in the future.

# Structure
The source code is contained in the [src](src) folder.
The implementation of the (C)ELM can be found in [src/learner.py](src/learner.py).

To install the package:
```
pip install -r requirements.txt
```

To run the synthetic examples: 
``` 
python examples\gaussian.py
python examples\monotone_quadratic.py
```

To run a more complex example that involves learning a Lyapunov function, which in turn
defines constraints for the learning process of a stable dynamical system:
``` 
python examples\stable_dynamical_system.py
```

For reference, take a look at [this](https://www.neuralautomation.de/app/download/25504160/LemmeNeumannReinhartSteil_NeuCom2014.pdf) paper.

# License
MIT license - See [LICENSE](LICENSE).