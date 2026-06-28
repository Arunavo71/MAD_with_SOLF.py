# MAD.py

*Mad.py* is a Python library for running Multi-Agent Debates (MADs), with observability and governance capabilities.

## Installation
Check out this repository and run ``pip install .`` or equivalent (e.g., for Conda).

## Running a Basic MAD

Create an `.env` file in the `examples` folder.
Specify variables as follows, for example:

```python
MISTRAL_API_KEY = "<your_mistral_key_here>"
MODEL_PROVIDER = "mistral"
MODEL = "mistral-medium-latest"
LOG_LEVEL = "INFO"
SEMANTICS = "DFQuAD_model"
WAIT_FOR_RATELIMIT = 3
```

Then, run:

```
python examples/basic_mad.py -t="Kermit the Frog should be the next president of the USA."
```

## Types of MAD

## MAD Governance and Observability

## Testing
Testing is done with [pytest](https://docs.pytest.org/en/stable/).
Install test dependencies with ``pip install -e '.[dev]'``.
Then, run `pytest´.

## Contributing
We welcome contributions, prepared with care and supported by tests.

## Authors

* Timotheus Kampik - [@TimKam](https://github.com/TimKam)

* Filip Naudot - [@filipnaudot][https://github.com/filipnaudot]
