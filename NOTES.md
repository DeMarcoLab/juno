
# Profiling
```bash
$ pip install snakeviz

$ python -m cProfile -o output.profile script.py

$ snakeviz output.profile
```


# Coverage

```bash

$ pip install coverage
$ coverage run -m pytest -vv
$ coverage report -m 
$ coverage html
```
