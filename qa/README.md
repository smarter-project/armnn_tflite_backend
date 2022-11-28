# Testing
In order to run the test suite for this backend perform the following steps:

Fetch the test images:
```bash
cd qa
wget https://images.getsmarter.io/ml-models/armnn_tflite_backend_triton_model_repo.tar.gz
tar -xvzf armnn_tflite_backend_triton_model_repo.tar.gz
```

The CI in this repository runs on github actions. To build and test locally you can use [act](https://github.com/nektos/act), which runs github workflows locally against your machine. You can install act on your local machine, then from the root of this repo simply run:
```bash
act
```
This will build and test the backend.