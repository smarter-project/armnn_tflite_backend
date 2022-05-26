# Testing
In order to run the test suite for this backend perform the following steps on an Arm64 machine:

Ensure you have git lfs installed and initialized on your machine, as the tflite models used for testing are tracked using lfs:
```
sudo apt install git-lfs
git lfs install
```

First clone the server repository from the fork and branch specified in the following snippet and move to the repo directory:
```
git clone -b feature/armnn_tflite_test_support https://github.com/jishminor/server
cd server
```

Next run the `build.py` script to just build the triton server with the identity and ensemble backend but without including this backend:
```bash
./build.py --enable-logging --enable-stats --enable-tracing --enable-metrics --endpoint=http --endpoint=grpc --backend=identity --backend=ensemble
```

Build the ArmNN TFLite Backend using the Dockerfile in this repo:
```bash
cd <backend_directory>
docker build --build-arg TFLITE_BAZEL_BUILD=ON --build-arg TFLITE_ENABLE_FLEX_OPS=ON -t armnn_tflite_backend .
```

This will produce the `tritonserver_build` and `tritonserver` images on your machine.

Build the QA image by running the following:
```bash
docker build -t tritonserver_qa -f Dockerfile.QA .
```

Now run the QA container and mount the QA model repositories into the container so the tests will be able to access them.

```bash
docker run -it --rm -v <backend_directory>/qa/triton_qa_models:/data/inferenceserver tritonserver_qa
```

Within the container the QA tests are in /opt/tritonserver/qa. To run a test simply change directory to the test and run the test.sh script.

To run the the full test suite run:
```bash
cd qa/armnn_tflite_qa/
python3 -m pytest tests/ --model-repo-path /data/inferenceserver/accuracy_test_model_repo
```