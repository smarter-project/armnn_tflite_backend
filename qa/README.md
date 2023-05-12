# Testing
To run tests you must first build triton server in your local environment. 
To do so you can look at the vscode devcontainer [Dockerfile](../.devcontainer/Dockerfile), which shows how to setup your development environment.

Then change directory back to the base of this repo build it using cmake (instructions for this can be found in the github ci). With the backend built you can then run:
```
cd qa
wget -q https://images.getsmarter.io/ml-models/armnn_tflite_backend_triton_model_repo.tar.gz
tar -xzf armnn_tflite_backend_triton_model_repo.tar.gz
python3 -m pytest -x -n $(( $(nproc) / 2 )) tests/ --model-repo-path $(pwd)/triton_qa_models/accuracy_test_model_repo --triton-path /workspaces/server/build/opt/tritonserver/bin/tritonserver --backend-directory $(pwd)/../install/backends -v
```
