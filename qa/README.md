# Testing

The CI in this repository runs on github actions. To build and test locally you can use [act](https://github.com/nektos/act), which runs github workflows locally against your machine. You can install act and docker on your local machine, then from the root of this repo simply run:
```bash
mkdir /tmp/artifacts
act -P self-hosted=catthehacker/ubuntu:act-22.04 --detect-event --artifact-server-path /tmp/artifacts -W .github/workflows/cmake.yml
```
This will build and test the backend.