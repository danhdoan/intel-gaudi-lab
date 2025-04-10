# Intel Gaudi Lab - Multi-Workers Setup and Benchmarking

This guide explains how to start multiple workers for the Stable Diffusion application and benchmark their performance.

## Prerequisites

- Ensure you have all dependencies installed for running the application.
- Make sure the `uvicorn` server and required Python packages are installed.
- Install `k6` for benchmarking:
    ```bash
    sudo apt install k6
    ```

## Starting Workers

The `start.sh` script is used to start multiple workers on different Habana Processing Units (HPUs).

### Steps to Start Workers

1. Navigate to the `scripts` directory:
     ```bash
     cd scripts
     ```

2. Run the `start.sh` script:
     ```bash
     ./start.sh
     ```

     This will start 8 workers (by default) on ports ranging from 8000 to 8007. Each worker will be assigned to a specific HPU.

3. Verify that the workers are running by checking the logs or using a tool like `curl`:
     ```bash
     curl http://localhost:8000
     ```

## Benchmarking the Workers

The `benchmark.js` script is used to test the performance of the workers using `k6`.

### Steps to Benchmark

1. Navigate to the `multi-workers` directory:
     ```bash
     cd multi-workers
     ```

2. Run the benchmark script with `k6`:
     ```bash
     k6 run benchmark.js
     ```

     This will simulate 200 virtual users sending requests for 30 seconds to the worker running on `http://localhost:8080`.

3. Monitor the output to analyze the performance metrics, such as response time and error rates.

## Notes

- You can modify the number of workers and base port in the `start.sh` script by changing the `NUM_WORKERS` and `BASE_PORT` variables.
- Update the target URL in `benchmark.js` if you want to test a specific worker or port.

## Troubleshooting

- If a worker fails to start, ensure that the required dependencies are installed and the ports are not in use.
- If benchmarking fails, verify that the workers are running and accessible on the specified ports.
