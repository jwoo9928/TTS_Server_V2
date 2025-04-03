#!/bin/bash
set -e

# Check CUDA availability and set environment accordingly
if python -c "import torch; print(torch.cuda.is_available());" | grep -q "True"; then
    echo "CUDA is available. Using GPU."
    export DEVICE=cuda
else
    echo "CUDA is not available. Using CPU."
    export DEVICE=cpu
fi

# Execute the command passed as arguments
exec "$@"
