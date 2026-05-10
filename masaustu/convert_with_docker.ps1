# PowerShell helper to build and run the conversion Docker image.
# Usage (from repo root):
#   .\masaustu\convert_with_docker.ps1
# This will build an image and run it, writing mobile_app/assets/model_cnn.tflite

$imageName = 'beed-convert:latest'
$workdir = Resolve-Path -Path .
Write-Host "Building Docker image $imageName..."
docker build -f masaustu/Dockerfile.convert -t $imageName .

Write-Host "Running conversion container (this will write mobile_app/assets/model_cnn.tflite)..."
docker run --rm -v "$($workdir):/work" $imageName

Write-Host "Done. Check mobile_app/assets/model_cnn.tflite"
