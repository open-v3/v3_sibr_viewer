# Streaming Volumetric Video Viewer for V3

## Installation

Additional required libs: 

```
nvml
curl
picojson
opencv
ffmpeg
```

```
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
```

## Usage

Please setup a nginx server in the internal network to serve as a streaming server. 

Then modify the network address in `src/projects/gaussianviewer/renderer/GaussianView.hpp`, rebuild it and run it. 