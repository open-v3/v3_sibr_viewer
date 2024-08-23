/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <rasterizer.h>
#include <imgui_internal.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <picojson/picojson.hpp>
#include <chrono>
#include <opencv2/core/utils/logger.hpp>
#include <imgui/imgui.h>
#include <projects/gaussianviewer/renderer/JsonUtils.hpp>
#include <projects/gaussianviewer/renderer/OpenCVVideoDecoder.hpp>
#include <projects/gaussianviewer/renderer/GSVideoDecoder.hpp>
#include <future>
#include <execution>
#include <bitset>
#include <fstream>
#include <string>
#include <sstream>
// #include <curl/curl.h>
// #include <imgui/imgui_impl_opengl3.h>
// #include <imgui/imgui_impl_glfw.h>
// #include <unistd.h>
// Define the types and sizes that make up the contents of each Gaussian 
// in the trained model.

// #define _sh_degree 1

typedef sibr::Vector3f Pos;
template<int D>
struct SHs
{
	float shs[(D+1)*(D+1)*3];
};
struct Scale
{
	float scale[3];
};
struct Rot
{
	float rot[4];
};
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

bool image_dequan(cv::Mat m_att_img, std::vector<float>& gaussian_vector, float max, float min) {
	cv::Mat att_img = m_att_img.clone();
	// perform dequantization
	att_img.convertTo(att_img, CV_32F, 1.0 / (std::pow(2.0, 8) - 1.0));
	cv::Mat dequantized_img = att_img * (max - min) + min;
	// convert to 1D vector
	std::vector<float> deimg_vector(dequantized_img.rows * dequantized_img.cols);
	if (dequantized_img.isContinuous()) {
		deimg_vector.assign((float*)dequantized_img.datastart, (float*)dequantized_img.dataend);
	} else {
		return false;
	}
	gaussian_vector = deimg_vector;
	return true;
}

// Load the Gaussians from the given png.
void sibr::GaussianView::loadVideo_func(int frame_index)
{
	// int json_index = frame_index % 101;
	int json_index = frame_index;
	int shs_dim = 3 * (_sh_degree + 1) * (_sh_degree + 1);
	int ply_dim = (14 + shs_dim);

	// get info from json
	picojson::object frameobj = minmax_obj[std::to_string(json_index)].get<picojson::object>();
	int m_count = static_cast<int>(frameobj["num"].get<double>());
	picojson::array arr = frameobj["info"].get<picojson::array>();
	std::vector<float> minmax_values;
	for (picojson::value& val : arr) {
		float value = static_cast<float>(val.get<double>());
		minmax_values.push_back(value);
	}
	std::cout << minmax_values.size() << std::endl;
	if (minmax_values.size() != 2 * ply_dim) {
		SIBR_ERR << "Error: " << "vector size not match" << std::endl;
	}

	std::vector<Pos> pos(m_count);
	std::vector<Rot> rot(m_count);
	std::vector<Scale> scale(m_count);
	std::vector<float> opacity(m_count);
	std::vector<SHs<3>> shs(m_count);

	std::vector<std::vector<float>> gaussian_data(ply_dim - 3);
	// xyz, fdc 012, frest0 - 44, opacity, scale 012, rot 0123
	//0 1 2,  3 4 5 , 6 - 50    , 51     , 52 53 54, 55 56 57 58

	// xyz, fdc 012, frest0 - 8, opacity, scale 012, rot 0123
	//0 1 2,  3 4 5 , 6 - 14    , 15     , 16 17 18, 19 20 21 22

	auto start = std::chrono::high_resolution_clock::now();
	// pos using single thread
	for (size_t att_index = 0; att_index <= 2; att_index ++) {
		float min = minmax_values[2 * att_index];
		float max = minmax_values[2 * att_index + 1];
		cv::Mat att_even_img = global_png_vector[att_index * 2][frame_index].clone();
		cv::Mat att_odd_img = global_png_vector[att_index * 2 + 1][frame_index].clone();

		int total_pixels = att_even_img.rows * att_even_img.cols;
		std::vector<uint8_t> even_bytes(total_pixels), odd_bytes(total_pixels);
		std::memcpy(even_bytes.data(), att_even_img.data, total_pixels * sizeof(uint8_t));
		std::memcpy(odd_bytes.data(), att_odd_img.data, total_pixels * sizeof(uint8_t));
		std::vector<uint16_t> combined(total_pixels);
		std::transform(std::execution::par, even_bytes.begin(), even_bytes.end(), odd_bytes.begin(), combined.begin(), [](uint8_t e, uint8_t o) {
			uint16_t result = 0;
			// for (int j = 0; j < 8; j++) {
			// 	uint16_t even_bit = (e >> j) & 1;
			// 	uint16_t odd_bit = (o >> j) & 1;
			// 	result |= (even_bit << (2 * j)) | (odd_bit << (2 * j + 1));
			// }
			// change e to 16 bit
			uint16_t odd = o;
			result = (odd << 8) | e;
			return result;
		});
		cv::Mat att_img = cv::Mat(att_even_img.size(), CV_16UC1, combined.data());

		att_img.convertTo(att_img, CV_32F, 1.0 / (std::pow(2.0, 16) - 1.0));
		cv::Mat dequantized_img = att_img * (max - min) + min;
		// convert to 1D vector
		std::vector<float> deimg_vector(dequantized_img.rows * dequantized_img.cols);
		if (dequantized_img.isContinuous()) {
			deimg_vector.assign((float*)dequantized_img.datastart, (float*)dequantized_img.dataend);
		}
		gaussian_data[att_index] = deimg_vector;
	}

	// std::cout << "Debuggggggggggg" << std::endl;

	// other attributes using multithreading
	std::vector<std::future<bool>> thread_futures;
	for (size_t att_index = 6; att_index < ply_dim; att_index ++) {
		float min = minmax_values[2 * att_index];
		float max = minmax_values[2 * att_index + 1];
		// std::cout << "666666666" << std::endl;
		thread_futures.push_back(std::async(std::launch::async, image_dequan, global_png_vector[att_index + 3][frame_index], std::ref(gaussian_data[att_index - 3]), max, min));
	}
	for (auto& f : thread_futures) {
		bool success = f.get(); // This will wait for the thread to finish
		if (!success) {
			std::cerr << "Failed to process some videos" << std::endl;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	// std::cout << frame_index << "First For Elapsed time: " << elapsed.count() << " ms" << std::endl;

	// std::cout << "Debuggggggggggg" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	for (size_t index = 0; index < m_count; index++) {
		// preprocessed
		pos[index].x() = gaussian_data[0][index];
		pos[index].y() = gaussian_data[1][index];
		pos[index].z() = gaussian_data[2][index];
		rot[index].rot[0] = gaussian_data[ply_dim - 7][index];
		rot[index].rot[1] = gaussian_data[ply_dim - 6][index];
		rot[index].rot[2] = gaussian_data[ply_dim - 5][index];
		rot[index].rot[3] = gaussian_data[ply_dim - 4][index];
		scale[index].scale[0] = gaussian_data[ply_dim - 10][index];
		scale[index].scale[1] = gaussian_data[ply_dim - 9][index];
		scale[index].scale[2] = gaussian_data[ply_dim - 8][index];
		opacity[index] = gaussian_data[ply_dim - 11][index];
		// for (int j = 0; j <= 44; j += 4) {
		// 	shs[index].shs[j] = gaussian_data[3 + j][index];
		// 	shs[index].shs[j+1] = gaussian_data[3 + j+1][index];
		// 	shs[index].shs[j+2] = gaussian_data[3 + j+2][index];
		// 	shs[index].shs[j+3] = gaussian_data[3 + j+3][index];
		// }
		for (int j = 0; j < shs_dim; j++) {
			shs[index].shs[j] = gaussian_data[3 + j][index];
		}
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	// std::cout << frame_index << "Second For Elapsed time: " << elapsed.count() << " ms" << std::endl;
	int cnt = frame_index;

	start = std::chrono::high_resolution_clock::now();
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda_array[cnt], sizeof(Pos) * m_count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda_array[cnt], pos.data(), sizeof(Pos) * m_count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda_array[cnt], sizeof(Rot) * m_count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda_array[cnt], rot.data(), sizeof(Rot) * m_count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda_array[cnt], sizeof(Scale) * m_count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda_array[cnt], scale.data(), sizeof(Scale) * m_count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda_array[cnt], sizeof(float) * m_count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda_array[cnt], opacity.data(), sizeof(float) * m_count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda_array[cnt], sizeof(SHs<3>) * m_count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda_array[cnt], shs.data(), sizeof(SHs<3>) * m_count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda_array[cnt], 2 * m_count * sizeof(int)));
	P_array[frame_index] = m_count;
	end = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	// std::cout << frame_index << "CUDA memcopy Elapsed time: " << elapsed.count() << " ms" << std::endl;
	// ready_frames = frame_index;
	ready_array[frame_index] = 1;
	ready_frames = frame_index;
	// std::cout << "ready_frames: " << ready_frames << std::endl;
}

// ready gaussian
void sibr::GaussianView::readyVideo_func() {
	int i;
	while (frame_changed == false) {
		{
			std::unique_lock<std::mutex> lock(mtx_ready);
			cv_ready.wait(lock, [this] { return !need_ready_q.empty(); });
			i = need_ready_q.front();
			if (i > frame_id + ready_cache_size) {
				continue;
			}
			need_ready_q.pop();
		}
		{	
			loadVideo_func(i);
			std::cout << "frame " << i << " ready" << std::endl;
			if (i == sequences_length) {
				break;
			}
		}
	}
}

unsigned long long getNetReceivedBytes() {
    std::ifstream file("/proc/net/dev");
	const std::string interface("enp4s0");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find(interface) != std::string::npos) {
            std::istringstream iss(line);
            std::string temp;
            iss >> temp;
            unsigned long long bytes;
            iss >> bytes;
            return bytes;
        }
    }
    return 0;
};

namespace sibr
{
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target. 
	class BufferCopyRenderer
	{

	public:

		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }
		int& width() { return _width.get(); }
		int& height() { return _height.get(); }

	private:

		GLShader			_shader; 
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};

	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr & ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, bool white_bg, bool useInterop, int device) :
	_scene(ibrScene),
	_dontshow(messageRead),
	sibr::ViewBase(render_w, render_h)
{
	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	// if (prop.major < 7)
	// {
	// 	SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
	// }

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;

	std::vector<uint> imgs_ulr;
	const auto & cams = ibrScene->cameras()->inputCameras();
	for(size_t cid = 0; cid < cams.size(); ++cid) {
		if(cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);


	// init folder
	folder = std::string(video_path[current_video_item]);
	_sh_degree = video_sh[current_video_item];
	num_att_index = (14 + (3 * (_sh_degree + 1) * (_sh_degree + 1))) + 3;

	// multi-frame setting
	global_png_vector.resize(num_att_index);

	// init download and ready to 0
	memset(downloaded_array, 0, sizeof(downloaded_array));
	memset(ready_array, 0, sizeof(ready_array));

	// std::string folder = "http://10.15.89.67:10000/video_new/";
	// std::string group_json_path = "http://10.15.89.67:10000/group_info.json";
	std::string group_json_path = folder + "group_info.json";
	picojson::object group_obj = fetchJsonObj(group_json_path);
	size_t num_groups = group_obj.size();
	// for (const auto& kv : group_obj) {
	// 	picojson::object innerObj = kv.second.get<picojson::object>();
    //     picojson::array frame_index = innerObj["frame_index"].get<picojson::array>();
    //     group_frame_index.push_back(std::make_pair((int)frame_index[0].get<double>(), (int)frame_index[1].get<double>()));
    //     picojson::array name_index = innerObj["name_index"].get<picojson::array>();
    //     group_name_index.push_back(std::make_pair((int)name_index[0].get<double>(), (int)name_index[1].get<double>()));
	// }
	for (int i = 0; i < num_groups; i++) {
		picojson::object innerObj = group_obj[std::to_string(i)].get<picojson::object>();
		picojson::array frame_index = innerObj["frame_index"].get<picojson::array>();
		group_frame_index.push_back(std::make_pair((int)frame_index[0].get<double>(), (int)frame_index[1].get<double>()));
		picojson::array name_index = innerObj["name_index"].get<picojson::array>();
		group_name_index.push_back(std::make_pair((int)name_index[0].get<double>(), (int)name_index[1].get<double>()));
	}

	// get last frame index
	sequences_length = group_frame_index[group_frame_index.size() - 1].second;
	// init global_png_vector for each attribute
	for (int i = 0; i < num_att_index; i++) {
		global_png_vector[i].resize(sequences_length + 1);
	}

	auto start1 = std::chrono::high_resolution_clock::now();

	std::vector<std::future<bool>> futures;
	int initial_group_index = 0;
	int download_start_index = group_frame_index[initial_group_index].first;
	int download_end_index = group_frame_index[initial_group_index].second;
	for (int att_index = 0; att_index < num_att_index; att_index ++) {
		std::string videopath = folder + "group" + std::to_string(initial_group_index) + "/" + std::to_string(att_index) + ".mp4";
        futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
	}

	for (auto& f : futures) {
        bool success = f.get(); // This will wait for the thread to finish
        if (!success) {
            std::cerr << "Failed to process some videos" << std::endl;
        }
    }
	// downloaded_frames = global_png_vector[0].size();
	downloaded_frames = download_end_index;
	// set downloaded from download_start_index to download_end_index
	std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);

	auto end1 = std::chrono::high_resolution_clock::now();
	double elapsed_seconds1 = std::chrono::duration<double>(end1 - start1).count();
	// std::cout << "OpenCV read 100 frames: " << elapsed_seconds1 << " seconds" << std::endl;

	// read the json info
	std::string minmax_json_path = folder + "viewer_min_max.json";
	minmax_obj = fetchJsonObj(minmax_json_path);

	// start timer
	
	auto start = std::chrono::high_resolution_clock::now();
	// std::cout << "preload index start: " << group_frame_index[0].first << std::endl;
	// std::cout << "preload index end: " << group_frame_index[0].second << std::endl;
	// for (int i = group_frame_index[0].first; i <= group_frame_index[0].second; i+=frame_step)
	// std::cout << "group frame index" << group_frame_index[0].second << std::endl;
	for (int i = download_start_index; i <= download_end_index; i+=frame_step)
	{	
		loadVideo_func(i);
	}
	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_seconds = std::chrono::duration<double>(end - start).count();
	std::cout << "Elapsed time: " << elapsed_seconds << " seconds" << std::endl;
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));

	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));
	count = P_array[0];
	pos_cuda = pos_cuda_array[0];
	rot_cuda = rot_cuda_array[0];
	scale_cuda = scale_cuda_array[0];
	opacity_cuda = opacity_cuda_array[0];
	shs_cuda = shs_cuda_array[0];
	rect_cuda = rect_cuda_array[0];

	_gaussianRenderer = new GaussianSurfaceRenderer();

	// Create GL buffer ready for CUDA/GL interop
	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	if (useInterop)
	{
		if (cudaPeekAtLastError() != cudaSuccess)
		{
			SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
		}
		cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
		useInterop &= (cudaGetLastError() == cudaSuccess);
	}
	if (!useInterop)
	{
		fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		_interop_failed = true;
	}

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

	download_thread_ = std::thread(&sibr::GaussianView::download_func, this);
	std::lock_guard<std::mutex> lock(mtx_ready);
	for (int group_index = 1; group_index < group_frame_index.size(); group_index++) {
		need_download_q.push(group_index);
	}
	cv_download.notify_one();

	ready_thread_ = std::thread(&sibr::GaussianView::readyVideo_func, this);

	// set up time recoder for stable play speed
	frameDuration = std::chrono::milliseconds(33);
	lastUpdateTimestamp = std::chrono::high_resolution_clock::now();

	// set up time recoder for memory read
	MemframeDuration = std::chrono::milliseconds(1000);
	MemlastUpdateTimestamp = std::chrono::high_resolution_clock::now();

	// read network
	last_total_bytes = getNetReceivedBytes();

	// std::cout << "the number of 0 gaussian" << count << std::endl;
}

// void sibr::GaussianView::download_func() {
// 	// 0 has been loaded at init
// 	for (int i = 1; i < group_frame_index.size(); i++) {
// 		if (frame_changed) {
// 			return;
// 		}
// 		std::vector<std::future<bool>> thread_futures;
// 		int download_start_index = group_frame_index[i].first;
// 		int download_end_index = group_frame_index[i].second;
// 		for (int att_index = 0; att_index < num_att_index; att_index ++) {
// 			std::string videopath = folder + "group" + std::to_string(i) + "/" + std::to_string(att_index) + ".mp4";
// 			thread_futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
// 		}

// 		for (auto& f : thread_futures) {
// 			bool success = f.get(); // This will wait for the thread to finish
// 			if (!success) {
// 				std::cerr << "Failed to process some videos" << std::endl;
// 			}
// 		}
// 		downloaded_frames = download_end_index;
// 		// std::cout << "before downloaded array" << std::endl;
// 		std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);
// 		std::lock_guard<std::mutex> lock(mtx);
// 		for (int j = download_start_index; j <= download_end_index; j+=frame_step) {
// 			std::cout << "Need frame " << j << " to ready" << std::endl;
// 			need_ready_q.push(j);
// 		}
// 		cv.notify_one();
// 	}
// }

// download gaussian
void sibr::GaussianView::download_func() {
	// // 0 has been loaded at init
	// for (int i = 1; i < group_frame_index.size(); i++) {
	// 	if (frame_changed) {
	// 		return;
	// 	}
	// 	std::vector<std::future<bool>> thread_futures;
	// 	int download_start_index = group_frame_index[i].first;
	// 	int download_end_index = group_frame_index[i].second;
	// 	for (int att_index = 0; att_index < num_att_index; att_index ++) {
	// 		std::string videopath = folder + "group" + std::to_string(i) + "/" + std::to_string(att_index) + ".mp4";
	// 		thread_futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
	// 	}

	// 	for (auto& f : thread_futures) {
	// 		bool success = f.get(); // This will wait for the thread to finish
	// 		if (!success) {
	// 			std::cerr << "Failed to process some videos" << std::endl;
	// 		}
	// 	}
	// 	downloaded_frames = download_end_index;
	// 	// std::cout << "before downloaded array" << std::endl;
	// 	std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);
	// 	std::lock_guard<std::mutex> lock(mtx);
	// 	for (int j = download_start_index; j <= download_end_index; j+=frame_step) {
	// 		std::cout << "Need frame " << j << " to ready" << std::endl;
	// 		need_ready_q.push(j);
	// 	}
	// 	cv.notify_one();
	// }
	int group_index;
	while (frame_changed == false) {
		{
			std::unique_lock<std::mutex> lock(mtx_download);
			cv_download.wait(lock, [this] { return !need_download_q.empty(); });
			group_index = need_download_q.front();
			// if (i > frame_id + download_cache_size) {
			// 	continue;
			// }
			need_download_q.pop();
		}
		{
			int download_start_index = group_frame_index[group_index].first;
			int download_end_index = group_frame_index[group_index].second;
			if (downloaded_array[download_start_index] == 0) {
				auto start = std::chrono::high_resolution_clock::now();
				std::cout << "Do not find cache" << std::endl;
				std::vector<std::future<bool>> thread_futures;
				for (int att_index = 0; att_index < num_att_index; att_index ++) {
					std::string videopath = folder + "group" + std::to_string(group_index) + "/" + std::to_string(att_index) + ".mp4";
					thread_futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
				}

				for (auto& f : thread_futures) {
					bool success = f.get(); // This will wait for the thread to finish
					if (!success) {
						std::cerr << "Failed to process some videos" << std::endl;
					}
				}
				auto end = std::chrono::high_resolution_clock::now();
				double elapsed_seconds = std::chrono::duration<double>(end - start).count();
				std::cout << "Download Elapsed time: " << elapsed_seconds << " seconds" << std::endl;
			} else {
				std::cout << "Cached group: " << group_index << std::endl;
				std::cout << "Cached frame index: " << download_start_index << " " << download_end_index << std::endl;
			}
			downloaded_frames = download_end_index;
			std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);
			std::lock_guard<std::mutex> lock(mtx_download);
			for (int j = download_start_index; j <= download_end_index; j+=frame_step) {
				// std::cout << "Need frame " << j << " to ready" << std::endl;
				need_ready_q.push(j);
			}
			cv_ready.notify_one();
			if (group_index == group_frame_index.size() - 1) {
				break;
			}
		}
	}
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr & newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto & cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget & dst, const sibr::Camera & eye)
{
	if (currMode == "Ellipsoids")
	{
		// stop ellipsoid rendering to avoid loading gData
		return;
		// _gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
	}
	else if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		// Convert view and projection to target coordinate system
		auto view_mat = eye.view();
		auto proj_mat = eye.viewproj();
		view_mat.row(1) *= -1;
		view_mat.row(2) *= -1;
		proj_mat.row(1) *= -1;

		// Compute additional view parameters
		float tan_fovy = tan(eye.fovy() * 0.5f);
		float tan_fovx = tan_fovy * eye.aspect();

		// Copy frame-dependent data to GPU
		CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

		float* image_cuda = nullptr;
		if (!_interop_failed)
		{
			// Map OpenGL buffer resource for use with CUDA
			size_t bytes;
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
		}
		else
		{
			image_cuda = fallbackBufferCuda;
		}

		// Rasterize
		int* rects = _fastCulling ? rect_cuda : nullptr;
		// std::cout << "Rendering resolution: " << _resolution.x() << "x" << _resolution.y() << "x" << std::endl;
		CudaRasterizer::Rasterizer::forward(
			geomBufferFunc,
			binningBufferFunc,
			imgBufferFunc,
			count, _sh_degree, 16,
			background_cuda,
			_resolution.x(), _resolution.y(),
			pos_cuda,
			shs_cuda,
			nullptr,
			opacity_cuda,
			scale_cuda,
			_scalingModifier,
			rot_cuda,
			nullptr,
			view_cuda,
			proj_cuda,
			cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			nullptr,
			rects
		);

		if (!_interop_failed)
		{
			// Unmap OpenGL resource for use with OpenGL
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		// Copy image contents to framebuffer
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}
}

void sibr::GaussianView::onUpdate(Input & input)
{
}

void sibr::GaussianView::onGUI()
{
	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str())) 
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))
				currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
	}

	ImGui::Begin("Play");
		ImGui::Checkbox("multi view play", &_multi_view_play);
		if (ImGui::SliderInt("Playing Frame", &frame_id, 0, (sequences_length - frame_st) / frame_step - 1)) {
			std::cout << "frame_id changed to " << frame_id << std::endl;
			// empty all cuda memory
			for (int i = 0; i < sequences_length; i++) {
				if (ready_array[i] == 1) {
					CUDA_SAFE_CALL_ALWAYS(cudaFree(pos_cuda_array[i]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(rot_cuda_array[i]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(scale_cuda_array[i]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(opacity_cuda_array[i]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(shs_cuda_array[i]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(rect_cuda_array[i]));
					// set ready to 0
					ready_array[i] = 0;
				}
			}
			frame_changed = true;
			_multi_view_play = false;
			download_thread_.join();
			ready_thread_.join();
			std::cout << "detech frame changed, reset download thread and ready thread" << std::endl;
			// clean the queue
			{
				std::lock_guard<std::mutex> lock(mtx_download);
				while (!need_download_q.empty()) {
					need_download_q.pop();
				}
			}
			{
				std::lock_guard<std::mutex> lock(mtx_ready);
				while (!need_ready_q.empty()) {
					need_ready_q.pop();
				}
			}
			// downloaded frames reserves, as its in cpu mem
			// ready all cleand, as its directly converts to cuda mem
			
			// find the group index that frame_id belongs to
			int group_index = 0;
			for (int i = 0; i < group_frame_index.size(); i++) {
				if (frame_id >= group_frame_index[i].first && frame_id <= group_frame_index[i].second) {
					group_index = i;
					break;
				}
			}

			std::cout << "group id changed to " << group_index << std::endl;

			int download_start_index = group_frame_index[group_index].first;
			int download_end_index = group_frame_index[group_index].second;
			// test if the group is already downloaded
			if (downloaded_array[group_frame_index[group_index].first] == 0) {
				std::cout << "group " << group_index << " not downloaded" << std::endl;
				std::cout << "download start index: " << download_start_index << std::endl;
				std::cout << "download end index: " << download_end_index << std::endl;
				std::vector<std::future<bool>> thread_futures;
				for (int att_index = 0; att_index < num_att_index; att_index ++) {
					std::string videopath = folder + "group" + std::to_string(group_index) + "/" + std::to_string(att_index) + ".mp4";
					thread_futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
				}

				for (auto& f : thread_futures) {
					bool success = f.get(); // This will wait for the thread to finish
					if (!success) {
						std::cerr << "Failed to process some videos" << std::endl;
					}
				}
				downloaded_frames = download_end_index;
				std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);
			}
			// ready all frames in the group just need current to download end
			for (int i = frame_id; i <= download_end_index; i+=frame_step) {
				loadVideo_func(i);
			}
			{
				// push all future undownloaded group index to queue
				std::lock_guard<std::mutex> lock(mtx_download);
				for (int i = group_index + 1; i < group_frame_index.size(); i++) {
					need_download_q.push(i);
				}
				cv_download.notify_one();
			}

			// restart the 2 threads
			frame_changed = false;
			download_thread_ = std::thread(&sibr::GaussianView::download_func, this);
			ready_thread_ = std::thread(&sibr::GaussianView::readyVideo_func, this);

			std::cout << "frame_id changed to " << frame_id  << " done" << std::endl;

		}
		ImGui::SliderInt("Download Frame", &downloaded_frames, 0, sequences_length);
		ImGui::SliderInt("Ready Frame", &ready_frames, 0, sequences_length);
		// ImGui::ProgressBar(float(downloaded_frames) / float(sequences_length), ImVec2(0.0f, 0.0f), "Download Frame");
	ImGui::End();

	// // convert folder to char
	// char buf[100];
	// strncpy(buf, folder.c_str(), sizeof(buf));
	// buf[sizeof(buf) - 1] = 0;
	// // add input url
	// if (ImGui::InputText("Remote URL", buf, sizeof(buf))) {
	// 	folder = buf;
	// }

	if (ImGui::Combo("Remote Video list", &current_video_item, video_path.data(), video_path.size())) {
		std::cout << "Changing Video to: " << video_path[current_video_item] << std::endl;
		// empty all cuda memory
		for (int i = 0; i < sequences_length; i++) {
			if (ready_array[i] == 1) {
				CUDA_SAFE_CALL_ALWAYS(cudaFree(pos_cuda_array[i]));
				CUDA_SAFE_CALL_ALWAYS(cudaFree(rot_cuda_array[i]));
				CUDA_SAFE_CALL_ALWAYS(cudaFree(scale_cuda_array[i]));
				CUDA_SAFE_CALL_ALWAYS(cudaFree(opacity_cuda_array[i]));
				CUDA_SAFE_CALL_ALWAYS(cudaFree(shs_cuda_array[i]));
				CUDA_SAFE_CALL_ALWAYS(cudaFree(rect_cuda_array[i]));
				// set ready to 0
				ready_array[i] = 0;
			}
		}
		frame_changed = true;
		_multi_view_play = false;
		download_thread_.join();
		ready_thread_.join();
		std::cout << "detect video changed, reset download thread and ready thread" << std::endl;
		
		// downloaded and ready frames need to be cleaned
		memset(downloaded_array, 0, sizeof(downloaded_array));
		memset(ready_array, 0, sizeof(ready_array));
		downloaded_frames = -1;
		ready_frames = -1;
		frame_id = 0;
		// clear group info
		group_frame_index.clear();
		group_name_index.clear();
		// clean global_png_vector
		for (int i = 0; i < num_att_index; i++) {
			global_png_vector[i].clear();
		}
		global_png_vector.clear();
		// clean the queue
		{
			std::lock_guard<std::mutex> lock(mtx_download);
			while (!need_download_q.empty()) {
				need_download_q.pop();
			}
		}
		{
			std::lock_guard<std::mutex> lock(mtx_ready);
			while (!need_ready_q.empty()) {
				need_ready_q.pop();
			}
		}
		// ready all cleand, as its directly converts to cuda mem
		
		// reset group info and folder and global_png_vector
		folder = video_path[current_video_item];
		_sh_degree = video_sh[current_video_item];
		num_att_index = (14 + (3 * (_sh_degree + 1) * (_sh_degree + 1))) + 3;
		std::string group_json_path = folder + "group_info.json";
		picojson::object group_obj = fetchJsonObj(group_json_path);
		std::string minmax_json_path = folder + "viewer_min_max.json";
		minmax_obj = fetchJsonObj(minmax_json_path);
		size_t num_groups = group_obj.size();
		for (int i = 0; i < num_groups; i++) {
			picojson::object innerObj = group_obj[std::to_string(i)].get<picojson::object>();
			picojson::array frame_index = innerObj["frame_index"].get<picojson::array>();
			group_frame_index.push_back(std::make_pair((int)frame_index[0].get<double>(), (int)frame_index[1].get<double>()));
			picojson::array name_index = innerObj["name_index"].get<picojson::array>();
			group_name_index.push_back(std::make_pair((int)name_index[0].get<double>(), (int)name_index[1].get<double>()));
		}
		global_png_vector.resize(num_att_index);
		sequences_length = group_frame_index[group_frame_index.size() - 1].second;
		std::cout << "New video length: " << sequences_length << std::endl;
		for (int i = 0; i < num_att_index; i++) {
			global_png_vector[i].resize(sequences_length + 1);
		}

		std::cout << "Reloading initial frames" << std::endl;

		// initial group index
		int group_index = 0;
		int download_start_index = group_frame_index[group_index].first;
		int download_end_index = group_frame_index[group_index].second;
		// test if the group is already downloaded
		std::vector<std::future<bool>> thread_futures;
		for (int att_index = 0; att_index < num_att_index; att_index ++) {
			std::string videopath = folder + "group" + std::to_string(group_index) + "/" + std::to_string(att_index) + ".mp4";
			thread_futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
		}
		for (auto& f : thread_futures) {
			bool success = f.get(); // This will wait for the thread to finish
			if (!success) {
				std::cerr << "Failed to process some videos" << std::endl;
			}
		}
		downloaded_frames = download_end_index;
		std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);
		// ready all frames in the group just need current to download end
		for (int i = download_start_index; i <= download_end_index; i+=frame_step) {
			loadVideo_func(i);
		}
		{
			// push all future undownloaded group index to queue
			std::lock_guard<std::mutex> lock(mtx_download);
			for (int i = 1; i < group_frame_index.size(); i++) {
				need_download_q.push(i);
			}
			cv_download.notify_one();
		}

		// restart the 2 threads
		frame_changed = false;
		download_thread_ = std::thread(&sibr::GaussianView::download_func, this);
		ready_thread_ = std::thread(&sibr::GaussianView::readyVideo_func, this);

		std::cout << "video changed to " << folder << " done" << std::endl;
	}

	auto now = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdateTimestamp);

	if (_multi_view_play) {
		// if (frame_id + 1 <= ready_frames) {
		if (ready_array[frame_id + 1] == 1) {
			if (elapsed > frameDuration) {
				frame_id += 1;
				if (frame_id >= sequences_length) {
					_multi_view_play = false;
				}
				lastUpdateTimestamp = now;
				if (frame_id - 1 >= 0) {
					CUDA_SAFE_CALL_ALWAYS(cudaFree(pos_cuda_array[frame_id - 1]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(rot_cuda_array[frame_id - 1]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(scale_cuda_array[frame_id - 1]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(opacity_cuda_array[frame_id - 1]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(shs_cuda_array[frame_id - 1]));
					CUDA_SAFE_CALL_ALWAYS(cudaFree(rect_cuda_array[frame_id - 1]));
					// set ready to 0
					ready_array[frame_id - 1] = 0;
				}
			}
		}
	}
	if (frame_id >= sequences_length) {
		_multi_view_play = false;
	}
	count = P_array[frame_id];
	pos_cuda = pos_cuda_array[frame_id];
	rot_cuda = rot_cuda_array[frame_id];
	scale_cuda = scale_cuda_array[frame_id];
	opacity_cuda = opacity_cuda_array[frame_id];
	shs_cuda = shs_cuda_array[frame_id];
	rect_cuda = rect_cuda_array[frame_id];
	ImGui::Checkbox("Fast culling", &_fastCulling);

	// visualize the data png
	static GLuint image_texture = 0;
	if (image_texture != 0) {
		glDeleteTextures(1, &image_texture);
		image_texture = 0;
	}
	glDeleteTextures(1, &image_texture);
	glGenTextures(1, &image_texture);
	glBindTexture(GL_TEXTURE_2D, image_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	cv::Mat rgba_image(global_png_vector[1][frame_id].rows, global_png_vector[1][frame_id].cols, CV_8UC4);
	cv::cvtColor(global_png_vector[1][frame_id], rgba_image, cv::COLOR_GRAY2RGBA);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba_image.cols, rgba_image.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_image.data);
	ImGui::Image((void*)(intptr_t)image_texture, ImVec2(rgba_image.cols, rgba_image.rows));

	ImGui::End();


	auto Memnow = std::chrono::high_resolution_clock::now();
	auto Memelapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Memnow - MemlastUpdateTimestamp);
	if (Memelapsed > MemframeDuration) {
		auto [used, total] = gpuInfoManager.getMemoryUsage();
		Memused = used;
		Memtotal = total;
		MemlastUpdateTimestamp = Memnow;

		// net_speed_buffer
		auto total_bytes = getNetReceivedBytes();
		// divide memelapsed
		float speed = (total_bytes - last_total_bytes) * 8 / (Memelapsed.count() / 1000.f) / 1024 / 1024;
		last_total_bytes = total_bytes;
		net_speed_buffer.push_back(speed);
		if (net_speed_buffer.size() > 100) {
			net_speed_buffer.erase(net_speed_buffer.begin());
		}
	}
	ImGui::Begin("GPU Memory Usage");
	std::vector<float> memoryUsage = gpuInfoManager.getMemoryUsageBuffer();
	if (!memoryUsage.empty()) {
		ImGui::PlotLines("Memory Usage", memoryUsage.data(), memoryUsage.size(), 0, NULL, 0, 100, ImVec2(0, 80));
		ImGui::Text("Used: %.2f(GB) / Total: %.2f(GB)", float(Memused) / 1024.f / 1024.f / 1024.f, float(Memtotal) / 1024.f / 1024.f / 1024.f);
	}
	ImGui::End();

	// net
	ImGui::Begin("Network Usage");
	if (!net_speed_buffer.empty()) {
		// find the max speed
		float max_speed = *std::max_element(net_speed_buffer.begin(), net_speed_buffer.end());
		ImGui::PlotLines("Network Speed", net_speed_buffer.data(), net_speed_buffer.size(), 0, NULL, 0, max_speed, ImVec2(0, 80));
		ImGui::Text("Current Speed: %.2f(Mbps)", net_speed_buffer[net_speed_buffer.size() - 1]);
	}
	ImGui::End();

	if(!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"\
			" It did NOT work for your current configuration.\n"\
			" For highest performance, OpenGL and CUDA must run on the same\n"\
			" GPU on an OS that supports interop.You can try to pass a\n"\
			" non-zero index via --device on a multi-GPU system, and/or try\n" \
			" attaching the monitors to the main CUDA card.\n"\
			" On a laptop with one integrated and one dedicated GPU, you can try\n"\
			" to set the preferred GPU via your operating system.\n\n"\
			" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  ")) {
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

sibr::GaussianView::~GaussianView()
{
	for (int i = 0; i < sequences_length; i++) {
		if (ready_array[i] == 1) {
			CUDA_SAFE_CALL_ALWAYS(cudaFree(pos_cuda_array[i]));
			CUDA_SAFE_CALL_ALWAYS(cudaFree(rot_cuda_array[i]));
			CUDA_SAFE_CALL_ALWAYS(cudaFree(scale_cuda_array[i]));
			CUDA_SAFE_CALL_ALWAYS(cudaFree(opacity_cuda_array[i]));
			CUDA_SAFE_CALL_ALWAYS(cudaFree(shs_cuda_array[i]));
			CUDA_SAFE_CALL_ALWAYS(cudaFree(rect_cuda_array[i]));
		}
	}
	frame_changed = true;
	download_thread_.detach();
	ready_thread_.detach();
	// Cleanup
	cudaFree(pos_cuda);
	cudaFree(rot_cuda);
	cudaFree(scale_cuda);
	cudaFree(opacity_cuda);
	cudaFree(shs_cuda);

	cudaFree(view_cuda);
	cudaFree(proj_cuda);
	cudaFree(cam_pos_cuda);
	cudaFree(background_cuda);
	cudaFree(rect_cuda);

	if (!_interop_failed)
	{
		cudaGraphicsUnregisterResource(imageBufferCuda);
	}
	else
	{
		cudaFree(fallbackBufferCuda);
	}
	glDeleteBuffers(1, &imageBuffer);

	if (geomPtr)
		cudaFree(geomPtr);
	if (binningPtr)
		cudaFree(binningPtr);
	if (imgPtr)
		cudaFree(imgPtr);

	delete _copyRenderer;
}
