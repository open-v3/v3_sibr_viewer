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
#pragma once

# include "Config.hpp"
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>
# include <core/system/SimpleTimer.hpp>
# include <core/system/Config.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/view/ViewBase.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <core/renderer/PointBasedRenderer.hpp>
# include <memory>
# include <core/graphics/Texture.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <functional>
# include "GaussianSurfaceRenderer.hpp"
#include "GSVideoDecoder.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <thread>
# include <queue>
# include <mutex>
# include <condition_variable>
#include <picojson/picojson.hpp>
#include "GPUMemoryInfo.hpp"

#define SEQUENCE_LENGTH 2000

namespace CudaRasterizer
{
	class Rasterizer;
}

namespace sibr { 

	class BufferCopyRenderer;
	class BufferCopyRenderer2;

	/**
	 * \class RemotePointView
	 * \brief Wrap a ULR renderer with additional parameters and information.
	 */
	class SIBR_EXP_ULR_EXPORT GaussianView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(GaussianView);

	public:

		/**
		 * Constructor
		 * \param ibrScene The scene to use for rendering.
		 * \param render_w rendering width
		 * \param render_h rendering height
		 */
		GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, bool* message_read, bool white_bg = false, bool useInterop = true, int device = 0);

		/** Replace the current scene.
		 *\param newScene the new scene to render */
		void setScene(const sibr::BasicIBRScene::Ptr & newScene);

		/**
		 * Perform rendering. Called by the view manager or rendering mode.
		 * \param dst The destination rendertarget.
		 * \param eye The novel viewpoint.
		 */
		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;

		/**
		 * Update inputs (do nothing).
		 * \param input The inputs state.
		 */
		void onUpdate(Input& input) override;

		/**
		 * Update the GUI.
		 */
		void onGUI() override;

		/** \return a reference to the scene */
		const std::shared_ptr<sibr::BasicIBRScene> & getScene() const { return _scene; }

		virtual ~GaussianView() override;

		bool* _dontshow;

		void download_func();
		void readyVideo_func();
		void loadVideo_func(int frame_index);

	protected:

		std::string currMode = "Splats";
		int _sh_degree = 1;
		int frame_st = 0;
		int frame_step = 1;
		int frame_id = 0;
		bool _fastCulling = true;
		int _device = 0;
		int num_att_index = 29;
		int sequences_length = 0;

		int ready_cache_size = 100;

		int count;
		float* pos_cuda;
		float* rot_cuda;
		float* scale_cuda;
		float* opacity_cuda;
		float* shs_cuda;
		int* rect_cuda;
		int P_array[SEQUENCE_LENGTH];
		float* pos_cuda_array[SEQUENCE_LENGTH];
		float* rot_cuda_array[SEQUENCE_LENGTH];
		float* scale_cuda_array[SEQUENCE_LENGTH];
		float* opacity_cuda_array[SEQUENCE_LENGTH];
		float* shs_cuda_array[SEQUENCE_LENGTH];
		int* rect_cuda_array[SEQUENCE_LENGTH];
		// std::vector<cv::Mat> png_vector;
		std::vector<std::vector<cv::Mat>> global_png_vector;
		// init decoder
		GSVideoDecoder decoder;

		// init memory info
		GPUInfoManager gpuInfoManager;
		float Memused, Memtotal, Memusage;

		// init network speed info
		unsigned long long last_total_bytes = 0;
		std::vector<float> net_speed_buffer;
		// std::string folder = "http://10.15.89.67:10000/video_new_new/";
		// std::string folder = "http://10.15.89.67:10000/video_new2_qp/video_qp20/";
		// std::string folder = "http://10.15.89.67:10000/coser18_qp15_new/";
		// std::string folder = "http://10.15.89.67:10000/1015hanfu_qp0/";
		std::string folder;

		int current_video_item = 0;
		std::vector<const char*> video_path = {
			// "http://10.15.89.67:10000/wyw_hifivv/",
			"http://10.15.89.67:10000/hanfu3_qp15/",
			"http://10.15.89.67:10000/0923dancer3/",
			"http://10.15.89.67:10000/jywq_demo/",
			"http://10.15.89.67:10000/ykx_boxing_long_qp15/",
			"http://10.15.89.67:10000/jywq_qp15/",
			"http://10.15.89.67:10000/coser18_0503_qp15/",
			"http://10.15.89.67:10000/png_all_0/", 
			"http://10.15.89.67:10000/coser18_qp15_new/", 
			"http://10.15.89.67:10000/coser18_0503_qp0/", 
			"http://10.15.89.67:10000/1015hanfu_qp0/", 
			"http://10.15.89.67:10000/coser18_qp0_new/"
		};
		std::vector<int> video_sh = {
			// 0,
			0,
			0,
			0,
			0,
			0,
			1,
			1,
			3,
			1,
			3,
			3
		};

		std::chrono::milliseconds frameDuration; // 33ms per frame -> 30fps for dynamic play
		std::chrono::high_resolution_clock::time_point lastUpdateTimestamp;

		std::chrono::milliseconds MemframeDuration; // read mem every 1s
		std::chrono::high_resolution_clock::time_point MemlastUpdateTimestamp;

		picojson::object minmax_obj;

		// multi thread helper
		std::vector<std::pair<int, int>> group_frame_index;
		std::vector<std::pair<int, int>> group_name_index;
		int downloaded_frames = -1;
		int downloaded_array[SEQUENCE_LENGTH];
		int ready_frames = -1;
		int ready_array[SEQUENCE_LENGTH];

		bool frame_changed = false;

		// ready queue note that the index is frame index
		std::queue<int> need_ready_q;
		std::mutex mtx_ready;
		std::condition_variable cv_ready;

		// note the index is group index
		std::queue<int> need_download_q;
		std::mutex mtx_download;
		std::condition_variable cv_download;
		std::thread download_thread_;
		std::thread ready_thread_;

		GLuint imageBuffer;
		cudaGraphicsResource_t imageBufferCuda;

		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* background_cuda;

		float _scalingModifier = 1.0f;
		GaussianData* gData;
		GaussianData* gData_array[500];
		// bool _multi_view_play = true;
		bool _multi_view_play = false;
		bool _interop_failed = false;
		std::vector<char> fallback_bytes;
		float* fallbackBufferCuda = nullptr;
		bool accepted = false;


		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< The current scene.
		PointBasedRenderer::Ptr _pointbasedrenderer;
		BufferCopyRenderer* _copyRenderer;
		GaussianSurfaceRenderer* _gaussianRenderer;
	};

} /*namespace sibr*/ 
