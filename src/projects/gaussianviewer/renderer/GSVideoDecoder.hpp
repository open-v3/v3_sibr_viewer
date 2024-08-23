// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <opencv2/opencv.hpp>

// extern "C" {
// #include <libavformat/avformat.h>
// #include <libavcodec/avcodec.h>
// #include <libavutil/imgutils.h>
// #include <libswscale/swscale.h>
// #include <libavutil/log.h>
// }

// bool FFmpeggetAllFrames(const std::string& filename, std::vector<cv::Mat>& frames) {
//     auto start = std::chrono::high_resolution_clock::now();
//     avformat_network_init();
//     AVFormatContext* formatCtx = avformat_alloc_context();
//     if (avformat_open_input(&formatCtx, filename.c_str(), NULL, NULL) != 0) {
//         std::cerr << "Cannot open input video file" << std::endl;
//         return false;
//     }
//     if (avformat_find_stream_info(formatCtx, NULL) < 0) {
//         std::cerr << "Cannot find stream information" << std::endl;
//         avformat_close_input(&formatCtx);
//         return false;
//     }
//     int videoStreamIndex = -1;
//     for (unsigned i = 0; i < formatCtx->nb_streams; i++) {
//         if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
//             videoStreamIndex = i;
//             break;
//         }
//     }
//     if (videoStreamIndex == -1) {
//         std::cerr << "No video stream found" << std::endl;
//         avformat_close_input(&formatCtx);
//         return false;
//     }
//     AVCodecParameters* codecParams = formatCtx->streams[videoStreamIndex]->codecpar;
//     AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
//     if (!codec) {
//         std::cerr << "Unsupported codec!" << std::endl;
//         avformat_close_input(&formatCtx);
//         return false;
//     }
//     AVCodecContext* codecCtx = avcodec_alloc_context3(codec);
//     avcodec_parameters_to_context(codecCtx, codecParams);
//     avcodec_open2(codecCtx, codec, NULL);

//     AVFrame* frame = av_frame_alloc();
//     AVPacket* packet = av_packet_alloc();

//     auto end = std::chrono::high_resolution_clock::now();
// 	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
// 	std::cout << "Download and init ffmpeg Elapsed time: " << elapsed.count() << " ms" << std::endl;

//     start = std::chrono::high_resolution_clock::now();

//     while (av_read_frame(formatCtx, packet) >= 0) {
//         if (packet->stream_index == videoStreamIndex) {
//             avcodec_send_packet(codecCtx, packet);
//             while (avcodec_receive_frame(codecCtx, frame) == 0) {
//                 SwsContext* swsCtx = sws_getContext(frame->width, frame->height, codecCtx->pix_fmt,
//                                                     frame->width, frame->height, AV_PIX_FMT_GRAY8,
//                                                     SWS_BILINEAR, NULL, NULL, NULL);
//                 uint8_t* dest[4] = { nullptr };
//                 int dest_linesize[4] = { 0 };
//                 int align = 32;  // This can be set to 16 if 32 is more than what is needed, but 32 is typically safer for modern use.
//                 if (av_image_alloc(dest, dest_linesize, frame->width, frame->height, AV_PIX_FMT_GRAY8, align) < 0) {
//                     std::cerr << "Could not allocate image buffer." << std::endl;
//                     return false;
//                 }
//                 sws_scale(swsCtx, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);
//                 cv::Mat img(frame->height, frame->width, CV_8UC1, dest[0], dest_linesize[0]);
//                 frames.push_back(img.clone());
//                 av_freep(&dest[0]);
//                 sws_freeContext(swsCtx);
//             }
//         }
//         av_packet_unref(packet);
//     }

//     end = std::chrono::high_resolution_clock::now();
// 	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
// 	std::cout << "Decode Elapsed time: " << elapsed.count() << " ms" << std::endl;

//     av_frame_free(&frame);
//     av_packet_free(&packet);
//     avcodec_close(codecCtx);
//     avcodec_free_context(&codecCtx);
//     avformat_close_input(&formatCtx);
//     avformat_network_deinit();
//     return true;
// }

#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class GSVideoDecoder {
public:
    GSVideoDecoder() : formatCtx(nullptr), codecCtx(nullptr), frame(nullptr), packet(nullptr), videoStreamIndex(-1) {
        avformat_network_init();
    }

    ~GSVideoDecoder() {
        cleanup();
        avformat_network_deinit();
    }

    bool open(const std::string& filename) {
        formatCtx = avformat_alloc_context();
        if (avformat_open_input(&formatCtx, filename.c_str(), NULL, NULL) != 0) {
            std::cerr << "Cannot open input video file" << std::endl;
            return false;
        }
        if (avformat_find_stream_info(formatCtx, NULL) < 0) {
            std::cerr << "Cannot find stream information" << std::endl;
            cleanup();
            return false;
        }
        for (unsigned i = 0; i < formatCtx->nb_streams; i++) {
            if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStreamIndex = i;
                break;
            }
        }
        if (videoStreamIndex == -1) {
            std::cerr << "No video stream found" << std::endl;
            cleanup();
            return false;
        }
        AVCodecParameters* codecParams = formatCtx->streams[videoStreamIndex]->codecpar;
        AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
        if (!codec) {
            std::cerr << "Unsupported codec!" << std::endl;
            cleanup();
            return false;
        }
        codecCtx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(codecCtx, codecParams);
        avcodec_open2(codecCtx, codec, NULL);

        frame = av_frame_alloc();
        packet = av_packet_alloc();
        return true;
    }

    bool getAllFrames(std::vector<cv::Mat>& frames) {
        if (!codecCtx || !frame || !packet) {
            std::cerr << "Decoder not properly initialized." << std::endl;
            return false;
        }

        while (av_read_frame(formatCtx, packet) >= 0) {
            if (packet->stream_index == videoStreamIndex) {
                avcodec_send_packet(codecCtx, packet);
                while (avcodec_receive_frame(codecCtx, frame) == 0) {
                    cv::Mat img = convertFrameToMat(frame);
                    if (!img.empty()) {
                        frames.push_back(img);
                    }
                }
            }
            av_packet_unref(packet);
        }
        return true;
    }

private:
    cv::Mat convertFrameToMat(AVFrame* frame) {
        SwsContext* swsCtx = sws_getContext(frame->width, frame->height, codecCtx->pix_fmt,
                                            frame->width, frame->height, AV_PIX_FMT_GRAY8,
                                            SWS_BILINEAR, NULL, NULL, NULL);
        if (!swsCtx) {
            std::cerr << "Could not initialize the conversion context\n";
            return cv::Mat();
        }
        uint8_t* dest[4] = { nullptr };
        int dest_linesize[4] = { 0 };
        av_image_alloc(dest, dest_linesize, frame->width, frame->height, AV_PIX_FMT_GRAY8, 32);
        sws_scale(swsCtx, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);
        cv::Mat img(frame->height, frame->width, CV_8UC1, dest[0], dest_linesize[0]);
        cv::Mat result = img.clone();
        av_freep(&dest[0]);
        sws_freeContext(swsCtx);
        return result;
    }

    void cleanup() {
        if (packet) {
            av_packet_free(&packet);
            packet = nullptr;
        }
        if (frame) {
            av_frame_free(&frame);
            frame = nullptr;
        }
        if (codecCtx) {
            avcodec_close(codecCtx);
            avcodec_free_context(&codecCtx);
            codecCtx = nullptr;
        }
        if (formatCtx) {
            avformat_close_input(&formatCtx);
            avformat_free_context(formatCtx);
            formatCtx = nullptr;
        }
    }

    AVFormatContext* formatCtx;
    AVCodecContext* codecCtx;
    AVFrame* frame;
    AVPacket* packet;
    int videoStreamIndex;
};
