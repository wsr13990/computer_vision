#include <gflags/gflags.h>


static const char help_message[] = "Print a usage message.";
static const char device_message[] = "Optional. Specify the target device for detection Default value is MYRIAD."\
                                        "Use \"-h\" to view all available device";
static const char detection_mode_message[] = "Optional. Specify detection mode to perform. Avaliable detection mode are:" \
                                                "facial_recognition, pedestrian_detection."\
                                                "Default value is facial_recognition. ";
static const char input_source_message[] = "Optional. Specify input source to perform detection from. Available input source are:" \
                                                "camera, file."\
                                                "Default value is camera. ";
static const char model_message[] = "Optional. Specify model to perform detection in pedestrian detection. Available input model are:" \
                                                "person_vehicle_bike_detection_crossroad_0078, ssd_mobilenet."\
                                                "Default value is person_vehicle_bike_detection_crossroad_0078. ";
static const char video_path[] = "Optional. If you are using video file as input, you can specify which video to use by providing it's absolute path" \
                                                "Default value is /home/pi/computer_vision/sample_video/motorcycles.mp4.";

static const char print_performance[] = "Optional. Specify if you whether you want to print model performance or not. Default value is false.";
static const char save_to_logfile[] = "Optional. Specify if you whether you want to save the log or not. Default value is false.";
static const char displaying_frame[] = "Optional. Specify if you whether you want display result detection or not. Default value is true.";
static const char use_perf_counter[] = "Optional. Specify if you whether you want to use performance counter or not. Default value is false.";
static const char recalculate_embedding[] = "Optional. Specify if you whether you want recalculate the embedding reference or not. Default value is true.";
static const char save_video_output[] = "Optional. Specify if you whether you want to save output video. Default value is true.";
static const char display_track[] = "Optional. Specify if you whether you want to display tracker worm or not. Default value is true.";

static const char max_tracker[] = "Optional. Specify the maximum number of tracker. Default value is 20.";
static const char detection_interval[] = "Optional. Specify frame interval between detection. Default value is 10.";
static const char embedding_treshold[] = "Optional. Specify threshold for face recognition. Default value is 1.1.";

DEFINE_bool(h, false, help_message);
DEFINE_string(target_device,"MYRIAD",device_message);
DEFINE_string(mode,"facial_recognition",detection_mode_message);
DEFINE_string(input,"camera",input_source_message);
DEFINE_string(model,"person_vehicle_bike_detection_crossroad_0078",model_message);
DEFINE_string(video_path, "/home/pi/computer_vision/sample_video/motorcycles.mp4", video_path);

DEFINE_bool(print_performance,false,print_performance);
DEFINE_bool(save_to_logfile,false,save_to_logfile);
DEFINE_bool(displaying_frame,true,displaying_frame);
DEFINE_bool(use_perf_counter,false,use_perf_counter);
DEFINE_bool(recalculate_embedding,true,recalculate_embedding);
DEFINE_bool(save_video_output,true,save_video_output);
DEFINE_bool(display_track,true,display_track);

DEFINE_int32(max_tracker,20,max_tracker);
DEFINE_int32(detection_interval,10,detection_interval);
DEFINE_double(embedding_treshold,1.1,embedding_treshold);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "pedestrian_tracker_demo [OPTION]" << std::endl << std::endl;;
    std::cout << "Options:" << std::endl << std::endl;;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -target_device            " << device_message << std::endl;
    std::cout << "    -mode                     " << detection_mode_message << std::endl;
    std::cout << "    -input                    " << input_source_message << std::endl;
    std::cout << "    -model                    " << model_message << std::endl;
    std::cout << "    -video_path \"<absolute_path>\" " << video_path << std::endl;
    std::cout << "    -print_performance        " << print_performance << std::endl;
    std::cout << "    -save_to_logfile          " << save_to_logfile << std::endl;
    std::cout << "    -displaying_frame         " << displaying_frame << std::endl;
    std::cout << "    -use_perf_counter         " << use_perf_counter << std::endl;
    std::cout << "    -recalculate_embedding    " << recalculate_embedding << std::endl;
    std::cout << "    -save_video_output        " << save_video_output << std::endl;
    std::cout << "    -display_track            " << display_track << std::endl;
    std::cout << "    -max_tracker              " << max_tracker << std::endl;
    std::cout << "    -detection_interval       " << detection_interval << std::endl;
    std::cout << "    -embedding_treshold       " << embedding_treshold << std::endl;
}