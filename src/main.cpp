#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "../include/tracker.hpp"
#include "../include/haar_cascade_detector.hpp"
#include "../include/detector.hpp"
#include "../include/cnn.hpp"
#include "../include/core.hpp"

#include <iostream>

#include <string>
#include <codecvt>
#include <locale>
#include <fstream>

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <cpp/ie_cnn_net_reader.h>

#include "../include/utils.hpp"
#include "../include/mode.hpp"
#include "../include/main.hpp"

//================================================================================
//                              TODO LIST
//================================================================================
// TODO:
// Implement logging (what information needed in logging?)
//1.    Modify main.cpp so executable can accept argument
//2.    For face recognition:
//	person name, timestamp
//3.    For pedestrian recognition:
//	object type, timestamp, trajectory
//4.    Implement trajectory calculation
//================================================================================

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        std::cout << std::endl;
        showAvailableDevices();
        std::cout << std::endl;
        return false;
    }
    return true;
}

int main_work(int argc, char** argv)
{
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }
    cv::CommandLineParser parser(argc, argv,
            "{help -h}");
    parser.about("Starting...\n\n");
    parser.printMessage();

    cv::String face_cascade_name = "/home/pi/opencv_install/share/opencv4/haarcascades/haarcascade_frontalface_alt_tree.xml";

    int camera_device = parser.get<int>("camera");
    std::cout << "Camera index: "<< camera_device << std::endl;

    // Input Parameter
    std::string video_file = FLAGS_video_path;

    // Input Channel Mode
    // int input_mode = FILE_VIDEO_INPUT;
    std::string input_source = FLAGS_input;

    // std::cout << cv::getBuildInformation();

    cv::VideoCapture capture;
    if (input_source == FILE_VIDEO_INPUT) {
            capture.open(video_file);
    }
    else if (input_source == CAMERA_INPUT) {
            //-- 2. Read the video stream
            capture.open(camera_device);
    }
    int frame_limit = capture.get(cv::CAP_PROP_FRAME_COUNT);

    if (!capture.isOpened())
    {
            std::cout << "--(!)Error opening video capture\n";
            return -1;
    }

    //================================================================================
    //Parameters
    //================================================================================
    int max_tracker = FLAGS_max_tracker;
    cv::Mat frame;

    std::string target_device_detector = FLAGS_target_device;
    bool print_performance = FLAGS_print_performance;
    bool save_to_logfile = FLAGS_save_to_logfile;
    bool displaying_frame = FLAGS_displaying_frame;

    bool should_use_perf_counter = FLAGS_use_perf_counter;
    bool recalculate_embedding = FLAGS_recalculate_embedding;
    bool save_video_output = FLAGS_save_video_output;
    float embedding_treshold = FLAGS_embedding_treshold;
    //================================================================================

    std::string reid_mode = target_device_detector;
    std::string custom_cpu_library = "";
    std::string path_to_custom_layers = "";

    // Detection Mode
    // int mode = PEDESTRIAN_DETECTION;
    std::string mode = FLAGS_mode;
    std::string model_used = FLAGS_model;
    bool display_track = FLAGS_display_track;

    std::string photo_reference_dir = "/home/pi/computer_vision/sample_photo";
    std::string embedding_file = "/home/pi/computer_vision/embedding/vector.xml";
    //================================================================================


    //================================================================================
    //Facenet Model IR
    //================================================================================
    std::string facenet_weight =
            "/home/pi/computer_vision/model/ir_facenet/20180408-102900.bin";
    std::string facenet_xml =
            "/home/pi/computer_vision/model/ir_facenet/20180408-102900.xml";
    //================================================================================


    //================================================================================
    //Face Detector Model IR
    //================================================================================
    std::string det_weight;
    std::string det_xml;
    if (mode == FACIAL_RECOGNITION) {
            det_weight =
                    "/home/pi/computer_vision/model/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin";
            det_xml =
                    "/home/pi/computer_vision/model/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
    }
    else if (mode == PEDESTRIAN_DETECTION) {
            if (model_used == PERSON_VEHICLE_BIKE_DETECTION_CROSSROAD_0078) {
                    //MODEL NAME	:person-vehicle-bike-detection-crossroad-0078
                    det_weight =
                            "/home/pi/computer_vision/model/intel/person-vehicle-bike-detection-crossroad-0078/FP16/person-vehicle-bike-detection-crossroad-0078.bin";
                    det_xml =
                            "/home/pi/computer_vision/model/intel/person-vehicle-bike-detection-crossroad-0078/FP16/person-vehicle-bike-detection-crossroad-0078.xml";
            }
            else if (model_used == SSD_MOBILENET) {
                    //MODEL NAME	:ssdlite_mobilenet_v2
                    det_weight =
                            "/home/pi/computer_vision/model/ssd_mobilenet/frozen_inference_graph.bin";
                    det_xml =
                            "/home/pi/computer_vision/model/ssd_mobilenet/frozen_inference_graph.xml";
            }
    }
    //================================================================================

    // Load the Inference Engine
    std::vector<std::string> devices{ target_device_detector, reid_mode };
    InferenceEngine::Core ie =
            LoadInferenceEngine(
                    devices, custom_cpu_library, path_to_custom_layers,
                    should_use_perf_counter);
    std::cout << "Load inference engine" << std::endl;
    std::cout << "Instantiate detector" << std::endl;

    TrackedObjects tracked_obj;

    //================================================================================
    //Instantiate Tracker & Detector & Object Detected
    //================================================================================
    ObjectTrackers tracker(max_tracker);
    DetectorConfig detector_confid(det_xml, det_weight);
    std::cout << det_weight << std::endl;
    std::cout << det_xml << std::endl << std::endl;
    std::cout << "Load detector" << std::endl;
    ObjectDetector detector(detector_confid, ie, target_device_detector);
    //================================================================================

    //================================================================================
    //Instantiate Facenet (Face Recognition)
    //================================================================================
    CnnConfig facenet_config(facenet_xml, facenet_weight);
    std::cout << facenet_weight << std::endl;
    std::cout << facenet_xml << std::endl;
    std::cout << "Load facenet" << std::endl;
    CnnBase facenet(facenet_config, ie, target_device_detector);
    facenet.Load();
    //================================================================================

    //================================================================================
    //Instantiate Hungarian Algorithm Solver to combine detection & tracking object
    KuhnMunkres solver;
    //================================================================================

    TrackedObjects objects;
    TrackedObjects detected_obj;

    int frame_idx = 0;
    int interval = FLAGS_detection_interval;
    double video_fps;
    bool processing = false;
    cv::Mat dissimilarity_mtx;

    //Create reference embedding from photo directory
    if (recalculate_embedding == true && mode == FACIAL_RECOGNITION) {
            std::cout << "Creating embedding";
            createAndWriteEmbedding(photo_reference_dir, embedding_file, facenet, detector);
    }

    //Read person name and it's embedding from file
    cv::Mat embedding_reference;
    std::vector<std::string> name_list;
    std::cout << "Reading embedding";
    cv::FileStorage file(embedding_file, cv::FileStorage::READ);
    cv::FileNode fn = file.root();
    for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit) {
            cv::FileNode item = *fit;
            name_list.push_back(item.name());
            cv::Mat row;
            item >> row;
            embedding_reference.push_back(row);
    }


    int frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)); //get the width of frames of the video
    int frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Size frame_size(frame_width, frame_height);
    int frames_per_second = 30;

    std::string output_filename = "/home/pi/computer_vision/sample_video/output.avi";
    cv::VideoWriter oVideoWriter(output_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            frames_per_second, frame_size, true);

    if (!oVideoWriter.isOpened())
    {
            std::cout << "Cannot save the video to a file" << std::endl;
            std::cin.get(); //wait for any key press
            return -1;
    }

    if (save_to_logfile == true){
            freopen("/home/pi/computer_vision/log/performance_log.txt", "w", stdout);
    }
    while (capture.read(frame))
    {
            if (frame.empty())
            {
                    std::cout << "--(!) No captured frame -- Break!\n";
                    return 1;
            }

            //================================================================================
            //If input come from video, loop the video
            //================================================================================
            int current_frame = capture.get(cv::CAP_PROP_POS_FRAMES);
            int frame_limit = capture.get(cv::CAP_PROP_FRAME_COUNT);
            if (input_source == FILE_VIDEO_INPUT && current_frame >= frame_limit) {
                    capture.set(cv::CAP_PROP_POS_FRAMES, 0);
            }
            //================================================================================

            std::cout << "Frame index: " << frame_idx << std::endl;
            video_fps = capture.get(cv::CAP_PROP_FPS);
            uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frame_idx);

            //For a time interval re-run detection & update tracked object by
            // combine it with tracking bbox using solver
            if (frame_idx % interval == 0) {
                    //In each 10 frame, we detect objects and initiate fresh new tracker
                    //Then track it for 10 more frame

                    //OpenVino Detector submit & fetch
                    std::cout << "Begin" << std::endl;
                    cv::Mat frame_sumbitted;
                    cv::cvtColor(frame, frame_sumbitted, cv::COLOR_BGR2RGB);

                    detector.submitFrame(frame_sumbitted, frame_idx);
                    detector.waitAndFetchResults();
                    std::cout << "Submit frame" << std::endl;

                    //Write performance log for detection process
                    if(print_performance == true){
                            detector.PrintPerformanceCounts("MYRIAD");
                    }

                    //face_detector.updateTrackedObjects(frame, detected_obj, frame_idx);
                    std::cout << "Get detection results" << std::endl;
                    detected_obj = detector.getResults();
                    getRoI(frame, detected_obj);

                    //Get RoI for each tracked Object
                    std::cout << "Getting ROI" << std::endl;
                    if (detected_obj.size() > 0) {
                            if (processing == true) {
                                    dissimilarity_mtx = solver.ComputeDissimilarityMatrix(
                                            tracked_obj, detected_obj);
                                    if (!dissimilarity_mtx.empty()) {
                                            std::vector<int> result = solver.Solve(dissimilarity_mtx);
                                            solver.UpdateAndRemoveNonMatch(result, tracked_obj, detected_obj);
                                            solver.AddNewTrackedObject(result, tracked_obj, detected_obj);
                                    }
                            }
                            tracker.clear();
                            if (tracked_obj.size() > 0) {
                                    //If tracked object not null build tracker using that
                                    for (int i = 0; i < tracked_obj.size();i++) {
                                            tracker.addTracker(frame, tracked_obj[i]);
                                    }
                            }
                            else {
                                    //Else build tracker using detected object and update processing flag
                                    for (int i = 0; i < detected_obj.size();i++) {
                                            tracker.addTracker(frame, detected_obj[i]);
                                    }
                                    processing = true;
                            }
                    }
            }
            //Update tracker
            if (tracked_obj.size() > 0 && processing == true) {
                    std::cout << "Update tracked obj" << std::endl;
                    //If tracked object not null update tracker using that
                    tracked_obj = tracker.updateTrackedObjects(frame, tracked_obj);
            }
            else if (detected_obj.size() > 0) {
                    std::cout << "Update detected obj" << std::endl;
                    std::cout << "Detected object size: " << detected_obj.size() << std::endl;
                    //Else update tracker using detected object and update processing flag
                    tracked_obj = tracker.updateTrackedObjects(frame, detected_obj);
                    processing = true;
            }
            else if (detected_obj.size() == 0) {
                    processing = false;
            }

            //Create embedding vector using Facenet
            if (mode == FACIAL_RECOGNITION && processing == true) {
                    std::cout << "Create embedding vector" << std::endl;
                    updateCommonName(frame, tracked_obj, facenet, embedding_reference,
                            embedding_treshold, name_list);
            }

            //If the VideoWriter object is not initialized successfully, exit the program
            std::cout << "Create frame overlay" << std::endl;
            cv::Mat result_frame = overlay_tracked_obj(frame, tracked_obj, display_track);
            if (displaying_frame == true){
                    imshow("Smart Camera", frame);
            }
            if (save_video_output == true){
                    std::cout << "Saving frame to output video." << std::endl;
                    oVideoWriter.write(result_frame);
            }
            std::cout << video_fps << " FPS" << std::endl;

            if (cv::waitKey(10) == 27)
            {
                    break; // escape
            }
            frame_idx += 1;
    }

    //Close performance log file
    oVideoWriter.release();
    return 0;
}

int main(int argc, char** argv) {
	try {
		main_work(argc, argv);
	}
	catch (const std::exception & error) {
		std::cerr << "[ ERROR ] " << error.what() << std::endl;
		return 1;
	}
	catch (...) {
		std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
		return 1;
	}

	std::cout << "Execution successful" << std::endl;

	return 0;
}