# computer_vision
Smart Camera Apps.
Current features:
  1. Facial recognition
  2. People, car & bike detection
  3. Draw custom roi & save or load it.

Currently can only run on raspbian os.
To get help & view all available arguments avaliable run "smart_camera -h"
To compile it yourself, you need to edit cmake and add linker with required library such as:
  1. OpenCV with tracker (in case of raspberry device you need Opnecv with Intel Backend)
  2. TBB
  3. Openvino
  4. FFMPEG
