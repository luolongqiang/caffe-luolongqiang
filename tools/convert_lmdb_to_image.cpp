//Usage:
//  convert_lmdb_to_image DB_NAME ROOTFOLDER

#include <fstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;
using boost::shared_ptr;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
  
  LOG(INFO) << "test";
  shared_ptr<db::DB> db(db::GetDB("lmdb"));
  db->Open(argv[1], db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  
  int i = 0;
  char label_txt[256];
  sprintf(label_txt, "%s/label.txt", argv[2]);
  std::ofstream fout(label_txt);
  LOG(INFO) << "test";
  
  while (cursor->valid()) {
    Datum datum;
    datum.ParseFromString(cursor->value());
    const string& data = datum.data();
    std::vector<unsigned char> vec_data(data.c_str(), data.c_str() + data.size());

    cv::Mat cv_img(datum.height(), datum.width(), CV_8UC3);
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < datum.height(); ++h) {
        for (int w = 0; w < datum.width(); ++w) {
          int data_index = (c * datum.height() + h) * datum.width() + w;
          cv_img.at<uchar>(h, w*3+c) = static_cast<uint8_t>(data[data_index]);
        }
      }
    }
    char image_path[256];
    sprintf(image_path, "%s/%d.bmp", argv[2], i);
    cv::imwrite(image_path, cv_img);
    fout << image_path << " " << datum.label() << std::endl;
    cursor->Next();
    i++;
  }
  
  fout.close();
  return 0;
}