#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <iomanip>	   
#include <iostream> 
#include "hdf5.h"
#include <math.h>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/io.hpp"																														 
#include "caffe/test/test_caffe_main.hpp"
#include <time.h>

int global_top_scale = 0;
int global_bias_scale = 0;

namespace caffe {
//add start	
int forward_num = 0;
int snapshot_num = 400;
char buf[128]= {0}; 
time_t t;
tm* gmt;
string str_time;	
int bottom_scale = 3; 
std::map<string, int>scale_weight_map;
std::map<string, int>scale_bias_map;

template <typename Dtype>			   
void find_quantitive_scale(const Dtype*f, int n, int bit_num, int& scale, 
			std::ofstream& scale_file, const string& source_layer_name)
{
	int k = 0;	
	Dtype maxabs = f[0];
	for(int i = 1;  i < n ; i++)
	{
		if(maxabs < fabs(f[i]))
		{
			maxabs = fabs(f[i]);
			k = i;
		}		
	}
	if(f[k] != 0)
		scale = floor( log10(pow(2, bit_num-1)/maxabs) / log10(2) );
	else
		scale = 1;
	scale_file << source_layer_name <<" , decisive value:"
	           << f[k] <<" , scale:"<< scale <<"\n";
}

template <typename Dtype>			   
void find_max_min_maxabs_minabs(const Dtype*f, int n, 
		Dtype& max, Dtype& min, Dtype& maxabs, Dtype& minabs)
{
	min = f[0];
	minabs = f[0];
	max = f[0];
	maxabs = f[0];
	
	for(int i = 1;  i < n ; i++)
	{
		if(min > f[i])
			min = f[i];
		if(minabs > fabs(f[i]))
			minabs = fabs(f[i]);	
		if(max < f[i])
			max = f[i];
		if(maxabs < fabs(f[i]))
			maxabs = fabs(f[i]);		
	}
}

template <typename Dtype>
void scale_input_img_data(Dtype*f, int n, int bit_num)
{
	for(int i = 0;  i < n ; i++)
	{
		if(f[i] > pow(2, bit_num-1)-1)
			f[i] = pow(2, bit_num-1)-1;
		else if(f[i] < -pow(2, bit_num-1))
			f[i] = -pow(2, bit_num-1);
		else
			f[i] = round(f[i]);
	}
}

template <typename Dtype>
void scale_weight_bias_data(Dtype*f, int n, int scale, int bit_num)
{
	Dtype base= pow(2, scale);
	for(int i = 0;  i < n ; i++)
	{
		f[i] *= base;
		if(f[i] > pow(2, bit_num-1)-1)
			f[i] = pow(2, bit_num-1)-1;
		else if(f[i] < -pow(2, bit_num-1))
			f[i] = -pow(2, bit_num-1);
		else
			f[i] = round(f[i]);
	}
}

template <typename Dtype>	
void top_data_protect(Dtype*f, int n, int bit_num)
{	
	for(int i = 0;  i < n ; i++)
	{		
		if(f[i] > pow(2, bit_num-1)-1)
			f[i] = pow(2, bit_num-1)-1;
		else if(f[i] < -pow(2, bit_num-1))
			f[i] = -pow(2, bit_num-1);
		else
			f[i] = round(f[i]);	
	}
}

template <typename Dtype>
void scale_output_bias_data(Dtype*f, int n, int scale)
{	
	Dtype base= pow(2, scale);
	for(int i = 0;  i < n ; i++)
		f[i] *= base;		
}	
						
template <typename Dtype>
void quantize_data_print(string weight_or_FM, string before_or_after, 
		const string& source_layer_name, const Dtype *data_con, 
		int data_num, std::ofstream& data_file)
{
	Dtype max = data_con[0];
	Dtype min = data_con[0];
	Dtype maxabs = data_con[0];
	Dtype minabs = data_con[0];
	find_max_min_maxabs_minabs(data_con, data_num, max, min, maxabs, minabs);		
	data_file<<source_layer_name<<" No."<<forward_num<<"Forward \n";
	data_file<<source_layer_name<<" "<<weight_or_FM<<", "
	         <<before_or_after<<" quantized, max    ="<<max<<"\n";
	data_file<<source_layer_name<<" "<<weight_or_FM<<", "
	         <<before_or_after<<" quantized, min    ="<<min<<"\n";				
	data_file<<source_layer_name<<" "<<weight_or_FM<<", "
	         <<before_or_after<<" quantized, maxabs="<<maxabs<<"\n";
	data_file<<source_layer_name<<" "<<weight_or_FM<<", "
	         <<before_or_after<<" quantized, minabs="<<minabs<<"\n";
	for(int k = 0; k < 10; k++)
	{
		data_file<< std::setw(15)<<data_con[k];
		if(0 == (k+1)%5)
			data_file << "\n";
	}
	data_file<< "\n";				 
}										
/* ===added end=== */													

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
		  
	CHECK_GE(start, 0);
	CHECK_LT(end, layers_.size());
	Dtype loss = 0;
	
	/* ===added start=== */
	int bit_num = 8;
	
	const Dtype *data_con = NULL;
	Dtype *data_mut = NULL;
	int data_num = 0;
	int target_layer_id = 0;
	
	string file_path = "log/"+str_time+"_FM_data"+  +".txt";
	std::ofstream data_file(file_path.c_str(), ios::app);

	data_file.setf(ios::fixed, ios::floatfield);
	data_file.precision(8);
	data_file.setf(ios::right);
	
	LOG(INFO) << "Forward " << forward_num++;

	/* ===added end=== */	
		
	for (int i = start; i <= end; ++i) {
		// LOG(ERROR) << "Forwarding " << layer_names_[i];
		target_layer_id = 0;
		while (layer_names_[target_layer_id] != layer_names_[i]) {
			++target_layer_id;
		}
		
	    /* ===added start=== */			
		if(layer_names_[i].find("conv")     != layer_names_[i].npos  
		 &&layer_names_[i].find("split")    == layer_names_[i].npos 
		 &&layer_names_[i].find("relu")     == layer_names_[i].npos 
		 &&layer_names_[i].find("flat")     == layer_names_[i].npos 
		 &&layer_names_[i].find("sum")      == layer_names_[i].npos 
		 &&layer_names_[i].find("perm")     == layer_names_[i].npos 
		 &&layer_names_[i].find("priorbox") == layer_names_[i].npos)
		{
			data_con = bottom_vecs_[i][0]->cpu_data();
			data_mut = bottom_vecs_[i][0]->mutable_cpu_data();
			data_num = bottom_vecs_[i][0]->count();	
			
			if(forward_num == 1 || forward_num%snapshot_num == 0)
				quantize_data_print("bottom", "before", layer_names_[i], data_con, data_num, data_file);
				
			if(layer_names_[i].find("conv1") != layer_names_[i].npos)
			{
				scale_input_img_data(data_mut, data_num, bit_num);
				global_top_scale  = scale_weight_map[layer_names_[i]]-bottom_scale;
				global_bias_scale = scale_weight_map[layer_names_[i]]-bottom_scale;
			}
			else if(layer_names_[i].find("mbox") != layer_names_[i].npos
				 &&(layer_names_[i].find("loc")  != layer_names_[i].npos
				  ||layer_names_[i].find("conf") != layer_names_[i].npos))
			{
				global_top_scale  = bottom_scale;
				global_bias_scale = bottom_scale;
			}	
			else
			{
				global_top_scale  = scale_weight_map[layer_names_[i]];
				global_bias_scale = scale_weight_map[layer_names_[i]];				
			}
				
        	if(forward_num == 1 || forward_num%snapshot_num == 0)
				quantize_data_print("bottom", "after", layer_names_[i], data_con, data_num, data_file);				
		}
	    /* ===added end=== */	
	
		Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
		
	    /* ===added start=== */	
		if(layer_names_[i].find("conv")     != layer_names_[i].npos 
		 &&layer_names_[i].find("mbox")     == layer_names_[i].npos 
		 &&layer_names_[i].find("split")    == layer_names_[i].npos 
		 &&layer_names_[i].find("relu")     == layer_names_[i].npos 
		 &&layer_names_[i].find("flat")     == layer_names_[i].npos 
		 &&layer_names_[i].find("perm")     == layer_names_[i].npos 
		 &&layer_names_[i].find("priorbox") == layer_names_[i].npos)
		{
			data_con = top_vecs_[i][0]->cpu_data();
			data_mut = top_vecs_[i][0]->mutable_cpu_data();
			data_num = top_vecs_[i][0]->count();
			
			if(forward_num == 1 || forward_num%snapshot_num == 0)			
				quantize_data_print("top","before", layer_names_[i], data_con, data_num, data_file);	
				
			top_data_protect(data_mut, data_num, bit_num);
				
            if(forward_num == 1 || forward_num%snapshot_num == 0)	
				quantize_data_print("top","after", layer_names_[i], data_con, data_num, data_file);
		}
		/* ===added end=== */
					   			 
	    loss += layer_loss;
	    if (debug_info_) { ForwardDebugInfo(i); }
	}
  return loss;
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
	/* ===added start=== */
	int weight_bit_num = 8;
	int bias_bit_num = 16;
				   
	const Dtype *data_con = NULL;
	Dtype *data_mut = NULL;
	int data_num = 0;
	int num_axes = 0;

	/* config output log file */
	t = time(NULL); 
	gmt = gmtime(&t);
	strftime(buf, 64, "%Y_%m_%d-%H:%M:%S", gmt); 
	str_time = buf;
	
	string file_path = "log/" + str_time + "_model_data.txt";
	std::ofstream data_file(file_path.c_str(), ios::app);
	data_file.setf(ios::fixed, ios::floatfield); 
	data_file.precision(8); 
	data_file.setf(ios::right);	
	
	string scale_path = "log/" + str_time + "_scale_data.txt";
	std::ofstream scale_file(scale_path.c_str(),ios::app);
	scale_file.setf(ios::fixed, ios::floatfield); 
	scale_file.precision(8); 
	scale_file.setf(ios::right);	
	/* ===added end=== */				
					  
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
	  
      /* ===added start=== */
	  if(source_layer_name.find("conv")       != source_layer_name.npos 		  
		&& source_layer_name.find("split")    == source_layer_name.npos 
		&& source_layer_name.find("relu")     == source_layer_name.npos 
		&& source_layer_name.find("flat")     == source_layer_name.npos 
		&& source_layer_name.find("sum")      == source_layer_name.npos 
		&& source_layer_name.find("perm")     == source_layer_name.npos 
		&& source_layer_name.find("priorbox") == source_layer_name.npos)			
		{	
			data_num = target_blobs[j]->count();
	  		data_mut = target_blobs[j]->mutable_cpu_data();
			data_con = target_blobs[j]->cpu_data();
			num_axes = target_blobs[j]->num_axes();
			
			if(num_axes != 1)
			{		  			
				quantize_data_print("weight", "before", source_layer_name, data_con, data_num, data_file);
				
				if(source_layer_name.find("mbox") == source_layer_name.npos)
				{
					scale_weight_map[source_layer_name] = 1;
					find_quantitive_scale(data_mut, data_num, weight_bit_num, 
						scale_weight_map[source_layer_name], scale_file, source_layer_name);
					scale_weight_bias_data(data_mut, data_num, scale_weight_map[source_layer_name], weight_bit_num);							
				}			
				
				quantize_data_print("weight", "after", source_layer_name, data_con, data_num, data_file); 
			}
			
			if(num_axes == 1)
			{				
				quantize_data_print("bias", "before", source_layer_name, data_con, data_num, data_file);
							
				if(source_layer_name.find("conv1") != source_layer_name.npos)
				{
					scale_bias_map[source_layer_name] = scale_weight_map[source_layer_name];
					scale_weight_bias_data(data_mut, data_num, scale_bias_map[source_layer_name], bias_bit_num);
				}	
				else if(source_layer_name.find("mbox") != source_layer_name.npos)
				{
					scale_bias_map[source_layer_name] = bottom_scale;
					scale_output_bias_data(data_mut, data_num, scale_bias_map[source_layer_name]);
				}
				else
				{
					scale_bias_map[source_layer_name] = scale_weight_map[source_layer_name] + bottom_scale;
					scale_weight_bias_data(data_mut, data_num, scale_bias_map[source_layer_name], bias_bit_num);
				}						
					
				quantize_data_print("bias", "after", source_layer_name, data_con, data_num, data_file);
			}
		}
	   /* ===added end=== */
    }
  }
}
}  // namespace caffe
