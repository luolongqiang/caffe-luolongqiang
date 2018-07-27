#include <iostream>
#include <string>
#include <vector>
#include <set>

void maJiang4(std::string src, bool& mark, std::set<char>& res){
	if(src[0] == src[1] && src[1] == src[2] && src[2] != src[3]){ // AAAB
		res.insert(src[3]);
		if(src[2] + 1 == src[3]){
			if(src[2]-1 >= '1'){
				res.insert(src[2]-1);
			}
			if(src[3]+1 <= '9'){
				res.insert(src[3]+1);
			}
		}
		mark = false;
	}
	else if(src[0] != src[1] && src[1] == src[2] && src[2] == src[3]){ // ABBB
		res.insert(src[0]);
		if(src[0] + 1 == src[1]){
			if(src[0]-1 >= '1'){
				res.insert(src[0]-1);
			}
			if(src[1]+1 <= '9'){
				res.insert(src[1]+1);
			}
		}
		mark = false;
	}
	else if(src[0] == src[1] and src[2] + 1 == src[3]){ // AABC
		res.insert(src[2]-1);
		if(src[3]+1 <= '9'){
			res.insert(src[3]+1);
		}
		mark = false;
	} 
	else if(src[0] + 1 == src[1] and src[2] == src[3]){ // ABCC
		res.insert(src[1]+1);
		if(src[0]-1 >= '1'){
			res.insert(src[0]-1);
		}
		mark = false;
	}
	else if(src[0] == src[1] && src[2]+2 == src[3]){ // AABD
		res.insert(src[2]+1);
		mark = false;
	}	
	else if(src[0]+2 == src[1] && src[2] == src[3]){ // ACDD
		res.insert(src[0]+1);
		mark = false;
	}
	else if(src[1]+1 == src[2]) { // ABCD
		if(src[0]+1 == src[1]){
			res.insert(src[3]);
			mark = false;
		}
		if(src[2]+1 == src[3]){
			res.insert(src[0]);
			mark = false;
		}
	}
}

void maJiangQingYiSe(std::string src, bool& mark, std::set<char>& res){
	int num = src.size();
	if(num == 1){
		res.insert(src[0]);
		mark = false;
		return;
	}
	if(num == 4){
		maJiang4(src, mark, res);
		return;
	}
	for(int i = 0; i < num - 2; ++i){
		if((src[i]==src[i+1] && src[i+1]==src[i+2]) || 
		   (src[i]+2==src[i+1]+1 && src[i+1]+1==src[i+2])){
		   	//std::cout<<src.substr(0,i)+src.substr(i+3)<<std::endl;
			maJiangQingYiSe(src.substr(0,i)+src.substr(i+3), mark, res);
			mark = false;
		}
	}	
}

int main(){
	std::string src = "1123445668";
	if(src.size() >= 4){
		for(int i = 0; i < src.size() - 3; ++i){
			if(src[i]   == src[i+1] &&
			   src[i+1] == src[i+2] &&
			   src[i+2] == src[i+3]){
			   std::cout<<0<<std::endl;
			   return 0;
			}
		}		
	}
	bool mark = true;
	std::set<char> res;
	maJiangQingYiSe(src, mark, res);
	if(mark){
		std::cout<<0<<std::endl;
	}
	else{
		std::set<char>::iterator it;
		for(it = res.begin(); it != res.end(); ++it){
			std::cout<<*it<<" ";
		}		
	}
	return 0;
}
