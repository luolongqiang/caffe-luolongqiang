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

//##################################################################################
//##################################################################################

#include <iostream>
#include <stack>
#include <vector>
#include <string>

bool isRightOutputOfStack(std::string src, std::string target){
	int num = src.size();
	if(num == 0){
		return true;	
	}
	if(num == 1){
	 	if(src[0] == target[0])
			return true;
		else
			return false;
	}
	int index = target.find(src[0]);
	std::string src1 = src.substr(1, index);
	std::string src2 = src.substr(index+1);
	std::string target1 = target.substr(0, index);
	std::string target2 = target.substr(index+1);
	bool mark1 = isRightOutputOfStack(src1, target1);
	bool mark2 = isRightOutputOfStack(src2, target2);
	return mark1&&mark2;
}

int main(){
	std::string src = "abcd";
	std::string target = "adbc";
	bool mark = isRightOutputOfStack(src, target); 
	if(mark)
		std::cout<<"true"<<std::endl;
	else
		std::cout<<"false"<<std::endl;
	return 0;	
}

/*
char stackIn[]    = {'a', 'b', 'c', 'd'};
char stackOut1[]  = {'a', 'd', 'c', 'b'};
char stackOut2[]  = {'a', 'c', 'd', 'b'};
char stackOut3[]  = {'a', 'c', 'b', 'd'};
char stackOut4[]  = {'a', 'b', 'c', 'd'};
char stackOut5[]  = {'a', 'b', 'd', 'c'};
char stackOut6[]  = {'b', 'a', 'c', 'd'};
char stackOut7[]  = {'b', 'a', 'd', 'c'};
char stackOut8[]  = {'c', 'b', 'a', 'd'};
char stackOut9[]  = {'b', 'c', 'a', 'd'};
char stackOut10[] = {'d', 'c', 'b', 'a'};
char stackOut11[] = {'c', 'd', 'b', 'a'};
char stackOut12[] = {'c', 'b', 'd', 'a'};
char stackOut13[] = {'b', 'c', 'd', 'a'};
char stackOut14[] = {'b', 'd', 'c', 'a'};
*/

//##################################################################################
//##################################################################################

#include <iostream>
#include <map>

int FindMaxLenSeries(int a[], int num){
	std::map<int, bool> mark;
	for(int i = 0; i < num; ++i){
		mark.insert(std::pair<int, bool>(a[i], true));
	}	
	int max = 1, temp1, temp2, count;
	for(int i = 0; i < num; ++i){
		count = 1;
		temp1 = temp2 = a[i];
		while(mark[temp1] || mark[temp2]){
			mark[temp1] = false;
			mark[temp2] = false;
			if(mark[temp1-1]){
				--temp1;
				++count; 
			}
			if(mark[temp2+1]){
				++temp2;
				++count;
			}			
		}
		if(count > max){
			max = count;
		}
	}
	return max;
}

int main(){
	int num = 8;
	int a[] = {100, 4, 200, 1, 3, 2, 5, 6};
	std::cout<<FindMaxLenSeries(a, num)<<std::endl;
	return 0;
}

//##################################################################################
//##################################################################################
