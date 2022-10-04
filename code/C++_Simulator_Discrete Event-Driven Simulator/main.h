#include <algorithm>
#include <string>
#include <vector>
#include <iomanip>
#include <iostream>
#include "mt19937ar.h"
using namespace std;
#define PI 3.1415926

struct call {   //呼叫
	//该实体状态量
	int number;
	char type;
	int state;   //0-空闲,1-处理中
	int time_arrive;
};

struct router {   //路由
	//该实体状态量
	int state;   //0-空闲,1-处理中
	int time_work;
	call call_now;
};

struct receiver {   //接线员
	//该实体状态量
	int state;   //0-空闲,1-处理中
	int time_work;
	call call_now;
};


struct Node {
	int NType;  //事件类型
	int Occurtime;  //事件发生时间
	//事件发生的对象
	call call;
	struct Node* next;
};

struct Node* head = (struct Node*)malloc(sizeof(struct Node));  //创建头结点

class Table {
private:
	//列属性
	vector<string> column_content;
	//列宽
	vector<int> column_size;
	//记录
	vector<vector<string>> record;
public:
	//默认构造函数
	Table() {

	}
	//构造函数1:无转换构造
	Table(vector<string> cc, vector<int> cs, vector<vector<string>> rc) {
		column_content = cc;
		column_size = cs;
		record = rc;
	}
	//构造函数2:自动生成列宽
	Table(vector<string> cc, vector<vector<string>> rc) {
		int cols = cc.size(), rows = rc.size();
		//生成列宽数组
		for (int col = 0; col < cols; col++) {
			int max = cc[col].size();
			for (int row = 0; row < rows; row++) {
				max = rc[row][col].size() > max ? rc[row][col].size() : max;
			}
			column_size.push_back(max);
		}
		column_content = cc;
		record = rc;

	}
	//画分隔线
	void Print_Line() {
		for (int col = 0; col < column_size.size(); col++) {
			cout << "+-";
			for (int i = 0; i <= column_size[col]; i++) {
				cout << '-';
			}
		}
		cout << '+' << endl;
	}
	//打印表
	void Print_Table() {
		Print_Line();
		//表头
		for (int col = 0; col < column_content.size(); col++) {
			cout << "| " << setw(column_size[col]) << setiosflags(ios::left) << setfill(' ') << column_content[col] << ' ';
		}
		cout << '|' << endl;
		//内容
		Print_Line();
		for (int row = 0; row < record.size(); row++) {
			for (int col = 0; col < column_content.size(); col++) {
				cout << "| " << setw(column_size[col]) << setiosflags(ios::left) << setfill(' ');
				cout << record[row][col] << ' ';
			}
			cout << '|' << endl;
		}
		Print_Line();
	}
	~Table() {

	}
};


vector<call>* generate_call_X(vector<int>* call_arrive_time) //根据到达时间数组产生的X_call实体数组
{
	static vector<call> call_list;   //产生call数组
	int N_X = (*call_arrive_time).size();
	for (int i = 0; i < N_X; i++)
	{
		call call;
		call.number = i;
		call.state = 0;  //空闲
		call.time_arrive = (*call_arrive_time)[i];
		call.type = 'X';
		call_list.push_back(call);
	}
	return &call_list;
};

vector<call>* generate_call_Y(vector<int>* call_arrive_time) //根据到达时间数组产生的Y_call实体数组
{
	static vector<call> call_list;   //产生call数组
	int N_Y = (*call_arrive_time).size();
	for (int i = 0; i < N_Y; i++)
	{
		call call;
		call.number = i;
		call.state = 0;  //空闲
		call.time_arrive = (*call_arrive_time)[i];
		call.type = 'Y';
		call_list.push_back(call);
	}
	return &call_list;
};


int generate_work_time(int average, int d)
{

	unsigned long init[4] = { 0x123, 0x234, 0x345, 0x456 }, length = 4;
	init_by_array(init, length);

	double a = genrand_real3();
	double b = genrand_real3();
	double k = (sqrt((-2) * log(a)) * cos(2 * PI * b)) * d + average;
	return (int)(k * 10);
}; //产生一个路由或接线员的处理时间


vector<vector<int>>* generate_call_arrive_time(double lambda1, double lambda2, int numX, int numY)
{//产生服从参数为lambda的指数分布的呼叫到达时间数组
	srand(time(NULL));
	unsigned long init[4] = { rand(), 0x534, 0x345, 0x456 }, length = 4;
	init_by_array(init, length);

	vector<int> arrive_time_X;
	vector<int> arrive_time_Y;
	static vector<vector<int>> arrive_time_all;
	int x_t = 0;
	int y_t = 0;
	for (int i = 0; i < numX; i++) //产生numX个样本
	{
		double u = genrand_real3();
		double x = -(1.0 / (double)lambda1) * log(u);
		x_t = x_t + (int)(x * 20);
		arrive_time_X.push_back(x_t);
	}
	for (int i = 0; i < numY; i++) //产生numY个样本
	{
		double u = genrand_real3();
		double x = -(1.0 / (double)lambda2) * log(u);
		y_t = y_t + (int)(x * 20);
		arrive_time_Y.push_back(y_t);
	}
	arrive_time_all.push_back(arrive_time_X);
	arrive_time_all.push_back(arrive_time_Y);
	return &arrive_time_all;
};

//把a和b两个call容器按照到达时间arrive_time按照一定规则排序
//1.a始终在前面
//2.b始终在前面
bool t_comp1(const call& a, const call& b) {
	return a.time_arrive < b.time_arrive;
}

bool t_comp2(const call& a, const call& b) {
	if (a.time_arrive == b.time_arrive) return a.type > b.type;
	else return a.time_arrive < b.time_arrive;
}

vector<call> merge(vector<call>* a, vector<call>* b, int numA, int numB, int mode)
{
	vector<call> temp;
	for (int i = 0; i < numA; i++) {
		temp.push_back((*a)[i]);
	}
	for (int j = 0; j < numB; j++) {
		temp.push_back((*b)[j]);
	}
	switch (mode) {
	case 1:
		sort(temp.begin(), temp.end(), t_comp1);
		break;
	case 2:
		sort(temp.begin(), temp.end(), t_comp2);
		break;
	default: break;
	}
	return temp;
};



vector<string> create_rc_row(string TYPE, int clock, vector<call> call_all, router router, receiver* receiver, vector<call> router_list, vector<call> receiver_X_list, vector<call> receiver_Y_list)
{
	vector<string> rc_row;
	//第一列，输出当前阶段
	rc_row.push_back(TYPE);

	//第二列，输出当前时间
	rc_row.push_back(to_string(clock));

	//第三列，输出路由队列
	if (!router_list.empty())
	{
		string router_list_out = "";
		int N_router = router_list.size();
		for (int i = 0; i < N_router; i++)
		{
			if (i != 0) router_list_out.append(",");
			string type(1, router_list[i].type);
			string number = to_string(router_list[i].number + 1);
			router_list_out.append(type);
			router_list_out.append(number);
		}
		rc_row.push_back(router_list_out);
	}
	else
		rc_row.push_back("无");

	//第四列，输出路由状态
	if (router.state == 0)
		rc_row.push_back("空闲");
	else
	{
		string router_call_now;
		string type(1, router.call_now.type);
		string number = to_string(router.call_now.number + 1);
		router_call_now.append(type);
		router_call_now.append(number);
		rc_row.push_back(router_call_now);
	}

	//第五列，输出接线员1队列
	if (!receiver_X_list.empty())
	{
		string receiver_X_list_out;
		int N_receiver1 = receiver_X_list.size();
		for (int i = 0; i < N_receiver1; i++)
		{
			if (i != 0) receiver_X_list_out.append(",");
			string type(1, receiver_X_list[i].type);
			string number = to_string(receiver_X_list[i].number + 1);
			receiver_X_list_out.append(type);
			receiver_X_list_out.append(number);
		}
		rc_row.push_back(receiver_X_list_out);
	}
	else
		rc_row.push_back("无");

	//第六列，输出接线员1状态
	if (receiver[0].state == 0)
		rc_row.push_back("空闲");
	else
	{
		string receiver_X_call_now;
		string type(1, receiver[0].call_now.type);
		string number = to_string(receiver[0].call_now.number + 1);
		receiver_X_call_now.append(type);
		receiver_X_call_now.append(number);
		rc_row.push_back(receiver_X_call_now);
	}

	//第七列，输出接线员2队列
	if (!receiver_Y_list.empty())
	{
		string receiver_Y_list_out;
		int N_reicever2 = receiver_Y_list.size();
		for (int i = 0; i < N_reicever2; i++)
		{
			if (i != 0) receiver_Y_list_out.append(",");
			string type(1, receiver_Y_list[i].type);
			string number = to_string(receiver_Y_list[i].number + 1);
			receiver_Y_list_out.append(type);
			receiver_Y_list_out.append(number);
		}
		rc_row.push_back(receiver_Y_list_out);
	}
	else
		rc_row.push_back("无");

	//第八列，输出接线员2状态
	if (receiver[1].state == 0)
		rc_row.push_back("空闲");
	else
	{
		string receiver_Y_call_now;
		string type(1, receiver[1].call_now.type);
		string number = to_string(receiver[1].call_now.number + 1);
		receiver_Y_call_now.append(type);
		receiver_Y_call_now.append(number);
		rc_row.push_back(receiver_Y_call_now);
	}
	return rc_row;
}

void my_print(vector<vector<string>> rc, int number_finish_X, int number_finish_Y)
{
	vector<string> cc = { "阶段","当前时间","路由队列","路由","接线员1队列","接线员1","接线员2队列","接线员2" };  //列属性
	Table my_table(cc, rc);
	my_table.Print_Table();

	cout << "当前已完成X呼叫个数：" << number_finish_X << endl;
	cout << "当前已完成Y呼叫个数：" << number_finish_Y << endl;
}