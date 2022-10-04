#include <algorithm>
#include <string>
#include <vector>
#include <iomanip>
#include <iostream>
#include "mt19937ar.h"
using namespace std;
#define PI 3.1415926

struct call {   //����
	//��ʵ��״̬��
	int number;
	char type;
	int state;   //0-����,1-������
	int time_arrive;
};

struct router {   //·��
	//��ʵ��״̬��
	int state;   //0-����,1-������
	int time_work;
	call call_now;
};

struct receiver {   //����Ա
	//��ʵ��״̬��
	int state;   //0-����,1-������
	int time_work;
	call call_now;
};


struct Node {
	int NType;  //�¼�����
	int Occurtime;  //�¼�����ʱ��
	//�¼������Ķ���
	call call;
	struct Node* next;
};

struct Node* head = (struct Node*)malloc(sizeof(struct Node));  //����ͷ���

class Table {
private:
	//������
	vector<string> column_content;
	//�п�
	vector<int> column_size;
	//��¼
	vector<vector<string>> record;
public:
	//Ĭ�Ϲ��캯��
	Table() {

	}
	//���캯��1:��ת������
	Table(vector<string> cc, vector<int> cs, vector<vector<string>> rc) {
		column_content = cc;
		column_size = cs;
		record = rc;
	}
	//���캯��2:�Զ������п�
	Table(vector<string> cc, vector<vector<string>> rc) {
		int cols = cc.size(), rows = rc.size();
		//�����п�����
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
	//���ָ���
	void Print_Line() {
		for (int col = 0; col < column_size.size(); col++) {
			cout << "+-";
			for (int i = 0; i <= column_size[col]; i++) {
				cout << '-';
			}
		}
		cout << '+' << endl;
	}
	//��ӡ��
	void Print_Table() {
		Print_Line();
		//��ͷ
		for (int col = 0; col < column_content.size(); col++) {
			cout << "| " << setw(column_size[col]) << setiosflags(ios::left) << setfill(' ') << column_content[col] << ' ';
		}
		cout << '|' << endl;
		//����
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


vector<call>* generate_call_X(vector<int>* call_arrive_time) //���ݵ���ʱ�����������X_callʵ������
{
	static vector<call> call_list;   //����call����
	int N_X = (*call_arrive_time).size();
	for (int i = 0; i < N_X; i++)
	{
		call call;
		call.number = i;
		call.state = 0;  //����
		call.time_arrive = (*call_arrive_time)[i];
		call.type = 'X';
		call_list.push_back(call);
	}
	return &call_list;
};

vector<call>* generate_call_Y(vector<int>* call_arrive_time) //���ݵ���ʱ�����������Y_callʵ������
{
	static vector<call> call_list;   //����call����
	int N_Y = (*call_arrive_time).size();
	for (int i = 0; i < N_Y; i++)
	{
		call call;
		call.number = i;
		call.state = 0;  //����
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
}; //����һ��·�ɻ����Ա�Ĵ���ʱ��


vector<vector<int>>* generate_call_arrive_time(double lambda1, double lambda2, int numX, int numY)
{//�������Ӳ���Ϊlambda��ָ���ֲ��ĺ��е���ʱ������
	srand(time(NULL));
	unsigned long init[4] = { rand(), 0x534, 0x345, 0x456 }, length = 4;
	init_by_array(init, length);

	vector<int> arrive_time_X;
	vector<int> arrive_time_Y;
	static vector<vector<int>> arrive_time_all;
	int x_t = 0;
	int y_t = 0;
	for (int i = 0; i < numX; i++) //����numX������
	{
		double u = genrand_real3();
		double x = -(1.0 / (double)lambda1) * log(u);
		x_t = x_t + (int)(x * 20);
		arrive_time_X.push_back(x_t);
	}
	for (int i = 0; i < numY; i++) //����numY������
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

//��a��b����call�������յ���ʱ��arrive_time����һ����������
//1.aʼ����ǰ��
//2.bʼ����ǰ��
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
	//��һ�У������ǰ�׶�
	rc_row.push_back(TYPE);

	//�ڶ��У������ǰʱ��
	rc_row.push_back(to_string(clock));

	//�����У����·�ɶ���
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
		rc_row.push_back("��");

	//�����У����·��״̬
	if (router.state == 0)
		rc_row.push_back("����");
	else
	{
		string router_call_now;
		string type(1, router.call_now.type);
		string number = to_string(router.call_now.number + 1);
		router_call_now.append(type);
		router_call_now.append(number);
		rc_row.push_back(router_call_now);
	}

	//�����У��������Ա1����
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
		rc_row.push_back("��");

	//�����У��������Ա1״̬
	if (receiver[0].state == 0)
		rc_row.push_back("����");
	else
	{
		string receiver_X_call_now;
		string type(1, receiver[0].call_now.type);
		string number = to_string(receiver[0].call_now.number + 1);
		receiver_X_call_now.append(type);
		receiver_X_call_now.append(number);
		rc_row.push_back(receiver_X_call_now);
	}

	//�����У��������Ա2����
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
		rc_row.push_back("��");

	//�ڰ��У��������Ա2״̬
	if (receiver[1].state == 0)
		rc_row.push_back("����");
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
	vector<string> cc = { "�׶�","��ǰʱ��","·�ɶ���","·��","����Ա1����","����Ա1","����Ա2����","����Ա2" };  //������
	Table my_table(cc, rc);
	my_table.Print_Table();

	cout << "��ǰ�����X���и�����" << number_finish_X << endl;
	cout << "��ǰ�����Y���и�����" << number_finish_Y << endl;
}