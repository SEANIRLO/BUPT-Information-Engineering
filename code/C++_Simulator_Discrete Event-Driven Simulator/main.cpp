#include "main.h"
#include <math.h>
#include <iostream>
#include<algorithm>
using namespace std;

int main()
{
	int time_end = 160;

	//初始化
	//创建呼叫实体

	vector<vector<int>>* arrive_time_all = generate_call_arrive_time(0.5, 0.5, 5, 5);
	vector<int> call_arrive_time_X = (*arrive_time_all)[0];
	vector<int> call_arrive_time_Y = (*arrive_time_all)[1];
	int N_X = call_arrive_time_X.size();
	int N_Y = call_arrive_time_Y.size();

	//创建呼叫实体
	vector<call>* X_call = generate_call_X(&call_arrive_time_X);
	vector<call>* Y_call = generate_call_Y(&call_arrive_time_Y);
	//把两个合并，得到一个实体表，mode为合并规则
	vector<call> call_all = merge(X_call, Y_call, N_X, N_Y, 2); //这时已经按照时间顺序排列好了


	/*
	int time_end=40;

	vector<int> call_arrive_time_X;
	call_arrive_time_X.push_back(5);
	call_arrive_time_X.push_back(10);
	call_arrive_time_X.push_back(15);
	call_arrive_time_X.push_back(20);
	int N_X = call_arrive_time_X.size();

	vector<int> call_arrive_time_Y;
	call_arrive_time_Y.push_back(5);
	call_arrive_time_Y.push_back(10);
	call_arrive_time_Y.push_back(15);
	call_arrive_time_Y.push_back(20);
	int N_Y = call_arrive_time_Y.size();

	//创建呼叫实体
	vector<call>* X_call;
	X_call = generate_call_X(&call_arrive_time_X);
	vector<call>* Y_call;
	Y_call = generate_call_Y(&call_arrive_time_Y);

	//把两个合并，得到一个实体表
	vector<call> call_all;
	call_all.push_back((*Y_call)[0]);  //Y1-5
	call_all.push_back((*X_call)[0]);  //X1-5
	call_all.push_back((*Y_call)[1]);  //Y2-10
	call_all.push_back((*X_call)[1]);  //X2-10
	call_all.push_back((*Y_call)[2]);  //Y3-15
	call_all.push_back((*X_call)[2]);  //X3-15
	call_all.push_back((*Y_call)[3]);  //Y4-20
	call_all.push_back((*X_call)[3]);  //X4-20

	*/


	router router;  //创建路由器实体
	router.state = 0;
	router.time_work = 0;
	router.call_now = { 0,'Z',0 ,0 };

	receiver receiver[2];  //创建接线员实体
	receiver[0].state = 0;  //接线员1
	receiver[0].time_work = 0;
	receiver[0].call_now = { 0,'Z',0 ,0 };
	receiver[1].state = 0;  //接线员2
	receiver[1].time_work = 0;
	receiver[1].call_now = { 0,'Z',0 ,0 };

	//开始仿真
	int clock = 0;

	//初始化
	vector<call> router_list;  //路由队列
	vector<call> receiver_X_list;  //接线员1队列
	vector<call> receiver_Y_list;  //接线员2队列

	//统计数据
	int number_finish_X = 0;  //完成的X呼叫接线数
	int number_finish_Y = 0;  //完成的Y呼叫接线数
	int sum_wait_time = 0;  //所有呼叫等待的总时间

	//创建B事件列表
	struct Node h; //创建链表的头结点（防止之后出现NULL）
	h.call = { 0,'Z',0 ,0 };
	h.next = NULL;
	h.NType = 0;
	h.Occurtime = 0; //仿真结束
	head = &h;

	struct Node* p;//创建p结点（尾结点）
	p = head;

	int n = call_all.size();  //链表长度
	for (int i = 0; i < n; i++)  //循环创建结点
	{
		struct Node* s = (struct Node*)malloc(sizeof(struct Node));//创建s结点，并分配内存
		//给s结点赋值
		s->call = call_all[i];
		if (call_all[i].type == 'X')	s->NType = 1;   //X呼叫到达并进入路由队列
		if (call_all[i].type == 'Y')	s->NType = 2;   //Y呼叫到达并进入路由队列
		s->Occurtime = call_all[i].time_arrive;  //到达时间即事件发生时间

		//尾插法建立链表
		p->next = s;  //在头结点后面插入刚刚初始化的结点
		s->next = NULL;  //让插在尾部的s结点的指针指向NULL
		p = s;  //p结点保存刚才的s结点，以保证p始终为链表的最后一个节点
	}

	struct Node end;   //创建仿真结束事件的尾结点
	end.call = { 0,'Z',0 ,0 };
	end.next = NULL;
	end.NType = 6;
	end.Occurtime = time_end; //仿真结束

	struct Node* tmp;
	tmp = head;
	while (tmp != NULL)
	{
		if (tmp->next == NULL || tmp->next->Occurtime > end.Occurtime)
		{
			end.next = tmp->next;
			tmp->next = &end;
			break;
		}
		tmp = tmp->next;
	}

	struct Node* s;
	s = head->next;  //头结点后第一个结点
	vector<vector<string>> rc_0;
	vector<string> rc_row_0 = create_rc_row("初始化", clock, call_all, router, receiver, router_list, receiver_X_list, receiver_Y_list);
	rc_0.push_back(rc_row_0);
	my_print(rc_0, number_finish_X, number_finish_Y);

	while (s != NULL && s->Occurtime <= time_end)
	{
		clock = s->Occurtime;  //推进到最近要发生的事件的时间
		//形成这个时刻要发生的B事件列表，即从头检查各个事件的发生时间
		while (s != NULL && s->Occurtime == clock)
		{
			if (s->NType == 1 || s->NType == 2)  //B1事件，X、Y呼叫到达路由
				router_list.push_back(s->call);
			if (s->NType == 3)  //B3事件，路由完成工作并输出X到接线员1队列，Y到接线员2队列
			{
				router.state = 0;
				if (s->call.type == 'X')  receiver_X_list.push_back(s->call);
				if (s->call.type == 'Y')  receiver_Y_list.push_back(s->call);
			}
			if (s->NType == 4)  //B4事件,接线员1完成工作（完成接线数加1）
			{
				receiver[0].state = 0;
				number_finish_X++;
			}
			if (s->NType == 5)  //B5事件,接线员2完成工作（完成接线数加1）
			{
				receiver[1].state = 0;
				number_finish_Y++;
			}
			//改变状态后就删除这个结点，使s指向新的头结点
			head->next = s->next;
			s = head->next;
		}
		vector<vector<string>> rc;
		vector<string> rc_row_B = create_rc_row("B阶段", clock, call_all, router, receiver, router_list, receiver_X_list, receiver_Y_list);
		rc.push_back(rc_row_B);

		//开始执行C事件

		//路由把电话从路由队列中取出并开始工作——C1
		if (!router_list.empty())
		{
			int work_time = generate_work_time(1, 1);
			int n = router_list.size();
			for (int i = 0; i < n; i++)
			{
				if (router.state == 0)  //如果路由空闲
				{
					router.call_now = router_list[i];  //把该呼叫交给路由处理
					//修改呼叫、路由的参数
					router_list[i].state = 1;
					router.state = 1;

					//路由器开始处理
					router.time_work = router.time_work + work_time;

					//产生下一个B事件——B3
					struct Node in;
					in.NType = 3;
					in.call = router_list[i];
					in.Occurtime = clock + work_time;
					in.next = NULL;

					//根据要插入的结点的occurtime把该结点插入链表的合适位置
					struct Node* s;//创建s结点
					s = head;
					while (s->next != NULL)
					{
						if (s->next == NULL || s->next->Occurtime > in.Occurtime)
						{
							in.next = s->next;
							s->next = &in;
							break;
						}
						s = s->next;
					}
					router_list.erase(router_list.begin());  //如果已经处理过了，就把这个呼叫从路由等待队列中删除
				}
				else //如果路由忙碌中,未成功处理，则这个call还要继续在等待队列中
				{
					sum_wait_time = sum_wait_time + work_time;
				}
			}
		}

		//接线员1把电话从接线员1的队列中取出并开始工作——C2
		if (!receiver_X_list.empty())
		{
			int work_time = generate_work_time(1, 1);
			int nx = receiver_X_list.size();
			for (int i = 0; i < nx; i++)
			{
				if (receiver[0].state == 0)  //如果接线员空闲
				{
					receiver[0].call_now = receiver_X_list[i];  //把该呼叫交给路由处理
					//修改呼叫、路由的参数
					receiver_X_list[i].state = 1;
					receiver[0].state = 1;

					//路由器开始处理
					receiver[0].time_work = receiver[0].time_work + work_time;

					//产生下一个B事件——B4
					struct Node in;
					in.NType = 4;
					in.call = receiver_X_list[i];
					in.Occurtime = clock + work_time;
					in.next = NULL;
					//根据要插入的结点的occurtime把该结点插入链表的合适位置
					struct Node* s;//创建s结点，并分配内存
					s = head;
					while (s->next != NULL)
					{
						if (s->next == NULL || s->next->Occurtime > in.Occurtime)
						{
							in.next = s->next;
							s->next = &in;
							break;
						}
						s = s->next;
					}
					receiver_X_list.erase(receiver_X_list.begin());  //如果已经处理过了，就把这个呼叫从接线员1等待队列中删除
				}
				else //如果路由忙碌中,未成功处理，则这个call还要继续在等待队列中
				{
					sum_wait_time = sum_wait_time + work_time;
				}
			}
		}

		//接线员2把电话从接线员2的队列中取出并开始工作——C3
		if (!receiver_Y_list.empty())
		{
			int work_time = generate_work_time(7, 1);
			int ny = receiver_Y_list.size();
			for (int i = 0; i < ny; i++)
			{
				if (receiver[1].state == 0)  //如果接线员空闲
				{
					receiver[1].call_now = receiver_Y_list[i];  //把该呼叫交给路由处理
					//修改呼叫、路由的参数
					receiver_Y_list[0].state = 1;
					receiver[1].state = 1;

					//路由器开始处理
					receiver[1].time_work = receiver[1].time_work + work_time;

					//产生下一个B事件——B4
					struct Node in;
					in.NType = 5;
					in.call = receiver_Y_list[i];
					in.Occurtime = clock + work_time;
					in.next = NULL;
					//根据要插入的结点的occurtime把该结点插入链表的合适位置
					struct Node* s;//创建s结点，并分配内存
					s = head;
					while (s->next != NULL)
					{
						if (s->next == NULL || s->next->Occurtime > in.Occurtime)
						{
							in.next = s->next;
							s->next = &in;
							break;
						}
						s = s->next;
					}
					receiver_Y_list.erase(receiver_Y_list.begin());  //如果已经处理过了，就把这个呼叫从接线员2等待队列中删除
				}
				else //如果路由忙碌中,未成功处理，则这个call还要继续在等待队列中
				{
					sum_wait_time = sum_wait_time + work_time;
				}
			}
		}
		s = head->next;
		vector<string> rc_row_C = create_rc_row("C阶段", clock, call_all, router, receiver, router_list, receiver_X_list, receiver_Y_list);
		rc.push_back(rc_row_C);
		my_print(rc, number_finish_X, number_finish_Y);
	}
	if (number_finish_X + number_finish_Y == 0)
		cout << "平均每个呼叫等待时间为：" << "未完成" << endl;
	else
	{
		double average_wait_time = (double)sum_wait_time / ((double)number_finish_X + (double)number_finish_Y);
		cout << "平均每个呼叫等待时间为：" << average_wait_time << endl;
	}
}
