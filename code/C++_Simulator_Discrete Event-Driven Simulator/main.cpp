#include "main.h"
#include <math.h>
#include <iostream>
#include<algorithm>
using namespace std;

int main()
{
	int time_end = 160;

	//��ʼ��
	//��������ʵ��

	vector<vector<int>>* arrive_time_all = generate_call_arrive_time(0.5, 0.5, 5, 5);
	vector<int> call_arrive_time_X = (*arrive_time_all)[0];
	vector<int> call_arrive_time_Y = (*arrive_time_all)[1];
	int N_X = call_arrive_time_X.size();
	int N_Y = call_arrive_time_Y.size();

	//��������ʵ��
	vector<call>* X_call = generate_call_X(&call_arrive_time_X);
	vector<call>* Y_call = generate_call_Y(&call_arrive_time_Y);
	//�������ϲ����õ�һ��ʵ���modeΪ�ϲ�����
	vector<call> call_all = merge(X_call, Y_call, N_X, N_Y, 2); //��ʱ�Ѿ�����ʱ��˳�����к���


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

	//��������ʵ��
	vector<call>* X_call;
	X_call = generate_call_X(&call_arrive_time_X);
	vector<call>* Y_call;
	Y_call = generate_call_Y(&call_arrive_time_Y);

	//�������ϲ����õ�һ��ʵ���
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


	router router;  //����·����ʵ��
	router.state = 0;
	router.time_work = 0;
	router.call_now = { 0,'Z',0 ,0 };

	receiver receiver[2];  //��������Աʵ��
	receiver[0].state = 0;  //����Ա1
	receiver[0].time_work = 0;
	receiver[0].call_now = { 0,'Z',0 ,0 };
	receiver[1].state = 0;  //����Ա2
	receiver[1].time_work = 0;
	receiver[1].call_now = { 0,'Z',0 ,0 };

	//��ʼ����
	int clock = 0;

	//��ʼ��
	vector<call> router_list;  //·�ɶ���
	vector<call> receiver_X_list;  //����Ա1����
	vector<call> receiver_Y_list;  //����Ա2����

	//ͳ������
	int number_finish_X = 0;  //��ɵ�X���н�����
	int number_finish_Y = 0;  //��ɵ�Y���н�����
	int sum_wait_time = 0;  //���к��еȴ�����ʱ��

	//����B�¼��б�
	struct Node h; //���������ͷ��㣨��ֹ֮�����NULL��
	h.call = { 0,'Z',0 ,0 };
	h.next = NULL;
	h.NType = 0;
	h.Occurtime = 0; //�������
	head = &h;

	struct Node* p;//����p��㣨β��㣩
	p = head;

	int n = call_all.size();  //������
	for (int i = 0; i < n; i++)  //ѭ���������
	{
		struct Node* s = (struct Node*)malloc(sizeof(struct Node));//����s��㣬�������ڴ�
		//��s��㸳ֵ
		s->call = call_all[i];
		if (call_all[i].type == 'X')	s->NType = 1;   //X���е��ﲢ����·�ɶ���
		if (call_all[i].type == 'Y')	s->NType = 2;   //Y���е��ﲢ����·�ɶ���
		s->Occurtime = call_all[i].time_arrive;  //����ʱ�伴�¼�����ʱ��

		//β�巨��������
		p->next = s;  //��ͷ���������ոճ�ʼ���Ľ��
		s->next = NULL;  //�ò���β����s����ָ��ָ��NULL
		p = s;  //p��㱣��ղŵ�s��㣬�Ա�֤pʼ��Ϊ��������һ���ڵ�
	}

	struct Node end;   //������������¼���β���
	end.call = { 0,'Z',0 ,0 };
	end.next = NULL;
	end.NType = 6;
	end.Occurtime = time_end; //�������

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
	s = head->next;  //ͷ�����һ�����
	vector<vector<string>> rc_0;
	vector<string> rc_row_0 = create_rc_row("��ʼ��", clock, call_all, router, receiver, router_list, receiver_X_list, receiver_Y_list);
	rc_0.push_back(rc_row_0);
	my_print(rc_0, number_finish_X, number_finish_Y);

	while (s != NULL && s->Occurtime <= time_end)
	{
		clock = s->Occurtime;  //�ƽ������Ҫ�������¼���ʱ��
		//�γ����ʱ��Ҫ������B�¼��б�����ͷ�������¼��ķ���ʱ��
		while (s != NULL && s->Occurtime == clock)
		{
			if (s->NType == 1 || s->NType == 2)  //B1�¼���X��Y���е���·��
				router_list.push_back(s->call);
			if (s->NType == 3)  //B3�¼���·����ɹ��������X������Ա1���У�Y������Ա2����
			{
				router.state = 0;
				if (s->call.type == 'X')  receiver_X_list.push_back(s->call);
				if (s->call.type == 'Y')  receiver_Y_list.push_back(s->call);
			}
			if (s->NType == 4)  //B4�¼�,����Ա1��ɹ�������ɽ�������1��
			{
				receiver[0].state = 0;
				number_finish_X++;
			}
			if (s->NType == 5)  //B5�¼�,����Ա2��ɹ�������ɽ�������1��
			{
				receiver[1].state = 0;
				number_finish_Y++;
			}
			//�ı�״̬���ɾ�������㣬ʹsָ���µ�ͷ���
			head->next = s->next;
			s = head->next;
		}
		vector<vector<string>> rc;
		vector<string> rc_row_B = create_rc_row("B�׶�", clock, call_all, router, receiver, router_list, receiver_X_list, receiver_Y_list);
		rc.push_back(rc_row_B);

		//��ʼִ��C�¼�

		//·�ɰѵ绰��·�ɶ�����ȡ������ʼ��������C1
		if (!router_list.empty())
		{
			int work_time = generate_work_time(1, 1);
			int n = router_list.size();
			for (int i = 0; i < n; i++)
			{
				if (router.state == 0)  //���·�ɿ���
				{
					router.call_now = router_list[i];  //�Ѹú��н���·�ɴ���
					//�޸ĺ��С�·�ɵĲ���
					router_list[i].state = 1;
					router.state = 1;

					//·������ʼ����
					router.time_work = router.time_work + work_time;

					//������һ��B�¼�����B3
					struct Node in;
					in.NType = 3;
					in.call = router_list[i];
					in.Occurtime = clock + work_time;
					in.next = NULL;

					//����Ҫ����Ľ���occurtime�Ѹý���������ĺ���λ��
					struct Node* s;//����s���
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
					router_list.erase(router_list.begin());  //����Ѿ�������ˣ��Ͱ�������д�·�ɵȴ�������ɾ��
				}
				else //���·��æµ��,δ�ɹ����������call��Ҫ�����ڵȴ�������
				{
					sum_wait_time = sum_wait_time + work_time;
				}
			}
		}

		//����Ա1�ѵ绰�ӽ���Ա1�Ķ�����ȡ������ʼ��������C2
		if (!receiver_X_list.empty())
		{
			int work_time = generate_work_time(1, 1);
			int nx = receiver_X_list.size();
			for (int i = 0; i < nx; i++)
			{
				if (receiver[0].state == 0)  //�������Ա����
				{
					receiver[0].call_now = receiver_X_list[i];  //�Ѹú��н���·�ɴ���
					//�޸ĺ��С�·�ɵĲ���
					receiver_X_list[i].state = 1;
					receiver[0].state = 1;

					//·������ʼ����
					receiver[0].time_work = receiver[0].time_work + work_time;

					//������һ��B�¼�����B4
					struct Node in;
					in.NType = 4;
					in.call = receiver_X_list[i];
					in.Occurtime = clock + work_time;
					in.next = NULL;
					//����Ҫ����Ľ���occurtime�Ѹý���������ĺ���λ��
					struct Node* s;//����s��㣬�������ڴ�
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
					receiver_X_list.erase(receiver_X_list.begin());  //����Ѿ�������ˣ��Ͱ�������дӽ���Ա1�ȴ�������ɾ��
				}
				else //���·��æµ��,δ�ɹ����������call��Ҫ�����ڵȴ�������
				{
					sum_wait_time = sum_wait_time + work_time;
				}
			}
		}

		//����Ա2�ѵ绰�ӽ���Ա2�Ķ�����ȡ������ʼ��������C3
		if (!receiver_Y_list.empty())
		{
			int work_time = generate_work_time(7, 1);
			int ny = receiver_Y_list.size();
			for (int i = 0; i < ny; i++)
			{
				if (receiver[1].state == 0)  //�������Ա����
				{
					receiver[1].call_now = receiver_Y_list[i];  //�Ѹú��н���·�ɴ���
					//�޸ĺ��С�·�ɵĲ���
					receiver_Y_list[0].state = 1;
					receiver[1].state = 1;

					//·������ʼ����
					receiver[1].time_work = receiver[1].time_work + work_time;

					//������һ��B�¼�����B4
					struct Node in;
					in.NType = 5;
					in.call = receiver_Y_list[i];
					in.Occurtime = clock + work_time;
					in.next = NULL;
					//����Ҫ����Ľ���occurtime�Ѹý���������ĺ���λ��
					struct Node* s;//����s��㣬�������ڴ�
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
					receiver_Y_list.erase(receiver_Y_list.begin());  //����Ѿ�������ˣ��Ͱ�������дӽ���Ա2�ȴ�������ɾ��
				}
				else //���·��æµ��,δ�ɹ����������call��Ҫ�����ڵȴ�������
				{
					sum_wait_time = sum_wait_time + work_time;
				}
			}
		}
		s = head->next;
		vector<string> rc_row_C = create_rc_row("C�׶�", clock, call_all, router, receiver, router_list, receiver_X_list, receiver_Y_list);
		rc.push_back(rc_row_C);
		my_print(rc, number_finish_X, number_finish_Y);
	}
	if (number_finish_X + number_finish_Y == 0)
		cout << "ƽ��ÿ�����еȴ�ʱ��Ϊ��" << "δ���" << endl;
	else
	{
		double average_wait_time = (double)sum_wait_time / ((double)number_finish_X + (double)number_finish_Y);
		cout << "ƽ��ÿ�����еȴ�ʱ��Ϊ��" << average_wait_time << endl;
	}
}
