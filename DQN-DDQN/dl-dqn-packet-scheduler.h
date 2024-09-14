#ifndef DL_DQN_PACKET_SCHEDULER_H_
#define DL_DQN_PACKET_SCHEDULER_H_

#include "downlink-packet-scheduler.h"
#include "DQNTrainer.h"

class DL_DQN_PacketScheduler : public DownlinkPacketScheduler
{
public:
   DL_DQN_PacketScheduler(int input_size, int output_size);
   // 構造函數，用於初始化 DQN 模型訓練器 m_dqnTrainer。
   virtual ~DL_DQN_PacketScheduler();
   // 解構函數，釋放資源。
   virtual void DoSchedule(void);
   // 計算調度度量 (metric)，返回值為 double 類型。
   virtual double ComputeSchedulingMetric(RadioBearer* bearer, double spectralEfficiency, int subChannel);
   // 選擇要調度的部分。
   void ComputeWeights();

private:
   DQNTrainer m_dqnTrainer;
   // DQN 模型的訓練器，負責模型的訓練和推理。

   torch::Tensor GetStateTensor(RadioBearer* bearer, double spectralEfficiency, int subChannel);
   // 構造目前狀態的張量表示(DQN 內部的)，供 DQN 模型輸入。
   torch::Tensor GetNextStateTensor(RadioBearer* bearer);
   // 構造下一個狀態的張量表示，供 DQN 模型用於訓練。
   float ComputeReward(RadioBearer* bearer);
   // 計算當前 action 對應的獎勵值，用於訓練 DQN 模型。

   // 標準化的最大常值
   const float max_HOL = 1.0f; 
   const float max_rate = 1000000.0f; 
   const float max_spectral_efficiency = 7.0f; 
   const float max_subChannel = 100.0f; 
   const float max_queue_size = 1000.0f; 
};

#endif /* DL_DQN_PACKET_SCHEDULER_H_ */
