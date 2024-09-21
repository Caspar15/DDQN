#ifndef DL_DQN_PACKET_SCHEDULER_H_ // 頭檔保護，防止重複包含此文件
#define DL_DQN_PACKET_SCHEDULER_H_

#include "downlink-packet-scheduler.h" // 引入下行套件調度器基底類
#include "DQNTrainer.h"                // 引進 DQN 訓練器

// 定義 DL_DQN_PacketScheduler 類，它繼承自 DownlinkPacketScheduler
class DL_DQN_PacketScheduler : public DownlinkPacketScheduler
{
public:
   // 帶參數的構造函數，使用 input_size 和 output_size 初始化 DQN
   DL_DQN_PacketScheduler(int input_size, int output_size);

   // 預設建構函數，預設使用 5 作為輸入和輸出大小
   DL_DQN_PacketScheduler();

   // 析構函數，釋放 DQNTrainer 的資源
   virtual ~DL_DQN_PacketScheduler();

   // 主調度函數，用於執行調度過程
   virtual void DoSchedule(void);

   // 計算調度指標，使用 DQN 的 Q 值作為調度的依據
   virtual double ComputeSchedulingMetric(RadioBearer *bearer, double spectralEfficiency, int subChannel);

   // 計算每個流的權重，並更新 DQN 模型
   void ComputeWeights();

private:
   DQNTrainer *m_dqnTrainer;    // DQN 訓練器的指針，用於呼叫訓練和前向傳播
   int train_step;              // 訓練步驟計數器，用於控制何時更新目標網絡
   int target_update_frequency; // 控制目標網路更新的頻率

   // 將流的狀態資訊轉換為張量，用於 DQN 模型輸入
   torch::Tensor GetStateTensor(RadioBearer *bearer, double spectralEfficiency, int subChannel);

   // 取得下一狀態的信息，並轉換為張量
   torch::Tensor GetNextStateTensor(RadioBearer *bearer);

   // 計算獎勵函數，根據吞吐量和延遲給予獎勵
   float ComputeReward(RadioBearer *bearer);
};

#endif /* DL_DQN_PACKET_SCHEDULER_H_ */