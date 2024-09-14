#include "dl-dqn-packet-scheduler.h"
#include "DQNTrainer.h"
#include "../mac-entity.h"
#include "../../packet/Packet.h"
#include "../../packet/packet-burst.h"
#include "../../../device/NetworkNode.h"
#include "../../../flows/radio-bearer.h"
#include "../../../protocolStack/rrc/rrc-entity.h"
#include "../../../flows/application/Application.h"
#include "../../../device/GNodeB.h"
#include "../../../protocolStack/mac/AMCModule.h"
#include "../../../phy/phy.h"
#include "../../../core/spectrum/bandwidth-manager.h"

// 初始化 DQN 模型訓練器 m_dqnTrainer，傳入 state 和 action 的空間維度、折扣因子(我設定 gamma = 0.99，網路上是 0.9-0.99 隨便調整，學習率 lr = 0.001，一樣可調整)
DL_DQN_PacketScheduler::DL_DQN_PacketScheduler(int input_size, int output_size)
    : m_dqnTrainer(input_size, output_size, 0.99, 0.001) // 初始化 DQN 訓練器
{
    SetMacEntity(nullptr);
    CreateFlowsToSchedule();
}

DL_DQN_PacketScheduler::~DL_DQN_PacketScheduler()
{
    Destroy(); // 清理資源
}

void
DL_DQN_PacketScheduler::DoSchedule()
{
    DEBUG_LOG_START_1(SIM_ENV_SCHEDULER_DEBUG)
    cout << "Start DL DQN-based packet scheduler for node "
                << GetMacEntity()->GetDevice()->GetIDNetworkNode() << endl;
    DEBUG_LOG_END

    UpdateAverageTransmissionRate();
    CheckForDLDropPackets();
    SelectFlowsToSchedule();
    ComputeWeights();

    if (GetFlowsToSchedule()->size() == 0)
        {

        }
    else
        {
        RBsAllocation();
        }

    StopSchedule();
    ClearFlowsToSchedule();
}

// 使用 GetStateTensor 取得當前 state，將 state 輸入到 DQN 模型中通過 select_action 選擇最優 action，然後將 action 作為調度度量返回。
double
DL_DQN_PacketScheduler::ComputeSchedulingMetric(RadioBearer* bearer, double spectralEfficiency, int subChannel)
{
    torch::Tensor state = GetStateTensor(bearer, spectralEfficiency, subChannel);

    // 使用 DQN 的 forward 方法取得 Q 值
    torch::Tensor q_values = m_dqnTrainer.forward(state); // 取第一個 Q 值做為調度度量
    double q_value = q_values[0].item<double>(); 

    return q_value; // 直接返回 Q 值為 metric
}

// 這部分就是計算權重的了，這次應該沒錯。
// 先遍歷所有可以調度的流，對每個流的 RadioBearer 對象計算 state (GetStateTensor) 和下一 state (GetNextStateTensor)，並通過 ComputeReward 計算 reward。
void
DL_DQN_PacketScheduler::ComputeWeights()
{
    DEBUG_LOG_START_1(SIM_ENV_SCHEDULER_DEBUG)
    cout << "Compute DQN Weights" << endl;
    DEBUG_LOG_END

    std::vector<Experience> replay_memory;
    for (auto flow : *GetFlowsToSchedule())
        {
        RadioBearer* bearer = flow->GetBearer();
        if (bearer->HasPackets())
            {
                double spectralEfficiency = bearer->GetSpectralEfficiency();
                int subChannel = 0; // 根據需要獲取正確正子信道信息

                torch::Tensor state = GetStateTensor(bearer, spectralEfficiency, subChannel);
                int action = m_dqnTrainer.select_action(state); // 選擇動作

                // 執行動作後，獲取下一狀態和獎勵
                // 在實際實現中，需要根據動作對環境進行更新，這裡簡化處理
                torch::Tensor next_state = GetNextStateTensor(bearer);
                float reward = ComputeReward(bearer);

                bool done = false; // 根據條件判斷是否為終止狀態

                Experience exp = {state, next_state, action, reward, done};
                replay_memory.push_back(exp);
            }
        }

    m_dqnTrainer.train(replay_memory); // 訓練 DQN 模型
}

torch::Tensor
DL_DQN_PacketScheduler::GetStateTensor(RadioBearer* bearer, double spectralEfficiency, int subChannel)
{
    // 標準化各項指標
    float normalized_HOL = static_cast<float>(bearer->GetHeadOfLinePacketDelay()) / max_HOL;
    float normalized_rate = static_cast<float>(bearer->GetAverageTransmissionRate()) / max_rate;
    float normalized_spectral_efficiency = static_cast<float>(spectralEfficiency) / max_spectral_efficiency;
    float normalized_subChannel = static_cast<float>(subChannel) / max_subChannel;
    float normalized_queue_size = static_cast<float>(bearer->GetQueueSize()) / max_queue_size;
    // state 表示：HOL 延遲、平均傳輸速率、光譜效率、子信到、對列大小。
    // 這部分都可以更改，看需要甚麼 state 去判斷。
    std::vector<float> state = {
        normalized_HOL,  // HOL 延遲
        normalized_rate, // 平均傳輸速率
        normalized_spectral_efficiency, // 光譜效率
        normalized_subChannel, // 子信道
        normalized_queue_size // 對列大小
    };
    
     return torch::tensor(state).unsqueeze(0);
     // 返回一個 torch::Tensor 對象，就是為了 DQN 模型輸入。
}

torch::Tensor
DL_DQN_PacketScheduler::GetNextStateTensor(RadioBearer* bearer)
{
    // 這部分都可以做更改，就是 action 的調整
    float next_HOL = static_cast<float>(bearer->GetHeadOfLinePacketDelay()) / max_HOL - 0.01f; // 假設 action 少了延遲
    float next_rate = static_cast<float>(bearer->GetAverageTransmissionRate()) / max_rate + 0.01f; // 假設 action 增加了速率
    float next_spectral_efficiency = static_cast<float>(bearer->GetSpectralEfficiency()) / max_spectral_efficiency;
    float next_subChannel = 0.0f;
    float next_queue_size = static_cast<float>(bearer->GetQueueSize()) / max_queue_size - 0.01f; // 假設對列大小減少
    
    std::vector<float> next_state = {
        next_HOL,  // 下一步 HOL 延遲
        next_rate, // 下一步的傳輸速率
        next_spectral_efficiency, // 假設光譜效率漢子信到保持不變
        next_subChannel,
        next_queue_size // 下一步的對列大小
    };
    
    return torch::tensor(next_state).unsqueeze(0);
}

float
DL_DQN_PacketScheduler::ComputeReward(RadioBearer* bearer)
{
    float normalized_rate = static_cast<float>(bearer->GetAverageTransmissionRate()) / max_rate;
    float normalized_HOL = static_cast<float>(bearer->GetHeadOfLinePacketDelay()) / max_HOL;
    float normalized_spectral_efficiency = static_cast<float>(bearer->GetSpectralEfficiency()) / max_spectral_efficiency;

    // 權重設定
    float rate_weight = 1.0f;
    float HOL_weight = -1.0f;
    float spectral_efficiency_weight = 1.0f;

    // 計算獎勵
    float reward = rate_weight * normalized_rate +
                   HOL_weight * normalized_HOL +
                   spectral_efficiency_weight * normalized_spectral_efficiency;

    // 添加懲罰或額外獎勵
    if (normalized_HOL > 0.8f) {
        reward -= 0.5f; // 懲罰高延遲
    }
    if (normalized_rate > 0.8f) {
        reward += 0.5f; // 獎勵高吞吐量
    }
    return reward;
}
