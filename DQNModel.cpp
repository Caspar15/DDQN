// fc1 到 fc4：這些層的初始化增加了神經網絡的深度，這可以使模型能夠學習到更複雜的特徵。就像是 CNN 之類的捲基層等等
// torch::relu：使用 ReLU（Rectified Linear Unit）作為激活函數，這是我增加的部分，這部分是深度學習中常用的激活函數，因為它可以在在計算上簡單地解決梯度消失問題。
// forward 函數：這個函數定義了神經網絡的前向傳播過程。每一層的輸出會被傳遞到下一層，直到得到最終的 Q 值輸出。.h 檔中我有定義，要更改可以從那裏去做更改

#include "DQNModel.h"

// 構造函數：定義神經網絡的層結構，這些層數未來都還可以再調整
DQN::DQN(int input_size, int output_size) {
    fc1 = register_module("fc1", torch::nn::Linear(input_size, 256)); // 第一層：從 input_size 到 256 個神經元
    fc2 = register_module("fc2", torch::nn::Linear(256, 128));        // 第二層：256 到 128 個神經元
    fc3 = register_module("fc3", torch::nn::Linear(128, 128));        // 第三層：128 到 128 個神經元
    fc4 = register_module("fc4", torch::nn::Linear(128, 64));         // 第四層：128 到 64 個神經元
    fc5 = register_module("fc5", torch::nn::Linear(64, output_size)); // 輸出層：64 到 output_size 個神經元

    dropout = register_module("dropout", torch::nn::Dropout(0.2)); // 加入 Dropout 層
}
// 前向傳播：數據如何流經每一層神經網絡 (這部分是 .h 裡面的前向傳播的函式 )
torch::Tensor DQN::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));  // ReLU 激活函數，增加非線性
    x = dropout->forward(x);
    x = torch::relu(fc2->forward(x));  // ReLU 激活函數，增加非線性
    x = dropout->forward(x);
    x = torch::relu(fc3->forward(x));  // ReLU 激活函數，增加非線性
    x = torch::relu(fc4->forward(x));  // ReLU 激活函數，增加非線性
    x = fc5->forward(x);  // 最後一層不使用激活函數，直接輸出 Q 值
    return x;
}

// DQN::DQN(int input_size, int output_size) {
//     fc1 = register_module("fc1", torch::nn::Linear(input_size, 512));
//     fc2 = register_module("fc2", torch::nn::Linear(512, 256));

//     // 定義價值流和優勢流的全連接層
//     value_stream_fc = register_module("value_stream_fc", torch::nn::Linear(256, 128));
//     advantage_stream_fc = register_module("advantage_stream_fc", torch::nn::Linear(256, 128));

//     value_output = register_module("value_output", torch::nn::Linear(128, 1));
//     advantage_output = register_module("advantage_output", torch::nn::Linear(128, output_size));

//     dropout = register_module("dropout", torch::nn::Dropout(0.2));
// }

// 前向傳播：數據如何流經每一層神經網路
// torch::Tensor DQN::forward(torch::Tensor x) {
//     x = torch::relu(fc1->forward(x));
//     x = dropout->forward(x);
//     x = torch::relu(fc2->forward(x));

//     // 價值流
//     auto value = torch::relu(value_stream_fc->forward(x));
//     value = value_output->forward(value);

//     // 優勢流
//     auto advantage = torch::relu(advantage_stream_fc->forward(x));
//     advantage = advantage_output->forward(advantage);

//     // 合併價值和優勢，計算最終的 Q 值
//     auto q_values = value + advantage - advantage.mean(1, true);
//     return q_values;
// }

// 保存模型權重
void DQN::save_weights(const std::string& file_path) {
    torch::serialize::OutputArchive archive;
    this->save(archive);
    archive.save_to(file_path);
}

// 加載模型權重
void DQN::load_weights(const std::string& file_path) {
    torch::serialize::InputArchive archive;
    archive.load_from(file_path);
    this->load(archive);
}