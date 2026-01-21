% 复现：FedADMM-InSa: An inexact and self-adaptive ADMM for federated learning（2025最新发表）
% FedADMM、FedADMM-In、FedADMM-InSa算法
% Code Author：方吉翔

% 设置随机种子
rng(86);

% 设置图片读取路径
image_folder = "mnist_jpg/train";

% 参数设置
E = 50;
B = 64;
C = 0.2;           % 可能是问题的关键 调大可以实现 明天保持这么大 把其他的地方改成FedADMM试一下
K = 100;           % K个clients
beta = 1;          % 固定二次惩罚项
delta = 0.2;       % memory
c = 0.00003;
sigma = sqrt(2) / (sqrt(2) + sqrt(beta / c));  % 误差系数
mu = 5;
tau = 2;
inputNums = 784;
outputNums = 10;
hideNums = 250;
eta = 0.001;
max_round = 200;
m = max(C * K, 1);
lambda = zeros(inputNums * hideNums + hideNums * outputNums, K);  % 拉格朗日乘子项
% 这里表示 K列 代表的是K个lambda 每一个客户端都有一个列向量

% 数据输入
% [train_data, train_labels, num_images] = ReadMNIST(image_folder);  % 将图片全部转化为数字来存储
% [clients] = Distribution(train_data, train_labels, K);  % 随机分配给clients（Non-IID）

load('non_iid_client_data.mat');  % 已经读取并随机分配完成，可以直接用

% Xavier初始化全局权值
global_v = randn(inputNums, hideNums) * sqrt(2/(inputNums + hideNums));
global_w = randn(hideNums, outputNums) * sqrt(2/(hideNums + outputNums));
global_z = [global_v(:); global_w(:)];
global_z_last = global_z;
u_k_last = global_z;

u_trained = cell(1, K);
for i = 1:K
    u_trained{i} = global_z;
end

loss = zeros(1, m);
loss_all_insa = zeros(1, max_round);

for round = 1:max_round
    % 随机选择m个客户端
    selected_clients = randperm(K, m);
    % disp(selected_clients);
    
    % 本地更新
    for idx = 1:m
        k = selected_clients(idx);  % 获取实际客户端索引
        fprintf(['\n===========' ...
        '============== 开始训练客户端 %d =============' ...
        '================\n'], k);
        
        in_k = clients(k).data;  % 二维矩阵 每一行代表一张图片 行数代表第k个客户端中的数据量
        expectout_k = clients(k).labels;
    
        % 调用训练函数
        [u_k, avg_error, num_epochs_used] = ClientUpdate(...
            inputNums, hideNums, outputNums,...
            E, B, eta, sigma,...
            in_k, expectout_k, ...
            global_z, k, ...
            lambda, beta);

        % 可以记录每个客户端实际使用的轮次
        % fprintf('Client %d used %d epochs\n', k, num_epochs_used);
        
        % 更新lambda
        lambda(:, k) = lambda(:, k) - beta * (u_k - global_z);

        % self_adaptive beta adjust
        p = beta * norm(u_k - u_k_last);  % 计算p
        d = norm(u_k - global_z);         % 计算d

        if d > mu * p
            beta = beta * tau;
            fprintf("case 1");
        elseif p > mu * d
            beta = beta / tau;
            fprintf("case 2");
        else
            fprintf("case 3");
        end

        u_k_last = u_k;
        
        % 存储训练后的权值
        u_trained{k} = u_k;  % 修改
        
        % 第k个client,E轮本地更新后的loss
        loss(idx) = avg_error / num_epochs_used;   % 修改！！！
    end
    
    % 模型聚合
    fprintf(['\n===========' ...
        '============== 开始模型聚合，第%d次 =============' ...
        '================\n'], round);
    [global_z] = ServerExecute(u_trained, clients,...
                K, beta, lambda, delta, global_z_last);
    global_z_last = global_z;
    
    % 总loss图绘制
    figure(1);
    loss_all_insa(round) = sum(loss);
    disp(loss_all_insa(round));
    plot(1: round, loss_all_insa(1: round), 'b-');
    xlabel('Communication rounds', 'FontSize', 12);
    ylabel('Loss', 'FontSize', 12);
end

% save('fedavg_loss.mat','loss_all_avg');
% save('fedadmm_loss.mat','loss_all');
% save('fedadmm_in_loss.mat','loss_all_in');
save('fedadmm_insa_loss.mat','loss_all_insa');

fprintf("------------Over!\n");
